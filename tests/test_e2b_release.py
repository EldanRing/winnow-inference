"""E2B profile identity, CLI wiring and decision contracts, without inference."""
import argparse
import copy
import hashlib
import io
import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assets
import serve
import winnow
from adaptive_policy import DecisionPipeline, keys_for, load_policy, softmax
from reasoning_contract import load_profile
from test_adaptive_pipeline import FakeTransport, body, generation, native

POLICY = "e2b-raw99-blend50-v3"
IMAGES = ["data:image/png;base64,Zmlyc3Q=", "data:image/png;base64,c2Vjb25k"]


def inspection(profile, images=False):
    runtime = copy.deepcopy(profile["runtime"])
    runtime.update(labels=["A", "B", "C"], label_token_ids=[101, 102, 103])
    return dict(runtime=runtime, prefix_tokens=120,
                suffix_tokens=[30], image_tokens=42 if images else 0)


def scored(request, logits, mtp):
    result = native(request, logits, POLICY)
    result["winnow"]["resident_speculative_chat"] = mtp == "on"
    return result


def completed():
    result = generation()
    result["__verbose"]["truncated"] = False
    return result


class E2BRelease(unittest.TestCase):
    def test_profile_matrix_preserves_modes_images_and_probability_math(self):
        for context, vision, mtp, mode, kind in itertools.product(
                (8192, 65536), ("off", "on"), ("off", "on"),
                ("off", "selective", "always"), ("choice", "noul", "score")):
            with self.subTest(context=context, vision=vision, mtp=mtp, mode=mode, kind=kind):
                name = assets.decision_preset("e2b", context, vision, mtp)
                profile = load_profile(name)
                request = body(kind, {"records": [1, 2], "model_reasoning": "owner data"})
                if vision == "on":
                    request["winnow"] = {"images": copy.deepcopy(IMAGES)}
                original = copy.deepcopy(request)
                size = len(keys_for(request["questions"]["unusual/id"]))
                direct_logits, augmented_logits = [0] * size, list(range(size))
                replies = [inspection(profile, vision == "on"), scored(request, direct_logits, mtp)]
                if mode != "off":
                    replies += [completed(), inspection(profile, vision == "on"), scored(request, augmented_logits, mtp)]
                transport = FakeTransport(replies)
                result = DecisionPipeline(transport, reasoning=mode, mtp=mtp,
                    policy_id=POLICY if mode != "off" else None, runtime_profile=name).decide(request)
                self.assertFalse(transport.replies)
                self.assertEqual(request, original)
                meta = result["winnow"]["adaptive"]
                self.assertEqual(meta["completed_blend"], mode != "off")
                self.assertFalse(meta["measured_profile"])
                expected = softmax(direct_logits, 1)
                if mode != "off":
                    expected = [(a+b)/2 for a, b in zip(expected, softmax(augmented_logits, 1))]
                    generated = transport.calls[2][1]
                    self.assertEqual(generated["chat_template_kwargs"], {"enable_thinking": False})
                    self.assertEqual((generated["max_tokens"], generated["temperature"], generated["seed"]), (-1, 0, 314159))
                    self.assertEqual(transport.calls[2][2], 180)
                    self.assertEqual(transport.calls[4][2], 45)
                    if vision == "on":
                        parts = generated["messages"][0]["content"]
                        self.assertEqual([p["image_url"]["url"] for p in parts if p["type"] == "image_url"], IMAGES)
                        self.assertEqual(transport.calls[4][1]["winnow"]["images"], IMAGES)
                    self.assertEqual(transport.calls[4][1]["state"]["original_state"], request["state"])
                answer = result["answers"]["unusual/id"]
                values = [1-answer["noul"], answer["noul"]] if kind == "noul" else list(answer["probabilities"].values())
                for a, b in zip(values, expected):
                    self.assertAlmostEqual(a, b)

    def test_gate_boundary_always_and_incomplete_generation(self):
        profile = load_profile("e2b-q8-text8k-mtp")
        request = body()
        for mode, routed in (("off", False), ("selective", False), ("always", True)):
            responses = [inspection(profile), scored(request, [0, math.log(99)], "on")]
            if routed:
                responses.append(generation(finish="length", stop="limit"))
            transport = FakeTransport(responses)
            result = DecisionPipeline(transport, reasoning=mode, policy_id=POLICY if mode != "off" else None,
                                      runtime_profile=profile).decide(request)
            self.assertEqual(result["winnow"]["adaptive"]["routed"], routed)
            self.assertFalse(result["winnow"]["adaptive"]["completed_blend"])
            self.assertAlmostEqual(result["answers"]["unusual/id"]["probabilities"]["a-second"], .99)
            self.assertFalse(transport.replies)

    def test_old_runtime_and_wrong_profile_are_rejected_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "explicit runtime profile"):
            DecisionPipeline(FakeTransport([]), reasoning="always", policy_id=POLICY)
        profile = load_profile("e2b-q8-text8k-mtp")
        for field, value in (("runtime_sha256", "7a2ab005c30b8c4cc8d7e2f5e9512581e851c7d072e55959d7638f21f1866354"),
                             ("target_sha256", "0" * 64), ("cache_type", "q8_0"), ("context", 65536)):
            response = inspection(profile)
            response["runtime"][field] = value
            transport = FakeTransport([response])
            with self.assertRaisesRegex(ValueError, "Backend does not match"):
                DecisionPipeline(transport, reasoning="always", policy_id=POLICY, runtime_profile=profile).decide(body())
            self.assertEqual(len(transport.calls), 1)

    def test_generation_override_does_not_change_released_recipes(self):
        base = json.loads((ROOT / "manifests/adaptive-v1.json").read_text())
        e2b, _ = load_policy(POLICY)
        self.assertEqual(e2b["generation"]["prompt_format"], "e2b-native-labels-v3")
        for identifier in ("q8-fixed50-v1", "nvfp4-entropy-v1", "e4b-calibrated50-v1", "e4b-calibrated75-g95-v1"):
            manifest, _ = load_policy(identifier)
            self.assertEqual(manifest["generation"], base["generation"])

    def test_cli_server_and_client_select_same_profile(self):
        for context, vision, mtp in itertools.product((8192, 65536), ("off", "on"), ("off", "on")):
            expected = assets.decision_preset("e2b", context, vision, mtp)
            for mode in ("off", "selective", "always", "on"):
                args = argparse.Namespace(model="e2b", context=context, vision=vision, mtp=mtp,
                                          reasoning=mode, model_dir=Path("/owner/models"), server=None)
                with patch("winnow.platform.system", return_value="Linux"):
                    command = winnow.serve_command(args)
                self.assertEqual(command[command.index("--preset")+1], expected)
                argv = ["winnow", "decide", "--model", "e2b", "--context", str(context),
                        "--vision", vision, "--mtp", mtp, "--reasoning", mode, "--input", "request.json"]
                with patch.object(sys, "argv", argv), patch.object(winnow.subprocess, "run") as run:
                    winnow.main()
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--runtime-profile")+1], expected)
                self.assertEqual(command[command.index("--mtp")+1], mtp)
        with self.assertRaisesRegex(ValueError, "8k or 64k"):
            assets.decision_preset("e2b", 16384, "on", "on")

    def test_launcher_matches_inspection_contract_and_chat_thinking_is_separate(self):
        for name, preset in serve.RUNTIME_PRESETS.items():
            if not name.startswith("e2b-"):
                continue
            for thinking in ("on", "off"):
                with self.subTest(name=name, thinking=thinking), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    files = [root/n for n in ("target", "assistant", "projector", "server")]
                    for path in files:
                        path.touch()
                    mtp = "on" if preset["resident_mtp"] else "off"
                    argv = ["serve", "--preset", name, "--mtp", mtp, "--model", str(files[0]),
                            "--server", str(files[3]), "--native-chat-reasoning", thinking]
                    if mtp == "on": argv += ["--assistant", str(files[1])]
                    if not preset["text_only"]: argv += ["--mmproj", str(files[2])]
                    with patch.object(sys, "argv", argv), patch("serve.platform.system", return_value="Linux"), \
                         patch.object(serve, "verify") as verify, patch.object(serve.os, "execve") as launch, \
                         patch.object(sys, "stdout", io.StringIO()), patch.object(sys, "stderr", io.StringIO()):
                        serve.main()
                    command, env = launch.call_args.args[1:]
                    self.assertEqual(command[command.index("--reasoning")+1], thinking)
                    self.assertEqual(command[command.index("--cache-type-k")+1], "f16")
                    self.assertEqual(command[command.index("--samplers")+1], "temperature")
                    self.assertIn("--backend-sampling", command)
                    self.assertIn("--no-context-shift", command)
                    self.assertEqual(env["WINNOW_CONTEXT"], str(preset["settings"]["context"]))
                    self.assertEqual(env["WINNOW_CHAT_CONTEXT"], env["WINNOW_CONTEXT"])
                    self.assertEqual(env["WINNOW_PARALLEL"], "1")
                    self.assertEqual(env["WINNOW_RESIDENT_MTP"], "1" if mtp == "on" else "0")
                    self.assertEqual(verify.call_count, 1 + (mtp == "on") + (not preset["text_only"]))
                    if mtp == "on":
                        self.assertEqual(command[command.index("--spec-draft-n-max")+1], "4")
                        self.assertEqual(command[command.index("--spec-draft-type-k")+1], "q8_0")
                    else:
                        self.assertNotIn("--spec-draft-model", command)
                        self.assertNotIn("WINNOW_ASSISTANT_SHA256", env)

    def test_private_payload_never_attempts_network_and_exact_local_reuse_works(self):
        for kind in ("model", "projector", "assistant"):
            artifact = assets.MODELS["e2b-q8"][kind]
            if kind != "assistant":
                self.assertEqual(artifact["revision"], "7439a194a1a948262115b02fb380daf4ecc77369")
            self.assertNotEqual(artifact["availability"], "published")
        with tempfile.TemporaryDirectory() as td, patch.object(assets, "fetch") as fetch:
            with self.assertRaisesRegex(ValueError, "not published"):
                assets.acquire("e2b", Path(td), vision="off")
            fetch.assert_not_called()
            payload = b"pinned fixture"
            cache = Path(td)/"cache"
            cache.mkdir()
            (cache/"model.gguf").write_bytes(payload)
            registry = {"fixture": {"model": dict(file="gguf/model.gguf", bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), availability="private_staging_payload_pending")}}
            with patch.object(assets, "MODELS", registry):
                result = assets.acquire("fixture", Path(td)/"models", vision="off", asset_dirs=[cache])
            self.assertEqual(Path(result["model"]).read_bytes(), payload)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
