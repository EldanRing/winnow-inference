"""Routing choices, legacy configuration compatibility and guarded always reasoning."""

import argparse
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assets
import decision_client
import setup
import winnow
from adaptive_policy import DecisionPipeline, load_policy, softmax
from test_adaptive_pipeline import FakeTransport, body, generation, identity, native


POLICIES = tuple(json.loads((ROOT / "manifests/adaptive-v1.json").read_text())["policies"])


def probabilities(answer):
    return [1 - answer["noul"], answer["noul"]] if answer["type"] == "noul" else list(answer["probabilities"].values())


class ReasoningModes(unittest.TestCase):
    def test_always_bypasses_both_gate_types_and_keeps_each_models_calibrated_blend(self):
        for policy_id in POLICIES:
            original = copy.deepcopy(load_policy(policy_id)[1])
            policy = original["policy"]
            for kind in ("choice", "noul", "score"):
                with self.subTest(policy=policy_id, kind=kind):
                    request = body(kind)
                    direct_logits = [0, 1000] if kind != "score" else [0, 1000, 0]
                    augmented_logits = [2, 0] if kind != "score" else [2, 0, 1]
                    # Raw confidence exactly 1 / entropy 0 must still route in always mode.
                    for reasoning in ("on", "selective", "always"):
                        replies = [identity(policy_id), native(request, direct_logits, policy_id)]
                        if reasoning == "always":
                            replies += [generation(), native(request, augmented_logits, policy_id)]
                        transport = FakeTransport(replies)
                        out = DecisionPipeline(transport, policy_id=policy_id, reasoning=reasoning).decide(request)
                        meta = out["winnow"]["adaptive"]
                        self.assertEqual(meta["routed"], reasoning == "always")
                        self.assertEqual(meta["gate_applied"], reasoning != "always")
                        self.assertEqual(meta["reasoning_mode"], "selective" if reasoning == "on" else reasoning)
                        self.assertEqual(meta["threshold"], policy["threshold"])
                        self.assertEqual(meta["blend_weight"], policy["weight"])
                        expected = softmax(direct_logits, policy["direct_temperature"])
                        if reasoning == "always":
                            reasoned = softmax(augmented_logits, policy["augmented_temperature"])
                            expected = [(1 - policy["weight"]) * a + policy["weight"] * b for a, b in zip(expected, reasoned)]
                            self.assertEqual(transport.calls[2][1]["max_tokens"], -1)
                            self.assertFalse(transport.calls[2][1]["ignore_eos"])
                            self.assertEqual(transport.calls[2][2], 75)
                            self.assertIn("Always reasoning", meta["configuration_note"])
                        for actual, value in zip(probabilities(out["answers"]["unusual/id"]), expected):
                            self.assertAlmostEqual(actual, value)
                        self.assertFalse(transport.replies)
            self.assertEqual(load_policy(policy_id)[1], original)

    def test_legacy_constructor_and_on_alias_keep_selective_results(self):
        policy = "e4b-calibrated75-g95-v1"
        request = body()
        results = []
        for options in ({"mode": "experimental-adaptive"}, {"reasoning": "on"}, {"reasoning": "selective"}):
            transport = FakeTransport([identity(policy), native(request, [0, 0], policy),
                                       generation(), native(request, [1, 0], policy)])
            results.append(DecisionPipeline(transport, policy_id=policy, **options).decide(request))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        for options in ({}, {"mode": "direct"}, {"reasoning": "off"}):
            response = {"untouched": True}
            transport = FakeTransport([response])
            self.assertEqual(DecisionPipeline(transport, **options).decide(request), response)
            self.assertEqual([call[0] for call in transport.calls], ["/v1/systemone"])
        for options in ({"mode": "direct", "reasoning": "always", "policy_id": policy},
                        {"mode": "experimental-adaptive", "reasoning": "off", "policy_id": policy},
                        {"reasoning": "always"}, {"reasoning": "off", "policy_id": policy},
                        {"reasoning": "unknown", "policy_id": policy}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DecisionPipeline(FakeTransport([]), **options)

    def test_always_preserves_identity_and_direct_error_checks_before_generation(self):
        policy = "e4b-calibrated75-g95-v1"
        for field in ("runtime_sha256", "target_sha256", "assistant_sha256"):
            wrong = identity(policy)
            wrong["runtime"][field] = "wrong"
            transport = FakeTransport([wrong])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Backend"):
                DecisionPipeline(transport, policy_id=policy, reasoning="always").decide(body())
            self.assertEqual(len(transport.calls), 1)
        transport = FakeTransport([identity(policy), OSError("direct failed")])
        with self.assertRaises(OSError):
            DecisionPipeline(transport, policy_id=policy, reasoning="always").decide(body())
        self.assertEqual(len(transport.calls), 2)
        transport = FakeTransport([])
        request = body()
        request["winnow"] = {"images": ["fixture"]}
        with self.assertRaises(ValueError):
            DecisionPipeline(transport, policy_id=policy, reasoning="always").decide(request)
        self.assertFalse(transport.calls)

    def test_always_keeps_calibrated_fallback_on_timeout_context_limit_and_augmented_failure(self):
        policy_id = "e4b-calibrated75-g95-v1"
        policy = load_policy(policy_id)[1]["policy"]
        request, logits = body(), [0, 4]
        failures = [[TimeoutError("deadline")], [generation(finish="length", stop="limit")],
                    [generation(content="")], [generation(), OSError("augmented failed")]]
        for tail in failures:
            with self.subTest(tail=repr(tail)):
                transport = FakeTransport([identity(policy_id), native(request, logits, policy_id), *tail])
                out = DecisionPipeline(transport, policy_id=policy_id, reasoning="always").decide(request)
                meta = out["winnow"]["adaptive"]
                self.assertTrue(meta["routed"])
                self.assertFalse(meta["completed_blend"])
                self.assertTrue(meta["fallback_reason"])
                self.assertEqual(probabilities(out["answers"]["unusual/id"]), softmax(logits, policy["direct_temperature"]))
                self.assertFalse(transport.replies)

    def test_always_with_mtp_off_keeps_identity_and_speculation_contract(self):
        policy = "e4b-calibrated75-g95-v1"
        request = body()
        ident = identity(policy)
        ident["runtime"].update(resident_mtp=False, assistant_sha256=None)
        direct, augmented = native(request, [0, 10], policy), native(request, [2, 0], policy)
        for response in (direct, augmented):
            response["winnow"]["resident_speculative_chat"] = False
        transport = FakeTransport([ident, direct, generation(), augmented])
        out = DecisionPipeline(transport, policy_id=policy, reasoning="always", mtp="off").decide(request)
        self.assertEqual(out["winnow"]["adaptive"]["mtp"], "off")
        self.assertTrue(out["winnow"]["adaptive"]["completed_blend"])

    def test_cli_forwards_modes_and_existing_user_options_without_migrating_files(self):
        for mode in ("off", "on", "selective", "always"):
            args = ["winnow", "decide", "--model", "e4b", "--reasoning", mode,
                    "--mtp", "on", "--input", "owner-request.json", "--base-url", "http://localhost:9999"]
            with self.subTest(mode=mode), patch.object(sys, "argv", args), patch.object(winnow.subprocess, "run") as run:
                winnow.main()
            command = run.call_args.args[0]
            self.assertEqual(command[-4:], ["--input", "owner-request.json", "--base-url", "http://localhost:9999"])
            if mode == "off":
                self.assertNotIn("--policy", command)
            else:
                self.assertEqual(command[command.index("--reasoning") + 1], "selective" if mode == "on" else mode)
                self.assertEqual(command[command.index("--policy") + 1], "e4b-calibrated75-g95-v1")
                self.assertEqual(command[command.index("--mtp") + 1], "on")

    def test_lower_level_cli_accepts_new_modes_and_old_mode_flags(self):
        policy = "e4b-calibrated75-g95-v1"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "request.json"
            request = body()
            path.write_text(json.dumps(request))
            original = path.read_bytes()
            for flags, routes in ((["--reasoning", "always"], True),
                                  (["--reasoning", "selective"], False),
                                  (["--reasoning", "on"], False),
                                  (["--mode", "experimental-adaptive"], False)):
                replies = [identity(policy), native(request, [0, 1000], policy)]
                if routes:
                    replies += [generation(), native(request, [2, 0], policy)]
                transport = FakeTransport(replies)
                argv = ["decide", "--input", str(path), "--policy", policy, *flags]
                stdout = io.StringIO()
                with self.subTest(flags=flags), patch.object(sys, "argv", argv), patch.object(decision_client, "HTTPTransport", return_value=transport), patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", io.StringIO()):
                    decision_client.main()
                self.assertEqual(json.loads(stdout.getvalue())["winnow"]["adaptive"]["routed"], routes)
            self.assertEqual(path.read_bytes(), original)

    def test_modes_reuse_identical_server_profile_and_required_assets(self):
        for model in assets.MODELS:
            for mtp in ("on", "off"):
                commands = []
                for mode in ("on", "selective", "always"):
                    spec, kinds = assets.selection(model, mode, mtp, "off")
                    self.assertEqual(kinds, ["model", "assistant"] if mtp == "on" else ["model"])
                    a = argparse.Namespace(model=model, reasoning=mode, mtp=mtp, vision="off", context=16384,
                                           model_dir=Path("/owner/models"), server=Path("/owner/server"))
                    with patch("winnow.platform.system", return_value="Linux"):
                        commands.append(winnow.serve_command(a))
                    with self.assertRaises(ValueError):
                        assets.selection(model, mode, mtp, "on")
                self.assertEqual(commands[0], commands[1])
                self.assertEqual(commands[0], commands[2])
                self.assertEqual(commands[0][commands[0].index("--context") + 1], "16384")
                self.assertIn("/owner/models/" + spec["model"]["file"], commands[0])

    def test_setup_migrates_on_alias_in_printed_commands_and_keeps_user_settings(self):
        for mode in ("on", "selective", "always"):
            argv = ["setup", "--model", "e4b", "--reasoning", mode, "--mtp", "on", "--context", "16k",
                    "--model-dir", "/owner/models", "--gpu", "1"]
            stdout = io.StringIO()
            with self.subTest(mode=mode), patch.object(sys, "argv", argv), patch.object(setup, "resolve_profile", return_value=("5070ti-64k", {})), patch.object(setup, "prerequisites", return_value=[]), patch.object(setup, "acquire") as acquire, patch.object(setup.subprocess, "run"), patch.object(sys, "stdout", stdout):
                self.assertEqual(setup.main(), 0)
            self.assertEqual(acquire.call_args.args[1], Path("/owner/models"))
            self.assertEqual(acquire.call_args.args[2], "selective" if mode == "on" else mode)
            self.assertIn("--context 16384 --gpu 1 --model-dir /owner/models", stdout.getvalue())
            self.assertIn("examples/adaptive-decision.json", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
