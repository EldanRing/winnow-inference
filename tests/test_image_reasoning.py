"""Image identity, context, cancellation and model-policy compatibility without inference."""

import argparse
import base64
import copy
import hashlib
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_policy import DecisionPipeline, keys_for, load_policy, rendered_options, softmax
from reasoning_contract import ContextLimitError, RequestCancelled, load_profile
from decision_client import HTTPTransport
import assets
import winnow
from test_adaptive_pipeline import FakeTransport, body, generation, native


POLICY = "e4b-calibrated75-g95-v1"
IMAGES = ["data:image/png;base64," + base64.b64encode(value).decode() for value in (b"first image", b"second image")]


def profile(policy=POLICY):
    p = load_profile("e4b-q8-vision8k-mtp")
    definition = load_policy(policy)[1]
    p.update(alias=definition["alias"])
    p["runtime"].update(target_sha256=definition["target"]["sha256"], assistant_sha256=definition["assistant"]["sha256"])
    return p


def inspection(p=None, prefix=100, suffix=20, images=True):
    return dict(runtime=copy.deepcopy((p or profile())["runtime"]), prefix_tokens=prefix,
                suffix_tokens=[suffix], image_tokens=42 if images else 0)


def image_body(kind="choice", state=None):
    result = body(kind, {"object": "compare image 1 and image 2", "model_reasoning": "original value"} if state is None else state)
    result["winnow"] = dict(images=copy.deepcopy(IMAGES), diagnostics=True, reuse_prefix=True)
    return result


def complete_generation():
    result = generation()
    result["__verbose"]["truncated"] = False
    return result


class ImageReasoning(unittest.TestCase):
    def test_local_prompt_preserves_native_candidate_keys_descriptions_and_order(self):
        cases = [
            ({"type": "choice", "criteria": {"red": None, "blue": None}}, ["red", "blue"]),
            ({"type": "choice", "criteria": {"z-last": "Warm color", "a-first": "Cool color"}},
             ["z-last: Warm color", "a-first: Cool color"]),
            ({"type": "choice", "criteria": {"plain": None, "named": "", "object": {"rgb": [1, 2]}, "array": ["red", "blue"]}},
             ["plain", "named: ", 'object: {"rgb":[1,2]}', 'array: ["red","blue"]']),
            ({"type": "noul"}, ["false", "true"]),
            ({"type": "noul", "criteria": {"true": "Same meaning", "false": "Different meaning"}},
             ["false: Different meaning", "true: Same meaning"]),
            ({"type": "noul", "criteria": {"true": {"answer": "yes"}}}, ["false", 'true: {"answer":"yes"}']),
            ({"type": "score", "criteria": [None, "high", {"rating": 2}, ["top"]]},
             ["0", "high", '{"rating":2}', '["top"]']),
            ({"type": "choice", "criteria": {f"candidate-{i}": None for i in range(64)}},
             [f"candidate-{i}" for i in range(64)]),
        ]
        manifest, _ = load_policy(POLICY)
        manifest["generation"]["prompt_format"] = "e2b-canonical-v2"
        for question, expected in cases:
            for images in (False, True):
                question = dict(question, instructions="Select the best candidate by its name and description.")
                keys = keys_for(question)
                self.assertEqual(rendered_options(question, keys), expected)
                request = image_body() if images else body()
                request["questions"] = {"unusual/id": question}
                logits = [0] * len(keys)
                transport = FakeTransport([inspection(images=images), native(request, logits, POLICY), complete_generation(),
                                           inspection(images=images), native(request, logits, POLICY)])
                output = DecisionPipeline(transport, reasoning="always", policy_id=POLICY,
                                          policy_manifest=manifest, runtime_profile=profile()).decide(request)
                self.assertTrue(output["winnow"]["adaptive"]["completed_blend"])
                content = transport.calls[2][1]["messages"][0]["content"]
                prompt = content[-1]["text"] if images else content
                expected_lines = "\n".join(f"{i+1}. {json.dumps(v, ensure_ascii=False)}" for i, v in enumerate(expected))
                self.assertEqual(prompt.split("\nOptions (in candidate order):\n")[1], expected_lines)
                self.assertEqual(transport.calls[1][1]["questions"], request["questions"])
                self.assertEqual(transport.calls[4][1]["questions"], request["questions"])
        # The old staged private format is rejected so it cannot silently retain the defect.
        manifest["generation"]["prompt_format"] = "e2b-v1"
        with self.assertRaises(ValueError):
            DecisionPipeline(FakeTransport([]), reasoning="always", policy_id=POLICY,
                             policy_manifest=manifest, runtime_profile=profile())

    def test_every_policy_and_kind_preserves_images_positions_and_calibrated_math(self):
        for policy in json.loads((ROOT / "manifests/adaptive-v1.json").read_text())["policies"]:
            for kind in ("choice", "noul", "score"):
                request, p = image_body(kind), profile(policy)
                saved = copy.deepcopy(request)
                n = len(keys_for(request["questions"]["unusual/id"]))
                d, a = [0] * n, list(range(n))
                transport = FakeTransport([inspection(p), native(request, d, policy), complete_generation(),
                                           inspection(p, prefix=130), native(request, a, policy)])
                output = DecisionPipeline(transport, reasoning="selective", policy_id=policy, runtime_profile=p).decide(request)
                self.assertEqual(request, saved)
                self.assertEqual([x[0] for x in transport.calls], ["/v1/winnow/inspect", "/v1/systemone",
                                 "/v1/chat/completions", "/v1/winnow/inspect", "/v1/systemone"])
                for i in (0, 1, 3, 4):
                    self.assertEqual(transport.calls[i][1]["winnow"]["images"], IMAGES)
                    self.assertFalse(transport.calls[i][1]["winnow"]["reuse_prefix"])
                content = transport.calls[2][1]["messages"][0]["content"]
                self.assertEqual([part["image_url"]["url"] for part in content if part["type"] == "image_url"], IMAGES)
                self.assertEqual(content[-1]["type"], "text")  # Images precede state, as in the native compiler.
                self.assertEqual(transport.calls[4][1]["state"]["original_state"], saved["state"])
                meta = output["winnow"]["adaptive"]
                self.assertEqual(meta["image_sha256"], [hashlib.sha256(v).hexdigest() for v in (b"first image", b"second image")])
                self.assertEqual(meta["native_context_positions"], 121)
                self.assertEqual(meta["augmented_context_positions"], 151)
                self.assertTrue(meta["completed_blend"])
                self.assertFalse(meta["measured_profile"])
                definition = load_policy(policy)[1]["policy"]
                expected = [(1-definition["weight"])*x + definition["weight"]*y for x,y in
                            zip(softmax(d, definition["direct_temperature"]), softmax(a, definition["augmented_temperature"]))]
                answer = output["answers"]["unusual/id"]
                actual = [1-answer["noul"], answer["noul"]] if kind == "noul" else list(answer["probabilities"].values())
                for x, y in zip(actual, expected):
                    self.assertAlmostEqual(x, y)

    def test_off_selective_always_modes_and_text_on_vision_backend(self):
        for mode in ("off", "selective", "on", "always"):
            for images in (False, True):
                request = image_body() if images else body()
                replies = [inspection(images=images), native(request, [0, 1000], POLICY)]
                if mode == "always":
                    replies += [complete_generation(), inspection(images=images), native(request, [1, 0], POLICY)]
                transport = FakeTransport(replies)
                output = DecisionPipeline(transport, reasoning=mode, policy_id=POLICY if mode != "off" else None,
                                          runtime_profile=profile()).decide(request)
                meta = output["winnow"]["adaptive"]
                self.assertEqual(meta["routed"], mode == "always")
                self.assertFalse(transport.replies)
                if mode == "off":
                    self.assertEqual(output["answers"]["unusual/id"]["probabilities"], {"z-first": 0.0, "a-second": 1.0})

    def test_invalid_or_unconfigured_images_fail_before_backend(self):
        bad = [None, "image", ["https://example.test/a.png"], ["file:///a.png"], ["data:image/png;base64,"],
               ["data:image/png;base64,YR=="], ["data:image/png;base64,AAAA!"], [17], IMAGES * 9]
        for images in bad:
            request = image_body(); request["winnow"]["images"] = images
            transport = FakeTransport([])
            with self.subTest(images=images), self.assertRaises(ValueError):
                DecisionPipeline(transport, reasoning="always", policy_id=POLICY, runtime_profile=profile()).decide(request)
            self.assertFalse(transport.calls)
        with self.assertRaisesRegex(ValueError, "explicit supported vision"):
            DecisionPipeline(FakeTransport([]), reasoning="always", policy_id=POLICY).decide(image_body())

    def test_profile_identity_and_direct_image_position_overflow_are_errors(self):
        for key in profile()["runtime"]:
            bad = inspection(); bad["runtime"][key] = "wrong"
            transport = FakeTransport([bad])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Backend"):
                DecisionPipeline(transport, reasoning="always", policy_id=POLICY, runtime_profile=profile()).decide(image_body())
            self.assertEqual(len(transport.calls), 1)
        for bad in (inspection(prefix=8192), inspection(images=False), dict(inspection(), suffix_tokens=[True]),
                    dict(inspection(), prefix_tokens=True)):
            with self.assertRaises(ValueError):
                DecisionPipeline(FakeTransport([bad]), reasoning="always", policy_id=POLICY, runtime_profile=profile()).decide(image_body())
        # Exactly context positions, including the native answer token, is accepted.
        request = image_body()
        output = DecisionPipeline(FakeTransport([inspection(prefix=8171), native(request, [0, 1000], POLICY)]),
                                  reasoning="selective", policy_id=POLICY, runtime_profile=profile()).decide(request)
        self.assertFalse(output["winnow"]["adaptive"]["routed"])

    def test_generation_and_augmented_limits_return_saved_calibrated_direct(self):
        request, policy = image_body(), load_policy(POLICY)[1]["policy"]
        variants = []
        for truncation in (True, None, "false"):
            result = complete_generation(); result["__verbose"]["truncated"] = truncation
            variants.append([result])
        result = complete_generation(); result["usage"]["prompt_tokens"] = 8192
        variants += [[result], [generation(finish="length", stop="limit")],
                     [complete_generation(), inspection(prefix=8192)],
                     [complete_generation(), dict(inspection(), image_tokens=0)],
                     [complete_generation(), inspection(), TimeoutError()], [TimeoutError()], [None],
                     [complete_generation(), {"runtime": None}]]
        for tail in variants:
            transport = FakeTransport([inspection(), native(request, [0, 1], POLICY), *tail])
            output = DecisionPipeline(transport, reasoning="always", policy_id=POLICY, runtime_profile=profile()).decide(request)
            meta = output["winnow"]["adaptive"]
            self.assertFalse(meta["completed_blend"])
            self.assertTrue(meta["fallback_reason"])
            self.assertEqual(list(output["answers"]["unusual/id"]["probabilities"].values()), softmax([0, 1], policy["direct_temperature"]))
            self.assertFalse(transport.replies)

    def test_cancelled_requests_do_not_fallback_or_start_another_phase(self):
        request = image_body()
        replies = [inspection(), native(request, [0, 0], POLICY), complete_generation(), inspection(), native(request, [1, 0], POLICY)]
        for phase in range(5):
            transport = FakeTransport(replies[:phase] + [RequestCancelled()])
            with self.assertRaises(RequestCancelled):
                DecisionPipeline(transport, reasoning="always", policy_id=POLICY, runtime_profile=profile()).decide(request)
            self.assertEqual(len(transport.calls), phase + 1)
        transport = FakeTransport([])
        with self.assertRaises(RequestCancelled):
            DecisionPipeline(transport, reasoning="always", policy_id=POLICY, runtime_profile=profile(), cancelled=lambda: True).decide(request)
        self.assertFalse(transport.calls)

    def test_request_lock_prevents_shared_backend_phase_interleaving(self):
        lock, calls = threading.Lock(), []
        class Transport:
            def post(self, path, request, seconds):
                calls.append((threading.current_thread().name, path))
                time.sleep(0.005)
                if path.endswith("inspect"): return inspection()
                if path.endswith("completions"): return complete_generation()
                return native(request, [0, 0], POLICY)
        outputs = []
        def run():
            outputs.append(DecisionPipeline(Transport(), reasoning="always", policy_id=POLICY,
                                            runtime_profile=profile(), request_lock=lock).decide(image_body()))
        threads = [threading.Thread(target=run, name=str(i)) for i in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(2)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(set(n for n, _ in calls[:5])), 1)
        self.assertEqual(len(set(n for n, _ in calls[5:])), 1)
        self.assertNotEqual(calls[0][0], calls[5][0])

    def test_runtime_contract_is_explicit_and_copied(self):
        p = profile()
        pipeline = DecisionPipeline(FakeTransport([]), reasoning="always", policy_id=POLICY, runtime_profile=p)
        p["runtime"]["context"] = 65536
        self.assertEqual(pipeline.profile["runtime"]["context"], 8192)
        for preset in ("12b-q8-vision8k-mtp", "12b-q8-text8k-mtp"):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_profile(preset)
        for key, value in (("context", True), ("resident_mtp", 1), ("chat_parallel", 4), ("cache_type", "q4_0")):
            p = profile(); p["runtime"][key] = value
            with self.assertRaises(ValueError): load_profile(p)
        p = profile(); p["launcher_requirements"]["no_context_shift"] = False
        with self.assertRaises(ValueError): load_profile(p)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            DecisionPipeline(FakeTransport([]), reasoning="always", policy_id=POLICY, runtime_profile=profile(), mtp="off")

    def test_vision_cli_uses_supported_presets_and_retains_model_specific_policy(self):
        for model in ("e4b", "nv4"):
            spec, kinds = assets.selection(model, "always", "on", "on")
            self.assertEqual(kinds, ["model", "projector", "assistant"])
            args = argparse.Namespace(model=model, reasoning="always", mtp="on", vision="on", context=8192,
                                      model_dir=Path("/owner/models"), server=None)
            with patch("winnow.platform.system", return_value="Linux"):
                command = winnow.serve_command(args)
                args.context = 65536
                with self.assertRaisesRegex(ValueError, "explicit operator"):
                    winnow.serve_command(args)
            self.assertIn("--mmproj", command)
            self.assertNotIn("--text-only", command)
            self.assertEqual(command[command.index("--preset")+1], spec["mtp_vision_preset"])
            with patch.object(sys, "argv", ["winnow", "decide", "--model", model, "--reasoning", "always", "--mtp", "on", "--vision", "on"]), patch.object(winnow.subprocess, "run") as run:
                winnow.main()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--runtime-profile")+1], spec["mtp_vision_preset"])
            self.assertEqual(command[command.index("--policy")+1], spec["policy"])
        with self.assertRaises(ValueError): assets.selection("q8", "always", "on", "on")
        with self.assertRaises(ValueError): assets.selection("e4b", "always", "off", "on")


class HTTPImageTransport(unittest.TestCase):
    def test_request_scoped_auth_is_copied_and_sent_without_shared_mutation(self):
        calls = []
        class Connection:
            sock = None
            def __init__(self, *args, **kwargs): pass
            def connect(self): pass
            def send(self, data): pass
            def request(self, method, path, payload, headers): calls.append((path, headers.copy(), json.loads(payload)))
            def getresponse(self): return self
            status = 200
            def read(self, maximum): return b'{"ok":true}'
            def close(self): pass
        with patch("decision_client.http.client.HTTPConnection", Connection):
            headers = {"Authorization": "Bearer first", "Content-Type": "application/json"}
            first = HTTPTransport("http://localhost:8080", headers)
            headers["Authorization"] = "Bearer changed"
            second = HTTPTransport("http://localhost:8080", {"Authorization": "Bearer second"})
            for path in ("/v1/winnow/inspect", "/v1/systemone", "/v1/chat/completions"):
                first.post(path, image_body(), 1)
            second.post("/v1/systemone", image_body(), 1)
        self.assertEqual([v[1]["Authorization"] for v in calls], ["Bearer first"]*3 + ["Bearer second"])
        self.assertEqual(calls[0][2]["winnow"]["images"], IMAGES)

    def test_cancellation_and_wall_timeout_close_inflight_connection_without_network(self):
        for cancel in (False, True):
            started, closed, cancelled = threading.Event(), threading.Event(), threading.Event()
            class Connection:
                def __init__(self, *args, **kwargs): self.sock = self
                def connect(self): pass
                def send(self, data): pass
                def request(self, *args): started.set()
                def getresponse(self):
                    closed.wait(2)
                    raise OSError("connection closed")
                def shutdown(self, how): closed.set()
                def close(self): closed.set()
            def trigger():
                started.wait(1)
                cancelled.set()
            with patch("decision_client.http.client.HTTPConnection", Connection):
                transport = HTTPTransport("http://localhost:8080", {}, cancelled=cancelled.is_set if cancel else None)
                thread = threading.Thread(target=trigger)
                if cancel: thread.start()
                before = time.monotonic()
                with self.assertRaises(RequestCancelled if cancel else TimeoutError):
                    transport.post("/v1/chat/completions", {}, 1 if cancel else 0.02)
                if cancel: thread.join(1)
            self.assertTrue(closed.is_set())
            self.assertLess(time.monotonic() - before, 1)


if __name__ == "__main__":
    unittest.main()
