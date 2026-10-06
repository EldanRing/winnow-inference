"""Contract, identity, raw gate, calibrated fallback and timeout regressions."""

import copy
import hashlib
import http.client
import io
import json
import math
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from adaptive_policy import DecisionPipeline, keys_for, load_policy, softmax  # noqa: E402
from decision_client import HTTPTransport  # noqa: E402
import serve  # noqa: E402


def body(kind="choice", state="Frozen text input."):
    question = {"type": kind, "instructions": "Select the option matching the record."}
    if kind == "choice":
        question["criteria"] = {"z-first": "first", "a-second": "second"}
    if kind == "score":
        question["criteria"] = ["low", "medium", "high"]
    return {"state": state, "questions": {"unusual/id": question}}


def identity(policy_id):
    _, p = load_policy(policy_id)
    return {"runtime": {"runtime_sha256": hashlib.sha256((ROOT / "runtime.lock.json").read_bytes()).hexdigest(),
                        "target_sha256": p["target"]["sha256"],
                        "assistant_sha256": p["assistant"]["sha256"], "resident_mtp": True,
                        "context": 8192, "vision": False, "parallel": 4,
                        "cache_type": "q8_0", "head": "selected", "pipeline": "optimized",
                        "batch": p["profile"]["batch"], "ubatch": p["profile"]["ubatch"]}}


def native(request, logits, policy_id):
    _, definition = load_policy(policy_id)
    qid, question = next(iter(request["questions"].items()))
    keys, kind = keys_for(question), question["type"]
    p = softmax(logits, 1)
    answer = {"type": kind, "winnow": {"temperature": 1.0, "logits": logits}}
    if kind == "noul":
        answer["noul"] = p[1]
    else:
        answer["probabilities"] = dict(zip(keys, p))
    return {"model": definition["alias"], "answers": {qid: answer},
            "usage": {"input_tokens": 37, "output_tokens": 0},
            "winnow": {"context_evictions": 0, "memory_policy": "mixed", "resident_speculative_chat": True}}


def generation(content="A complete input-only analysis.", finish="stop", stop="eos"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "__verbose": {"stop_type": stop}, "usage": {"prompt_tokens": 23, "completion_tokens": 7}}


class FakeTransport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def post(self, endpoint, request, seconds):
        self.calls.append((endpoint, copy.deepcopy(request), seconds))
        if not self.replies:
            raise AssertionError("Unexpected backend call")
        result = self.replies.pop(0)
        if isinstance(result, BaseException):
            raise result
        return copy.deepcopy(result)


class AdaptivePipeline(unittest.TestCase):
    def test_direct_default_is_exact_passthrough(self):
        request = {"state": {"items": [1, 2]}, "questions": {"a": {}, "b": {}},
                   "winnow": {"temperature": 2, "images": ["fixture"]}}
        response = {"original": "response", "answers": {}}
        transport = FakeTransport([response])
        self.assertEqual(DecisionPipeline(transport).decide(request), response)
        self.assertEqual(transport.calls, [("/v1/systemone", request, 30)])
        for mode, policy in [("direct", "nvfp4-entropy-v1"), ("experimental-adaptive", None)]:
            with self.assertRaises(ValueError):
                DecisionPipeline(transport, mode, policy)

    def test_exact_frozen_config_and_artifact_bindings(self):
        _, nv = load_policy("nvfp4-entropy-v1")
        self.assertEqual(nv["policy"], {"direct_temperature": 1.0, "augmented_temperature": 1.0,
                                     "weight": 0.5, "gate": "normalized_entropy",
                                     "threshold": 0.48619198949270803})
        _, e4 = load_policy("e4b-calibrated50-v1")
        self.assertEqual(e4["policy"]["direct_temperature"], 1.2041180007310734)
        self.assertEqual(e4["policy"]["augmented_temperature"], 3.4209273427377678)
        self.assertNotEqual(nv["assistant"]["sha256"], e4["assistant"]["sha256"])

    def test_e4b_release_policy_routes_and_blends_as_shipped(self):
        manifest, current = load_policy("e4b-calibrated75-g95-v1")
        _, legacy = load_policy("e4b-calibrated50-v1")
        self.assertEqual(manifest["policy_set_version"], "adaptive-20261006-e2b-v3")
        self.assertEqual(current["target"], legacy["target"])
        self.assertEqual(current["source_selection_sha256"],
                         "41fedd8f22fb23e1d2046c22a8b460a86af8ad2c7032ec76ff1bb045d690356d")
        self.assertEqual(current["inherited_calibration_provenance"], {
            "policy_id": "e4b-calibrated50-v1",
            "source_selection_sha256": legacy["source_selection_sha256"]})
        aggregate = json.loads((ROOT / "docs/reasoning-results.json").read_text())
        released = aggregate["released_policy_contract"]
        self.assertEqual(released["policy_set_version"], "adaptive-20261005-e4b-v2")
        self.assertEqual(released["generation"], manifest["generation"])
        for identifier, definition in released["policies"].items():
            self.assertEqual(definition, manifest["policies"][identifier])
        study = aggregate["e4b_staged_288_holdout"]
        self.assertEqual(study["current_policy_contract"]["source_selection_sha256"],
                         current["source_selection_sha256"])
        self.assertEqual(study["current_policy_contract"]["weight"], 0.75)
        self.assertEqual(study["current_policy_contract"]["threshold"], 0.95)
        self.assertEqual(current["assistant"], legacy["assistant"])
        self.assertEqual(current["policy"], {
            "direct_temperature": 1.2041180007310734,
            "augmented_temperature": 3.4209273427377678,
            "weight": 0.75, "gate": "raw_maxP", "threshold": 0.95})
        assets = json.loads((ROOT / "manifests/release-assets-v1.json").read_text())
        self.assertEqual(assets["models"]["e4b-q8"]["policy"], current["id"])
        self.assertEqual(assets["models"]["12b-q8"]["policy"], "q8-fixed50-v1")
        request = body()
        logits = [0.0, 0.4]
        augmented = [0.0, 2.0]
        t = FakeTransport([identity(current["id"]), native(request, logits, current["id"]),
                           generation(), native(request, augmented, current["id"])])
        out = DecisionPipeline(t, "experimental-adaptive", current["id"]).decide(request)
        meta = out["winnow"]["adaptive"]
        self.assertTrue(meta["routed"])
        self.assertTrue(meta["completed_blend"])
        direct = softmax(logits, 1.2041180007310734)
        reasoned = softmax(augmented, 3.4209273427377678)
        values = list(out["answers"]["unusual/id"]["probabilities"].values())
        for got, a, b in zip(values, direct, reasoned):
            self.assertAlmostEqual(got, 0.25*a + 0.75*b)

    def test_unsupported_adaptive_contract_rejected_before_backend(self):
        for request in [body(state=None), body(state=17), body(state=True),
                        {"state": "text", "questions": {"a": {}, "b": {}}},
                        dict(body(), winnow={"images": ["fixture"]}),
                        dict(body(), winnow={"temperature": 2}), dict(body(), model="Wrong-model")]:
            t = FakeTransport([])
            with self.subTest(request=request), self.assertRaises(ValueError):
                DecisionPipeline(t, "experimental-adaptive", "nvfp4-entropy-v1").decide(request)
            self.assertEqual(t.calls, [])

    def test_structured_states_preserve_original_native_shape_and_prompt(self):
        policy = "q8-fixed50-v1"
        for state in ({"events": ["received", "approved"], "active": True}, ["first", {"second": 2}]):
            for kind in ("noul", "choice", "score"):
                request = body(kind, state=state)
                original = copy.deepcopy(request)
                keys = keys_for(request["questions"]["unusual/id"])
                logits = [0.0] * len(keys)
                t = FakeTransport([identity(policy), native(request, logits, policy),
                                   generation(), native(request, logits, policy)])
                out = DecisionPipeline(t, "experimental-adaptive", policy).decide(request)
                with self.subTest(state=state, kind=kind):
                    self.assertEqual(request, original)
                    self.assertEqual(t.calls[1][1]["state"], state)
                    expected_prompt = ("Analyze this bounded decision briefly. Work through the rule and relevant facts, "
                                       "including any necessary intermediate steps. Use at most 100 words. Do not mention "
                                       "benchmark labels or evaluation. Your analysis will be supplied as context to a "
                                       "separate native decision scorer.\nTask state:\n"
                                       + json.dumps(state, separators=(",", ":")) + "\nDecision question:\n"
                                       + json.dumps(request["questions"]["unusual/id"], separators=(",", ":")))
                    self.assertEqual(t.calls[2][1]["messages"][0]["content"], expected_prompt)
                    self.assertEqual(t.calls[3][1]["state"],
                                     {"original_state": state, "model_reasoning": "A complete input-only analysis."})
                    self.assertTrue(out["winnow"]["adaptive"]["completed_blend"])

    def test_identity_context_precision_and_residency_are_required(self):
        for key, value in [("target_sha256", "wrong"), ("assistant_sha256", "wrong"),
                           ("runtime_sha256", "wrong"), ("resident_mtp", False),
                           ("context", 511), ("vision", True)]:
            i = identity("nvfp4-entropy-v1")
            i["runtime"][key] = value
            t = FakeTransport([i])
            with self.subTest(key=key), self.assertRaises(ValueError):
                DecisionPipeline(t, "experimental-adaptive", "nvfp4-entropy-v1").decide(body())
            self.assertEqual(len(t.calls), 1)

    def test_custom_context_keeps_policy_and_marks_unmeasured_configuration(self):
        policy = "nvfp4-entropy-v1"; request = body()
        ident = identity(policy); ident["runtime"]["context"] = 4096
        response = native(request, [10.0, 0.0], policy)
        out = DecisionPipeline(FakeTransport([ident, response]), "experimental-adaptive", policy).decide(request)
        self.assertFalse(out["winnow"]["adaptive"]["measured_profile"])
        self.assertIn("Custom configuration", out["winnow"]["adaptive"]["configuration_note"])
        self.assertEqual(out["winnow"]["adaptive"]["direct_temperature"], 1)

    def test_raw_gate_precedes_calibration_and_boundary_bypasses(self):
        request = body()
        policy = "e4b-calibrated50-v1"
        logits = [0, math.log(4)]
        direct = native(request, logits, policy)
        t = FakeTransport([identity(policy), direct])
        out = DecisionPipeline(t, "experimental-adaptive", policy).decide(request)
        m = out["winnow"]["adaptive"]
        self.assertFalse(m["routed"])
        self.assertEqual(m["gate_value_raw_T1"], 0.8)
        self.assertLess(max(out["answers"]["unusual/id"]["probabilities"].values()), 0.8)
        self.assertEqual(len(t.calls), 2)

    def test_entropy_boundary_is_strict(self):
        request, policy = body(), "nvfp4-entropy-v1"
        logits = [0, 0]
        t = FakeTransport([identity(policy), native(request, logits, policy)])
        pipeline = DecisionPipeline(t, "experimental-adaptive", policy)
        pipeline.definition["policy"]["threshold"] = 1.0  # Boundary fixture only; shipped config untouched.
        self.assertFalse(pipeline.decide(request)["winnow"]["adaptive"]["routed"])

    def test_complete_blend_preserves_all_kinds_keys_ties_score_legend_and_usage(self):
        for policy in ["nvfp4-entropy-v1", "e4b-calibrated50-v1"]:
            for kind in ["noul", "choice", "score"]:
                request = body(kind)
                keys = keys_for(request["questions"]["unusual/id"])
                direct_logits = [0] * len(keys)
                augmented_logits = [0, 2] if len(keys) == 2 else [0, 1, 2]
                t = FakeTransport([identity(policy), native(request, direct_logits, policy),
                                   generation(), native(request, augmented_logits, policy)])
                out = DecisionPipeline(t, "experimental-adaptive", policy).decide(request)
                _, definition = load_policy(policy)
                p = softmax(augmented_logits, definition["policy"]["augmented_temperature"])
                expected = [0.5 / len(keys) + 0.5 * v for v in p]
                answer = out["answers"]["unusual/id"]
                with self.subTest(policy=policy, kind=kind):
                    if kind == "noul":
                        self.assertEqual(answer["noul"], expected[1])
                    else:
                        self.assertEqual(list(answer["probabilities"]), keys)
                        self.assertEqual(list(answer["probabilities"].values()), expected)
                    if kind == "score":
                        self.assertEqual(answer["score"], sum(i * p for i, p in enumerate(expected)))
                        self.assertEqual(answer["legend"], dict(zip(keys, ["low", "medium", "high"])))
                    self.assertTrue(out["winnow"]["adaptive"]["completed_blend"])
                    self.assertEqual(out["usage"], {"input_tokens": 74, "output_tokens": 0})
                    self.assertEqual(out["winnow"]["adaptive"]["generation_output_tokens"], 7)
                    self.assertEqual(out["winnow"]["adaptive"]["generation_prompt_tokens"], 23)
                    self.assertEqual(t.calls[2][1]["max_tokens"], -1)
                    self.assertEqual(t.calls[2][1]["seed"], 314159)
                    self.assertEqual(t.calls[2][1]["reasoning_effort"], "none")
                    self.assertEqual(t.calls[3][1]["state"], request["state"] + "\n\nModel reasoning:\nA complete input-only analysis.")
        t = FakeTransport([identity("nvfp4-entropy-v1"), native(body(), [0, 0], "nvfp4-entropy-v1"),
                           generation(), native(body(), [0, 0], "nvfp4-entropy-v1")])
        self.assertEqual(DecisionPipeline(t, "experimental-adaptive", "nvfp4-entropy-v1")
                         .decide(body())["answers"]["unusual/id"]["choice"], "z-first")

    def test_every_generation_failure_preserves_calibrated_direct_without_partial_scoring(self):
        malformed = generation()
        malformed["choices"] = [None]
        malformed_usage = generation()
        malformed_usage["usage"]["completion_tokens"] = True
        failures = [TimeoutError("deadline"), OSError("network"), RuntimeError("Backend HTTP500"),
                    http.client.BadStatusLine("bad"), generation("partial", "length", "limit"),
                    generation("partial", "stop", "limit"), generation("partial", "stop", None),
                    generation(""), generation(17), malformed, malformed_usage]
        for bad in failures:
            request, policy, logits = body(), "e4b-calibrated50-v1", [0.0, 0.2]
            t = FakeTransport([identity(policy), native(request, logits, policy), bad])
            out = DecisionPipeline(t, "experimental-adaptive", policy).decide(request)
            expected = softmax(logits, load_policy(policy)[1]["policy"]["direct_temperature"])
            with self.subTest(bad=bad):
                self.assertEqual(list(out["answers"]["unusual/id"]["probabilities"].values()), expected)
                self.assertIsNotNone(out["winnow"]["adaptive"]["fallback_reason"])
                self.assertEqual(len(t.calls), 3)
                self.assertFalse(out["winnow"]["adaptive"]["completed_blend"])
                if isinstance(bad, BaseException):
                    self.assertFalse(out["winnow"]["adaptive"]["generation_usage_available"])
                    self.assertIsNone(out["winnow"]["adaptive"]["generation_output_tokens"])

    def test_augmented_failure_does_not_change_saved_policy_direct(self):
        request, policy = body(), "e4b-calibrated50-v1"
        malformed = native(request, [0, 3], policy)
        malformed["usage"]["input_tokens"] = None
        wrong_order = native(request, [0, 3], policy)
        wrong_order["answers"]["unusual/id"]["probabilities"] = dict(reversed(list(
            wrong_order["answers"]["unusual/id"]["probabilities"].items())))
        for error in [TimeoutError(), RuntimeError(), malformed, wrong_order]:
            t = FakeTransport([identity(policy), native(request, [0, 0.2], policy), generation(), error])
            out = DecisionPipeline(t, "experimental-adaptive", policy).decide(request)
            self.assertEqual(list(out["answers"]["unusual/id"]["probabilities"].values()),
                             softmax([0, 0.2], load_policy(policy)[1]["policy"]["direct_temperature"]))
            self.assertTrue(out["winnow"]["adaptive"]["fallback_reason"].startswith("augmented_failure:"))
            self.assertEqual(out["usage"]["input_tokens"], 37)

    def test_direct_failure_and_bad_native_mapping_remain_errors(self):
        request, policy = body(), "nvfp4-entropy-v1"
        for bad in [OSError(), native(request, [0, 0], policy)]:
            if isinstance(bad, dict):
                bad["answers"]["unusual/id"]["winnow"]["logits"] = [float("nan"), 0]
            t = FakeTransport([identity(policy), bad])
            with self.assertRaises((OSError, ValueError)):
                DecisionPipeline(t, "experimental-adaptive", policy).decide(request)
            self.assertEqual(len(t.calls), 2)

    def test_normal_launcher_ignores_stale_experimental_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ["model", "server"]:
                (directory / name).touch()
            with patch.dict("os.environ", {"WINNOW_RESIDENT_MTP": "1", "WINNOW_RESIDENT_NGRAM": "1",
                                           "WINNOW_NATIVE_CHECKPOINT": "1", "WINNOW_TARGET_SHA256": "stale"}):
                out = json.loads(subprocess.check_output([
                    sys.executable, str(ROOT / "scripts/serve.py"), "--model", str(directory / "model"),
                    "--server", str(directory / "server"), "--dry-run"], text=True))
            self.assertEqual(out["environment"]["WINNOW_RESIDENT_MTP"], "0")
            self.assertNotIn("WINNOW_NATIVE_CHECKPOINT", out["environment"])
            self.assertNotIn("WINNOW_TARGET_SHA256", out["environment"])
            self.assertEqual(out["command"][out["command"].index("--spec-type") + 1], "none")
            self.assertIn("65536", out["command"])

    def test_transport_wall_timeout_disconnects_without_process_signals(self):
        ready, received, disconnected = threading.Event(), threading.Event(), threading.Event()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def backend():
            ready.set()
            connection, _ = listener.accept()
            connection.settimeout(2)
            try:
                data = b""
                while b"\r\n\r\n" not in data:
                    data += connection.recv(4096)
                received.set()
                while connection.recv(4096):
                    pass
                disconnected.set()
            finally:
                connection.close()
                listener.close()

        worker = threading.Thread(target=backend, daemon=True)
        worker.start()
        ready.wait(1)
        start = time.monotonic()
        with self.assertRaises((TimeoutError, OSError)):
            HTTPTransport("http://127.0.0.1:" + str(port), {}).post("/v1/systemone", {}, 0.15)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertTrue(received.wait(1))
        self.assertTrue(disconnected.wait(1))
        worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_opt_in_profile_resolves_exact_safety_settings_and_verifies_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ["model", "assistant", "server"]:
                (directory / name).touch()
            base = ["serve.py", "--model", str(directory / "model"),
                    "--assistant", str(directory / "assistant"), "--server", str(directory / "server"),
                    "--experimental-adaptive", "e4b-calibrated50-v1", "--text-only", "--dry-run"]
            out = io.StringIO()
            with patch.object(sys, "argv", base), patch.object(sys, "stdout", out), \
                    patch("serve.platform.system", return_value="Linux"), patch("serve.verify") as verify:
                serve.main()
            self.assertEqual(verify.call_count, 2)
            result = json.loads(out.getvalue())
            self.assertEqual(result["environment"]["WINNOW_RESIDENT_MTP"], "1")
            self.assertEqual(result["environment"]["WINNOW_CONTEXT"], "8192")
            self.assertEqual(result["environment"]["WINNOW_MEMORY"], "auto")
            self.assertEqual(result["environment"]["WINNOW_CACHE"], "q8_0")
            command = result["command"]
            self.assertEqual(command[command.index("--alias") + 1], "Winnow-E4B")
            self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "4")
            for extra in [["--chat-parallel", "2"], ["--memory", "exclusive"],
                          ["--spec-type", "draft-dflash"],
                          ["--cache-type-k", "f16"], ["--ctx-size", "65536"], ["--lora", "fixture"]]:
                with self.subTest(extra=extra), patch.object(sys, "argv", base + extra), \
                        patch.object(sys, "stderr", io.StringIO()), \
                        patch("serve.platform.system", return_value="Linux"), patch("serve.verify") as verify:
                    with self.assertRaises(SystemExit):
                        serve.main()
                    verify.assert_not_called()
            with patch.object(sys, "argv", base), patch.object(sys, "stderr", io.StringIO()), \
                    patch("serve.platform.system", return_value="Linux"):
                with self.assertRaises(SystemExit):
                    serve.main()  # Real identity check rejects tiny incompatible artifact before launch.


if __name__ == "__main__":
    unittest.main()
