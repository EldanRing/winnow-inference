"""Versioned experimental decisions; direct API requests remain the default."""

import copy
import hashlib
import http.client
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = (
    "Analyze this bounded decision briefly. Work through the rule and relevant facts, "
    "including any necessary intermediate steps. Use at most 100 words. Do not mention "
    "benchmark labels or evaluation. Your analysis will be supplied as context to a "
    "separate native decision scorer."
)


def load_policy(policy_id):
    manifest = json.loads((ROOT / "manifests/adaptive-v1.json").read_text())
    if manifest.get("schema_version") != 1 or policy_id not in manifest["policies"]:
        raise ValueError("Unknown adaptive policy/version")
    return manifest, manifest["policies"][policy_id]


def keys_for(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
        ):
            raise ValueError("Invalid noul criteria")
        keys = ["false", "true"]
    elif kind == "choice":
        if not isinstance(criteria, dict) or any(not isinstance(k, str) or not k for k in criteria):
            raise ValueError("Invalid choice criteria")
        keys = list(criteria)
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError("Invalid score criteria")
        keys = [str(i) for i in range(len(criteria))]
    else:
        raise ValueError("Unsupported question type")
    if not 2 <= len(keys) <= 64:
        raise ValueError("Expected 2–64 candidates")
    return keys


def softmax(logits, temperature):
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid probability inputs")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in logits):
        raise ValueError("Invalid native logits")
    top = max(logits)
    values = [math.exp((v - top) / temperature) for v in logits]
    total = sum(values)
    return [v / total for v in values]


def native_probabilities(response, qid, kind, keys, alias, mtp=True):
    if not isinstance(response, dict) or response.get("model") != alias:
        raise ValueError("Native model identity changed")
    answers = response.get("answers")
    if not isinstance(answers, dict) or list(answers) != [qid]:
        raise ValueError("Native question mapping changed")
    answer = answers[qid]
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise ValueError("Native question type changed")
    diagnostic = answer.get("winnow", {})
    if not isinstance(diagnostic, dict):
        raise ValueError("Malformed native diagnostics")
    logits = diagnostic.get("logits")
    if diagnostic.get("temperature") != 1.0 or not isinstance(logits, list) or len(logits) != len(keys):
        raise ValueError("Expected raw native T=1 candidate logits")
    p = softmax(logits, 1.0)
    if kind == "noul":
        value = answer.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Malformed noul probability")
        actual = [1 - value, value]
    else:
        values = answer.get("probabilities")
        if not isinstance(values, dict) or list(values) != keys:
            raise ValueError("Native candidate order changed")
        actual = list(values.values())
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1 for v in actual):
        raise ValueError("Malformed native probabilities")
    if any(abs(a - b) > 1e-10 for a, b in zip(actual, p)):
        raise ValueError("Native probability/logit mismatch")
    usage = response.get("usage", {})
    if (not isinstance(usage, dict) or isinstance(usage.get("input_tokens"), bool)
            or not isinstance(usage.get("input_tokens"), int) or usage["input_tokens"] < 0
            or usage.get("output_tokens") != 0):
        raise ValueError("Malformed native usage")
    runtime = response.get("winnow", {})
    if (not isinstance(runtime, dict)
            or runtime.get("resident_speculative_chat") is not mtp
            or (mtp and (runtime.get("context_evictions") != 0 or runtime.get("memory_policy") != "mixed"))
            or (not mtp and runtime.get("memory_policy") not in {"mixed", "exclusive"})):
        raise ValueError("Resident native context/profile changed")
    return list(logits), p


def render(question, keys, probabilities):
    kind = question["type"]
    answer = {"type": kind}
    if kind == "noul":
        answer["noul"] = probabilities[1]
        return answer
    answer["probabilities"] = dict(zip(keys, probabilities))
    entropy = -sum(p * math.log(p) for p in probabilities if p)
    answer["confidence"] = max(0.0, min(1.0, 1 - entropy / math.log(len(keys))))
    if kind == "choice":
        answer["choice"] = keys[max(range(len(keys)), key=lambda i: probabilities[i])]
    else:
        answer["score"] = sum(i * p for i, p in enumerate(probabilities))
        answer["legend"] = dict(zip(keys, question["criteria"]))
    return answer


class DecisionPipeline:
    def __init__(self, transport, mode="direct", policy_id=None, mtp="on", target=None):
        if mode not in {"direct", "experimental-adaptive"}:
            raise ValueError("Unknown decision mode")
        if (mode == "experimental-adaptive") != (policy_id is not None):
            raise ValueError("A policy is required only for experimental-adaptive mode")
        self.transport, self.mode, self.target = transport, mode, target
        if mtp not in {"on", "off"}:
            raise ValueError("MTP must be on or off")
        self.mtp = mtp == "on"
        self.manifest, self.definition = load_policy(policy_id) if policy_id else (None, None)

    def decide(self, body):
        if self.mode == "direct":
            # The unified CLI binds a verified artifact; the low-level API retains legacy aliases.
            if self.target:
                identity = self.transport.post("/v1/winnow/inspect", body, 30).get("runtime", {})
                if identity.get("target_sha256") != self.target["model"]["sha256"]:
                    raise ValueError("Selected model/quantization does not match the server; restart with the matching winnow serve --model")
            response = self.transport.post("/v1/systemone", body, 30)
            if self.target and response.get("model") != self.target["alias"]:
                raise ValueError("Selected model alias does not match the server response")
            return response
        if not isinstance(body, dict) or not isinstance(body.get("state"), (str, dict, list)):
            raise ValueError("Experimental adaptive mode requires a text, object, or array state")
        questions = body.get("questions")
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError("Experimental adaptive mode requires exactly one question")
        qid, question = next(iter(questions.items()))
        if not isinstance(qid, str) or not qid or not isinstance(question, dict):
            raise ValueError("Invalid question")
        extension = body.get("winnow", {})
        if not isinstance(extension, dict) or extension.get("images") or extension.get("temperature", 1) != 1:
            raise ValueError("Experimental adaptive mode is text-only and requires raw T=1")
        keys, kind = keys_for(question), question["type"]
        definition, policy = self.definition, self.definition["policy"]
        if body.get("model", definition["alias"]) != definition["alias"]:
            raise ValueError("Request model must match the selected policy")
        request = copy.deepcopy(body)
        request["model"] = definition["alias"]
        request["winnow"] = dict(extension, diagnostics=True, temperature=1.0)
        identity = self.transport.post("/v1/winnow/inspect", request, 30).get("runtime", {})
        expected_runtime = hashlib.sha256((ROOT / "runtime.lock.json").read_bytes()).hexdigest()
        if (
            not isinstance(identity, dict) or identity.get("runtime_sha256") != expected_runtime
            or identity.get("target_sha256") != definition["target"]["sha256"]
            or (self.mtp and identity.get("assistant_sha256") != definition["assistant"]["sha256"])
            or (not self.mtp and identity.get("assistant_sha256") is not None)
            or identity.get("resident_mtp") is not self.mtp
            or identity.get("vision") is not False
            or not isinstance(identity.get("context"), int) or identity["context"] < 512
        ):
            raise ValueError("Backend does not match the pinned experimental policy/profile")
        measured_profile = all(identity.get(k) == v for k, v in dict(
            context=8192, chat_context=8192, chat_parallel=1, memory_setting="auto",
            parallel=4, cache_type="q8_0", head="selected", pipeline="optimized",
            batch=definition["profile"]["batch"], ubatch=definition["profile"]["ubatch"]).items())
        # A failed/malformed direct decision remains an error; it is never fabricated.
        direct = self.transport.post("/v1/systemone", request, 30)
        logits, raw = native_probabilities(direct, qid, kind, keys, definition["alias"], self.mtp)
        final = softmax(logits, policy["direct_temperature"])
        if policy["gate"] == "normalized_entropy":
            gate_value = -sum(p * math.log(p) for p in raw if p) / math.log(len(raw))
            routed = gate_value > policy["threshold"]
        else:
            gate_value = max(raw)
            routed = gate_value < policy["threshold"]
        metadata = {
            "mode": self.mode, "policy_set_version": self.manifest["policy_set_version"],
            "policy_id": definition["id"], "experimental": True,
            "gate": policy["gate"], "gate_value_raw_T1": gate_value,
            "threshold": policy["threshold"], "routed": routed,
            "direct_temperature": policy["direct_temperature"],
            "augmented_temperature": policy["augmented_temperature"],
            "blend_weight": policy["weight"], "fallback_reason": None,
            "generation_prompt_tokens": None if routed else 0,
            "generation_output_tokens": None if routed else 0,
            "generation_usage_available": not routed, "native_input_tokens": direct["usage"]["input_tokens"],
            "native_output_tokens": 0, "completed_blend": False,
            "mtp": "on" if self.mtp else "off",
            "measured_profile": measured_profile,
            "configuration_note": None if measured_profile else "Custom configuration: frozen policy retained; published calibration/quality/latency results do not validate these settings.",
        }
        if routed:
            generation = dict(self.manifest["generation"])
            generation.pop("context")
            seconds = generation.pop("request_seconds")
            prompt = (
                INSTRUCTION + "\nTask state:\n" + json.dumps(body["state"], separators=(",", ":"))
                + "\nDecision question:\n" + json.dumps(question, separators=(",", ":"))
            )
            generation.update(model=definition["alias"], messages=[{"role": "user", "content": prompt}],
                              cache_prompt=False, verbose=True, return_tokens=True, stream=False)
            try:
                result = self.transport.post("/v1/chat/completions", generation, seconds)
                usage = result.get("usage", {})
                if not isinstance(usage, dict):
                    raise ValueError("Malformed generation usage")
                for key in ["prompt_tokens", "completion_tokens"]:
                    if isinstance(usage.get(key), bool) or not isinstance(usage.get(key), int) or usage[key] < 0:
                        raise ValueError("Malformed generation usage")
                metadata.update(generation_prompt_tokens=usage["prompt_tokens"],
                                generation_output_tokens=usage["completion_tokens"],
                                generation_usage_available=True)
                choices = result.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("Malformed generation choices")
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                    raise ValueError("Malformed generation message")
                content = choice["message"].get("content")
                verbose = result.get("__verbose", {})
                if not isinstance(verbose, dict):
                    raise ValueError("Malformed generation stop metadata")
                # Native partial text is never scored; only supported chat's actual EOS is eligible.
                if choice.get("finish_reason") != "stop" or verbose.get("stop_type") != "eos":
                    metadata["fallback_reason"] = "generation_not_natural_EOS"
                elif not isinstance(content, str) or not content.strip():
                    metadata["fallback_reason"] = "generation_empty_or_malformed"
                else:
                    augmented_request = copy.deepcopy(request)
                    if isinstance(body["state"], str):
                        augmented_request["state"] = body["state"] + "\n\nModel reasoning:\n" + content
                    else:
                        augmented_request["state"] = {"original_state": body["state"], "model_reasoning": content}
                    try:
                        augmented = self.transport.post("/v1/systemone", augmented_request, 30)
                        alogits, _ = native_probabilities(augmented, qid, kind, keys, definition["alias"], self.mtp)
                        calibrated = softmax(alogits, policy["augmented_temperature"])
                        metadata["native_input_tokens"] += augmented["usage"]["input_tokens"]
                        final = [(1 - policy["weight"]) * a + policy["weight"] * b
                                 for a, b in zip(final, calibrated)]
                        metadata["completed_blend"] = True
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                            http.client.HTTPException) as error:
                        metadata["fallback_reason"] = "augmented_failure:" + type(error).__name__
            except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                    http.client.HTTPException) as error:
                metadata["fallback_reason"] = "generation_failure:" + type(error).__name__
        response = {"model": definition["alias"], "answers": {qid: render(question, keys, final)},
                    "usage": {"input_tokens": metadata["native_input_tokens"], "output_tokens": 0},
                    "winnow": {"adaptive": metadata}}
        if extension.get("diagnostics"):
            metadata["raw_direct_logits"] = logits
        return response
