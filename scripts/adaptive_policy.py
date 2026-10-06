"""Versioned experimental decisions; direct API requests remain the default."""

import copy
import hashlib
import http.client
import json
import math
import threading
from pathlib import Path
from reasoning_contract import (ContextLimitError, RequestCancelled, check_inspection,
                                load_profile, read_object, validate_images)

ROOT = Path(__file__).resolve().parents[1]
REASONING_CHOICES = ("off", "selective", "always", "on")


def normalize_reasoning(value):
    """Keep the original on/off interface while naming selective routing explicitly."""
    if value not in REASONING_CHOICES:
        raise ValueError("Reasoning must be off, selective, or always (on aliases selective)")
    return "selective" if value == "on" else value


INSTRUCTION = (
    "Analyze this bounded decision briefly. Work through the rule and relevant facts, "
    "including any necessary intermediate steps. Use at most 100 words. Do not mention "
    "benchmark labels or evaluation. Your analysis will be supplied as context to a "
    "separate native decision scorer."
)


def load_policy(policy_id, policy_manifest=None):
    manifest = read_object(policy_manifest or ROOT / "manifests/adaptive-v1.json")
    if manifest.get("schema_version") != 1 or policy_id not in manifest["policies"]:
        raise ValueError("Unknown adaptive policy/version")
    definition = manifest["policies"][policy_id]
    policy = definition["policy"]
    if definition["id"] != policy_id or policy["gate"] not in {"raw_maxP", "normalized_entropy"}:
        raise ValueError("Invalid policy identity/gate")
    for key in ("direct_temperature", "augmented_temperature", "weight", "threshold"):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if "temperature" in key else not 0 <= value <= 1):
            raise ValueError("Invalid policy parameter: " + key)
    generation = manifest["generation"]
    if (generation.get("max_tokens") != -1 or generation.get("ignore_eos") is not False
            or generation.get("temperature") != 0 or generation.get("seed") != 314159
            or generation.get("reasoning_effort") != "none"
            or generation.get("prompt_format", "bounded-v1") not in {"bounded-v1", "e2b-canonical-v2"}):
        raise ValueError("Unsupported reasoning generation contract")
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


def rendered_options(question, keys):
    """Match native/protocol.h: keys name choice/noul options, values describe them."""
    criteria = question.get("criteria")
    values = ([criteria.get(key) for key in keys] if isinstance(criteria, dict)
              else criteria if isinstance(criteria, list) else [None] * len(keys))
    options = []
    for key, value in zip(keys, values):
        if value is None:
            options.append(key)
            continue
        if not isinstance(value, (str, dict, list)):
            raise ValueError("Invalid criterion description")
        description = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
        options.append(description if question["type"] == "score" else key + ": " + description)
    return options


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
    def __init__(self, transport, mode=None, policy_id=None, mtp="on", target=None, *, reasoning=None,
                 policy_manifest=None, runtime_profile=None, cancelled=None, request_lock=None):
        if mode not in {None, "direct", "experimental-adaptive"}:
            raise ValueError("Unknown decision mode")
        if reasoning is None:
            reasoning = "selective" if mode == "experimental-adaptive" else "off"
        self.reasoning = normalize_reasoning(reasoning)
        resolved_mode = "direct" if self.reasoning == "off" else "experimental-adaptive"
        if mode is not None and mode != resolved_mode:
            raise ValueError("Conflicting mode and reasoning options")
        mode = resolved_mode
        if (mode == "experimental-adaptive") != (policy_id is not None):
            raise ValueError("A policy is required only for experimental-adaptive mode")
        self.transport, self.mode, self.target = transport, mode, target
        if mtp not in {"on", "off"}:
            raise ValueError("MTP must be on or off")
        self.mtp = mtp == "on"
        self.manifest, self.definition = load_policy(policy_id, policy_manifest) if policy_id else (None, None)
        if policy_manifest is not None and not policy_id:
            raise ValueError("A policy manifest requires a policy id")
        self.profile = load_profile(runtime_profile) if runtime_profile is not None else None
        if policy_manifest is not None and self.profile is None:
            raise ValueError("A local policy requires an explicit runtime profile")
        if self.profile:
            expected = self.profile["runtime"]
            if expected["resident_mtp"] is not self.mtp:
                raise ValueError("MTP setting conflicts with runtime profile")
            if self.definition and (self.profile["alias"] != self.definition["alias"]
                    or expected["target_sha256"] != self.definition["target"]["sha256"]
                    or (self.mtp and expected["assistant_sha256"] != self.definition["assistant"]["sha256"])):
                raise ValueError("Runtime profile conflicts with selected policy artifacts")
            if target and (self.profile["alias"] != target["alias"] or expected["target_sha256"] != target["model"]["sha256"]):
                raise ValueError("Runtime profile conflicts with selected target")
        self.cancelled = cancelled
        # A service sharing one backend should pass its common lock to every request-scoped pipeline.
        self.request_lock = request_lock if request_lock is not None else threading.Lock()

    def _check_cancelled(self):
        if self.cancelled and self.cancelled():
            raise RequestCancelled("Decision request cancelled")

    def _post(self, endpoint, body, seconds):
        self._check_cancelled()
        result = self.transport.post(endpoint, body, seconds)
        self._check_cancelled()
        return result

    def decide(self, body):
        self._check_cancelled()
        while not self.request_lock.acquire(timeout=0.1):
            self._check_cancelled()
        try:
            return self._decide(copy.deepcopy(body))
        finally:
            self.request_lock.release()

    def _decide(self, body):
        if self.mode == "direct" and not self.profile:
            # The unified CLI binds a verified artifact; the low-level API retains legacy aliases.
            if self.target:
                identity = self._post("/v1/winnow/inspect", body, 30).get("runtime", {})
                if identity.get("target_sha256") != self.target["model"]["sha256"]:
                    raise ValueError("Selected model/quantization does not match the server; restart with the matching winnow serve --model")
            response = self._post("/v1/systemone", body, 30)
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
        if not isinstance(extension, dict) or type(extension.get("temperature", 1)) not in (int, float) or extension.get("temperature", 1) != 1:
            raise ValueError("Reasoning decisions require raw T=1")
        images, image_hashes = validate_images(body, self.profile)
        keys, kind = keys_for(question), question["type"]
        definition = self.definition
        alias = definition["alias"] if definition else self.profile["alias"]
        policy = definition["policy"] if definition else None
        if body.get("model", alias) != alias:
            raise ValueError("Request model must match the selected policy")
        request = copy.deepcopy(body)
        request["model"] = alias
        request["winnow"] = dict(extension, diagnostics=True, temperature=1.0)
        native_seconds = self.profile["native_seconds"] if self.profile else 30
        if self.profile:
            request["winnow"]["reuse_prefix"] = False
        inspected = self._post("/v1/winnow/inspect", request, native_seconds)
        positions = check_inspection(inspected, self.profile, len(images)) if self.profile else None
        identity = inspected.get("runtime", {})
        expected_runtime = hashlib.sha256((ROOT / "runtime.lock.json").read_bytes()).hexdigest()
        if not self.profile and (
            not isinstance(identity, dict) or identity.get("runtime_sha256") != expected_runtime
            or identity.get("target_sha256") != definition["target"]["sha256"]
            or (self.mtp and identity.get("assistant_sha256") != definition["assistant"]["sha256"])
            or (not self.mtp and identity.get("assistant_sha256") is not None)
            or identity.get("resident_mtp") is not self.mtp
            or identity.get("vision") is not False
            or not isinstance(identity.get("context"), int) or identity["context"] < 512
        ):
            raise ValueError("Backend does not match the pinned experimental policy/profile")
        measured_profile = not self.profile and all(identity.get(k) == v for k, v in dict(
            context=8192, chat_context=8192, chat_parallel=1, memory_setting="auto",
            parallel=4, cache_type="q8_0", head="selected", pipeline="optimized",
            batch=definition["profile"]["batch"], ubatch=definition["profile"]["ubatch"]).items())
        # A failed/malformed direct decision remains an error; it is never fabricated.
        direct = self._post("/v1/systemone", request, native_seconds)
        logits, raw = native_probabilities(direct, qid, kind, keys, alias, self.mtp)
        if self.mode == "direct":
            direct.setdefault("winnow", {})["adaptive"] = dict(
                mode="direct", reasoning_mode="off", gate_applied=False, routed=False,
                completed_blend=False, fallback_reason=None, runtime_profile=self.profile["id"],
                image_sha256=image_hashes, native_context_positions=positions, measured_profile=False)
            return direct
        final = softmax(logits, policy["direct_temperature"])
        if policy["gate"] == "normalized_entropy":
            gate_value = -sum(p * math.log(p) for p in raw if p) / math.log(len(raw))
            routed = gate_value > policy["threshold"]
        else:
            gate_value = max(raw)
            routed = gate_value < policy["threshold"]
        if self.reasoning == "always":
            routed = True
        notes = []
        if not measured_profile:
            notes.append("Custom configuration: frozen policy retained; published calibration/quality/latency results do not validate these settings.")
        if self.profile:
            notes.append("Explicit runtime capability contract; image reasoning and this context configuration have no quality validation claim.")
        if self.reasoning == "always":
            notes.append("Always reasoning bypasses the gate; selective-policy quality and latency results do not validate this mode.")
        metadata = {
            "mode": self.mode, "policy_set_version": self.manifest["policy_set_version"],
            "policy_id": definition["id"], "experimental": True,
            "reasoning_mode": self.reasoning, "gate_applied": self.reasoning == "selective",
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
            "configuration_note": " ".join(notes) or None,
        }
        if self.profile:
            metadata.update(runtime_profile=self.profile["id"], image_sha256=image_hashes,
                            native_context_positions=positions, augmented_context_positions=None,
                            native_context_limit=identity["context"], chat_context_limit=identity["chat_context"])
        if routed:
            generation = dict(self.manifest["generation"])
            generation.pop("context")
            seconds = generation.pop("request_seconds")
            prompt_format = generation.pop("prompt_format", "bounded-v1")
            if self.profile:
                seconds = self.profile["generation_seconds"]
            prompt = (
                INSTRUCTION + "\nTask state:\n" + json.dumps(body["state"], separators=(",", ":"))
                + "\nDecision question:\n" + json.dumps(question, separators=(",", ":"))
            )
            if prompt_format == "e2b-canonical-v2":
                descriptions = rendered_options(question, keys)
                prompt = ("Analyze the task state and question. Work through the relevant rules and facts, "
                          "including intermediate steps when useful. Explain your reasoning as context for "
                          "a separate decision scorer.\nState:\n" + json.dumps(body["state"], ensure_ascii=False, separators=(",", ":"))
                          + "\nQuestion:\n" + json.dumps(question.get("instructions"), ensure_ascii=False, separators=(",", ":"))
                          + "\nOptions (in candidate order):\n" + "\n".join(
                              f"{i+1}. {json.dumps(value, ensure_ascii=False)}" for i, value in enumerate(descriptions)))
            if self.profile:
                # Escape token markers inside user data, as the native state compiler does.
                prompt = prompt.replace("<", "\\u003c")
            content = prompt
            if images:
                content = ([{"type": "text", "text": "Images (in order):\n"}]
                           + [{"type": "image_url", "image_url": {"url": url}} for url in images]
                           + [{"type": "text", "text": prompt}])
            generation.update(model=alias, messages=[{"role": "user", "content": content}],
                              cache_prompt=False, verbose=True, return_tokens=True, stream=False)
            try:
                result = self._post("/v1/chat/completions", generation, seconds)
                if not isinstance(result, dict):
                    raise ValueError("Malformed generation response")
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
                if self.profile and (verbose.get("truncated") is not False
                        or usage["prompt_tokens"] <= 0
                        or usage["prompt_tokens"] + usage["completion_tokens"] > identity["chat_context"]):
                    metadata["fallback_reason"] = "generation_context_limit_or_unverified"
                elif choice.get("finish_reason") != "stop" or verbose.get("stop_type") != "eos":
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
                        if self.profile:
                            checked = self._post("/v1/winnow/inspect", augmented_request, native_seconds)
                            metadata["augmented_context_positions"] = check_inspection(checked, self.profile, len(images))
                        augmented = self._post("/v1/systemone", augmented_request, native_seconds)
                        alogits, _ = native_probabilities(augmented, qid, kind, keys, definition["alias"], self.mtp)
                        calibrated = softmax(alogits, policy["augmented_temperature"])
                        metadata["native_input_tokens"] += augmented["usage"]["input_tokens"]
                        final = [(1 - policy["weight"]) * a + policy["weight"] * b
                                 for a, b in zip(final, calibrated)]
                        metadata["completed_blend"] = True
                    except ContextLimitError:
                        metadata["fallback_reason"] = "augmented_context_limit"
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
