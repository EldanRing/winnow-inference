"""Trusted operator profiles for image-aware decisions, separate from quality evidence."""

import base64
import copy
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 32 * 1024 * 1024


class RequestCancelled(Exception):
    """The caller went away. Never turn cancellation into a fallback decision."""


class ContextLimitError(ValueError):
    pass


def read_object(value):
    result = copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text())
    if not isinstance(result, dict):
        raise ValueError("Expected a trusted JSON object")
    return result


def load_profile(value):
    """A path/dict is operator configuration; never take it from a decision request."""
    presets = json.loads((ROOT / "manifests/runtime-presets-v1.json").read_text())["presets"]
    if isinstance(value, str) and value in presets:
        preset = presets[value]
        if preset["status"] != "validated" or preset["text_only"]:
            raise ValueError("Unsupported image reasoning preset; choose a supported vision profile")
        settings = preset["settings"]
        expected = dict(runtime_sha256=hashlib.sha256((ROOT / "runtime.lock.json").read_bytes()).hexdigest(),
                        target_sha256=preset["target"]["sha256"], assistant_sha256=preset["assistant"]["sha256"],
                        resident_mtp=True, vision=True, context=settings["decision_context"],
                        chat_context=settings["context"], cache_type=settings["cache"],
                        memory_setting=settings["memory"], parallel=settings["decision_parallel"],
                        **{k: settings[k] for k in ("chat_parallel", "head", "pipeline", "batch", "ubatch")})
        value = dict(schema_version=1, id=value, alias=preset["alias"], runtime=expected,
                     max_images=16, native_seconds=30, generation_seconds=75,
                     launcher_requirements={"no_context_shift": True,
                                            "projector_sha256": preset["projector"]["sha256"],
                                            "mtp_draft_n_max": 4})
    profile = read_object(value)
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1 or not all(isinstance(profile.get(k), str) and profile[k] for k in ("id", "alias")):
        raise ValueError("Invalid runtime profile version/id/alias")
    expected = profile.get("runtime", {})
    required = {"runtime_sha256", "target_sha256", "assistant_sha256", "resident_mtp", "vision",
                "context", "chat_context", "cache_type", "memory_setting", "parallel", "chat_parallel",
                "head", "pipeline", "batch", "ubatch"}
    if not isinstance(expected, dict) or set(expected) != required:
        raise ValueError("Runtime profile must pin every supported identity and capacity field")
    for key in ("runtime_sha256", "target_sha256", "assistant_sha256"):
        if key == "assistant_sha256" and expected[key] is None and expected["resident_mtp"] is False:
            continue
        if not isinstance(expected[key], str) or not re.fullmatch(r"[0-9a-f]{64}", expected[key]):
            raise ValueError("Invalid pinned artifact hash: " + key)
    if type(expected["resident_mtp"]) is not bool or type(expected["vision"]) is not bool:
        raise ValueError("Profile vision/MTP capabilities must be explicit booleans")
    if not expected["resident_mtp"] and expected["assistant_sha256"] is not None:
        raise ValueError("MTP-off profile must not advertise an assistant")
    for key in ("context", "chat_context", "parallel", "chat_parallel", "batch", "ubatch"):
        if type(expected[key]) is not int or expected[key] < (512 if "context" in key else 1):
            raise ValueError("Invalid runtime capacity: " + key)
    if (expected["chat_parallel"] != 1 or expected["memory_setting"] != "auto"
            or expected["cache_type"] not in {"q8_0", "f16"}
            or expected["head"] != "selected" or expected["pipeline"] != "optimized"
            or expected["ubatch"] > expected["batch"]):
        raise ValueError("Unsupported reasoning runtime configuration")
    for key, minimum, maximum in (("max_images", 0, 16), ("native_seconds", 1, 300), ("generation_seconds", 1, 300)):
        if type(profile.get(key)) is not int or not minimum <= profile[key] <= maximum:
            raise ValueError("Invalid profile limit: " + key)
    if bool(profile["max_images"]) != expected["vision"]:
        raise ValueError("Image limit must match vision capability")
    launch = profile.get("launcher_requirements", {})
    if not isinstance(launch, dict) or launch.get("no_context_shift") is not True:
        raise ValueError("Profile requires a launcher that disables chat context shifting")
    if expected["vision"] and not re.fullmatch(r"[0-9a-f]{64}", str(launch.get("projector_sha256", ""))):
        raise ValueError("Vision profile requires a launcher-verified projector hash")
    if expected["resident_mtp"] and launch.get("mtp_draft_n_max") != 4:
        raise ValueError("Only MTP4 profiles are supported")
    return profile


def validate_images(body, profile):
    """Native and chat receive the same canonical data URLs, in the same order."""
    ext = body.get("winnow", {})
    images = ext.get("images", [])
    if not isinstance(images, list) or len(images) > (profile["max_images"] if profile else 0):
        raise ValueError("Images require an explicit supported vision runtime profile (at most 16)")
    hashes = []
    for url in images:
        if not isinstance(url, str) or not re.fullmatch(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/]*={0,2}", url):
            raise ValueError("Images must be base64 image data URLs, not remote URLs or filesystem paths")
        encoded = url.partition(",")[2]
        if not encoded or len(encoded) % 4 or len(encoded) > 24 * 1024 * 1024:
            raise ValueError("Invalid image payload size")
        decoded = base64.b64decode(encoded, validate=True)
        if base64.b64encode(decoded).decode() != encoded:
            raise ValueError("Invalid noncanonical base64 image")
        hashes.append(hashlib.sha256(decoded).hexdigest())
    if len(json.dumps(body, allow_nan=False).encode()) > MAX_BODY:
        raise ValueError("Request exceeds 32 MiB")
    return images, hashes


def check_inspection(result, profile, image_count):
    if not isinstance(result, dict) or not isinstance(result.get("runtime"), dict):
        raise ValueError("Malformed runtime inspection")
    identity = result.get("runtime", {})
    for key, value in profile["runtime"].items():
        actual = identity.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError("Backend does not match runtime profile: " + key)
    prefix, suffix, image_tokens = result.get("prefix_tokens"), result.get("suffix_tokens"), result.get("image_tokens")
    if (type(prefix) is not int or prefix < 0 or not isinstance(suffix, list) or len(suffix) != 1
            or type(suffix[0]) is not int or suffix[0] < 0
            or type(image_tokens) is not int or image_tokens < 0 or bool(image_tokens) != bool(image_count)):
        raise ValueError("Missing or invalid native text/image position counts")
    positions = prefix + suffix[0] + 1
    if positions > profile["runtime"]["context"]:
        raise ContextLimitError("Native input plus answer exceeds the configured context")
    return positions
