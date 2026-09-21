"""Shared launch presets. Explicit launcher flags always override a preset."""

import platform

PROFILES = {
    "5070ti-64k": {
        "context": 65536,
        "cache": "q8_0",
        "decision_parallel": 4,
        "chat_parallel": 1,
        "memory": "exclusive",
    },
    "apple-silicon": {
        "context": 65536,
        "cache": "f16",
        "decision_parallel": 1,
        "chat_parallel": 1,
        "memory": "exclusive",
    },
}


def resolve_profile(name="auto"):
    system = platform.system()
    if system not in {"Linux", "Darwin"}:
        raise ValueError(
            "Use Linux with NVIDIA CUDA or macOS on Apple Silicon. Native Windows is not supported."
        )
    if system == "Darwin" and platform.machine() != "arm64":
        raise ValueError(
            "The macOS profile requires Apple Silicon and an arm64 Python installation."
        )
    selected = ("apple-silicon" if system == "Darwin" else "5070ti-64k") if name == "auto" else name
    if (selected == "apple-silicon") != (system == "Darwin"):
        raise ValueError(f"Profile {selected} is not supported on {system}.")
    return selected, dict(PROFILES[selected])
