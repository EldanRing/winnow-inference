#!/usr/bin/env python3
"""Launch Winnow with full GPU residency and configurable context capacity."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from profiles import PROFILES, resolve_profile
from adaptive_policy import load_policy
from verify_model import verify
from assets import MODELS, MODEL_ALIASES, selection
from launch_options import context_size

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PRESETS = json.loads((ROOT / "manifests/runtime-presets-v1.json").read_text())["presets"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        choices=["auto", *PROFILES],
        default="auto",
        help="Launch preset; explicit flags override it",
    )
    p.add_argument(
        "--model-dir", type=Path, default=ROOT / "models", help="Directory populated by setup.py"
    )
    p.add_argument("--model", type=Path, help="Custom GGUF; defaults to the release in --model-dir")
    p.add_argument("--preset", choices=list(RUNTIME_PRESETS), help="Exact Linux/CUDA 8K MTP artifact preset; direct decision API remains the default")
    p.add_argument("--alias", default="Winnow-12B", help="Model name advertised by the server")
    p.add_argument("--mmproj", type=Path)
    p.add_argument("--text-only", action="store_true", help="Do not load a vision projector")
    p.add_argument(
        "--dry-run", action="store_true", help="Print the resolved command without loading a model"
    )
    p.add_argument("--context", type=context_size)
    p.add_argument("--target", choices=[*MODEL_ALIASES, *MODELS], help="Verify and advertise the selected release artifact")
    p.add_argument(
        "--decision-context",
        type=int,
        help="Defaults to --context; no fixed memory partition",
    )
    p.add_argument("--head", choices=["selected", "full"], default="selected")
    p.add_argument("--pipeline", choices=["optimized", "reference"], default="optimized")
    p.add_argument(
        "--cache",
        choices=["f16", "q8_0"],
    )
    p.add_argument("--memory", choices=["auto", "exclusive"])
    p.add_argument(
        "--decision-parallel",
        type=int,
    )
    p.add_argument("--chat-parallel", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--ubatch", type=int)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--gpu", default="0")
    p.add_argument("--server", type=Path, default=ROOT / ".build/bin/winnow-server")
    p.add_argument(
        "--experimental-adaptive",
        choices=["nvfp4-entropy-v1", "e4b-calibrated50-v1", "e4b-calibrated75-g95-v1", "q8-fixed50-v1"],
        help="Explicit Linux/CUDA text-only 8K resident MTP4 profile; does not change direct default",
    )
    p.add_argument("--assistant", type=Path, help="Exact matching BF16 MTP GGUF for the opt-in profile")
    p.add_argument("--mtp", choices=["on", "off"], default="on",
                   help="Adaptive profile drafting; on preserves the original profile. Named MTP presets require on.")
    p.add_argument("--api-key-file", type=Path, help="Backend authentication key file")
    p.add_argument("--http-threads", type=int, help="HTTP workers; does not change GPU inference slots")
    a, extra = p.parse_known_args()
    experimental = None
    serving = None
    if a.preset:
        if a.mtp != "on":
            p.error("Named MTP presets require --mtp on; use winnow.py for an MTP-off profile")
        serving = RUNTIME_PRESETS[a.preset]
        if serving["status"] != "validated":
            print("Memory warning: Q8 vision + MTP exceeded the measured 16 GB profile; custom configurations are unvalidated.", file=sys.stderr, flush=True)
        if platform.system() != "Linux" or a.profile != "auto" or a.experimental_adaptive:
            p.error("Named MTP presets require Linux/CUDA; choose one preset without --profile/--experimental-adaptive")
        if a.text_only != serving["text_only"] and a.text_only:
            p.error("Vision preset requires its projector; use the explicit text preset for text only")
        a.text_only = serving["text_only"]
        for name, value in serving["settings"].items():
            if name == "decision_context":
                value = a.context
            if getattr(a, name) is None:
                setattr(a, name, value)
        a.alias = serving["alias"]
        a.model = a.model or a.model_dir / serving["target"]["file"]
        a.assistant = a.assistant or a.model_dir / "assistants" / serving["assistant"]["file"]
        if serving["projector"]:
            a.mmproj = a.mmproj or a.model_dir / serving["projector"]["file"]
        if any(flag not in {"--metrics", "--slots"} for flag in extra):
            p.error("Named preset allows only --metrics/--slots as extra runtime flags")
    if a.experimental_adaptive:
        if platform.system() != "Linux" or not a.text_only or a.mmproj:
            p.error("Experimental adaptive profile requires Linux/CUDA and explicit --text-only")
        if not a.model or (a.mtp == "on" and not a.assistant):
            p.error("Experimental profile requires --model and, with MTP on, matching --assistant")
        if a.mtp == "off" and a.assistant:
            p.error("MTP off does not load an assistant")
        _, experimental = load_policy(a.experimental_adaptive)
        expected = dict(context=8192, decision_context=8192, decision_parallel=4,
                        chat_parallel=1, cache="q8_0", head="selected", pipeline="optimized",
                        memory="auto")
        expected.update(experimental["profile"])
        for name, value in expected.items():
            if name == "decision_context":
                value = a.context
            if getattr(a, name) is None:
                setattr(a, name, value)
        if any(flag not in {"--metrics", "--slots"} for flag in extra):
            p.error("Experimental profile allows only --metrics/--slots as extra runtime flags")
        a.alias = experimental["alias"]
    elif a.assistant and not serving:
        p.error("--assistant requires an explicit MTP preset or --experimental-adaptive")
    try:
        selected, preset = resolve_profile(a.profile)
    except ValueError as error:
        p.error(str(error))
    for name, value in preset.items():
        if getattr(a, name) is None:
            setattr(a, name, value)
    a.batch = 2048 if a.batch is None else a.batch
    a.ubatch = 1024 if a.ubatch is None else a.ubatch
    if a.text_only and a.mmproj:
        p.error("Choose --text-only or --mmproj, not both")
    if a.model is None:
        release = json.loads((ROOT / "manifests/models.json").read_text())["release"]
        a.model = a.model_dir / release["model"]["file"]
        if not a.text_only and a.mmproj is None:
            a.mmproj = a.model_dir / release["projector"]["file"]
    if (serving or (experimental and a.mtp == "on")) and (a.chat_parallel != 1 or a.memory != "auto"):
        p.error("MTP requires one chat slot (--chat-parallel 1) and --memory auto")
    if a.ubatch > a.batch:
        p.error("--ubatch must not exceed --batch")
    if min(a.decision_parallel, a.chat_parallel, a.batch, a.ubatch, a.threads) < 1:
        p.error("Parallel counts, batches and threads must be positive")
    if a.context < 512 or (a.decision_context and a.decision_context < 512):
        p.error("context must be at least 512")
    if not a.model.is_file() or (a.mmproj and not a.mmproj.is_file()):
        p.error(
            "Model/projector file not found. Run python3 scripts/setup.py, use --model-dir for an existing download, or --text-only to skip vision."
        )
    if not a.server.is_file():
        p.error("server not built; run scripts/build.py")
    forbidden = ("--spec-", "--model-url", "--mmproj-url", "--hf-", "--lora", "-hf", "-mu", "-mmu", "-md", "--model-draft")
    if any(flag.startswith(forbidden) for flag in extra):
        p.error("Use the explicit Winnow model/vision/MTP flags; raw asset and speculation overrides are unsupported")
    env = os.environ.copy()
    # Keep unrelated environment settings; remove inherited asset/mode controls.
    for key in list(env):
        if key.startswith(("LLAMA_ARG_MODEL", "LLAMA_ARG_MMPROJ", "LLAMA_ARG_SPEC", "LLAMA_ARG_HF", "LLAMA_ARG_LORA")):
            env.pop(key)
    env["WINNOW_MANAGED_LAUNCH"] = "1"
    # Stale experimental variables must not enable features in the normal launcher.
    for name in ["WINNOW_RESIDENT_MTP", "WINNOW_RESIDENT_NGRAM", "WINNOW_NATIVE_CHECKPOINT",
                 "WINNOW_COMMITTED_CHAT", "WINNOW_SPEC_STATS", "WINNOW_PROCESS_STATS",
                 "WINNOW_TARGET_SHA256", "WINNOW_ASSISTANT_SHA256", "WINNOW_RESIDENT_VISION_MTP"]:
        env.pop(name, None)
    env["WINNOW_RESIDENT_MTP"] = "0"
    env["WINNOW_RESIDENT_VISION_MTP"] = "0"
    if a.target:
        try:
            spec, _ = selection(a.target)
            verify(a.model, spec["model"])
            env["WINNOW_TARGET_SHA256"] = spec["model"]["sha256"]
        except (OSError, ValueError) as error:
            p.error(str(error))
    if serving:
        try:
            verify(a.model, serving["target"])
            verify(a.assistant, serving["assistant"])
            if serving["projector"]:
                verify(a.mmproj, serving["projector"])
        except (OSError, ValueError) as error:
            p.error(str(error))
        env.update(WINNOW_RESIDENT_MTP="1", WINNOW_RESIDENT_VISION_MTP="0" if a.text_only else "1",
                   WINNOW_TARGET_SHA256=serving["target"]["sha256"],
                   WINNOW_ASSISTANT_SHA256=serving["assistant"]["sha256"])
    if experimental:
        try:
            verify(a.model, experimental["target"])
            if a.mtp == "on":
                verify(a.assistant, experimental["assistant"])
        except (OSError, ValueError) as error:
            p.error(str(error))
        env.update(WINNOW_RESIDENT_MTP="1" if a.mtp == "on" else "0",
                   WINNOW_TARGET_SHA256=experimental["target"]["sha256"], LLAMA_ARG_N_PREDICT="-1")
        if a.mtp == "on":
            env["WINNOW_ASSISTANT_SHA256"] = experimental["assistant"]["sha256"]
    env.update(
        WINNOW_CONTEXT=str(a.decision_context or a.context),
        WINNOW_CHAT_CONTEXT=str(a.context),
        WINNOW_CHAT_PARALLEL=str(a.chat_parallel),
        WINNOW_HEAD=a.head,
        WINNOW_CACHE=a.cache,
        WINNOW_PIPELINE=a.pipeline,
        WINNOW_MEMORY=a.memory,
        WINNOW_PARALLEL=str(a.decision_parallel),
        WINNOW_BATCH=str(a.batch),
        WINNOW_UBATCH=str(a.ubatch),
        CUDA_VISIBLE_DEVICES=a.gpu,
    )
    args = [
        str(a.server.resolve()),
        "--model",
        str(a.model.resolve()),
        "--alias",
        a.alias,
        "--ctx-size",
        str(a.context),
        "--parallel",
        str(a.chat_parallel),
        "--n-gpu-layers",
        "999",
        "--fit",
        "off",
        "--flash-attn",
        "on",
        "--cache-type-k",
        a.cache,
        "--cache-type-v",
        a.cache,
        "--no-context-shift",
        "--offline",
        "--lazy-mode",
        "off",
        "--split-mode",
        "none",
        "--override-tensor",
        r"^(token_embd|per_layer_token_embd)\.weight$="
        + ("MTL0" if platform.system() == "Darwin" else "CUDA0"),
        "--batch-size",
        str(a.batch),
        "--ubatch-size",
        str(a.ubatch),
        "--threads",
        str(a.threads),
        "--host",
        a.host,
        "--port",
        str(a.port),
        "--jinja",
        "--reasoning",
        "off",
        "--no-warmup",
        "--cache-ram",
        "0",
        "--cors-origins",
        "",
    ]
    if a.text_only:
        args += ["--no-mmproj"]
    if a.mmproj:
        args += ["--mmproj", str(a.mmproj.resolve())]
    if a.api_key_file:
        args += ["--api-key-file", str(a.api_key_file.resolve())]
    if a.http_threads is not None:
        if not 1 <= a.http_threads <= 256:
            p.error("HTTP workers must be1–256")
        args += ["--threads-http", str(a.http_threads)]
    if (experimental and a.mtp == "on") or serving:
        args += ["--spec-type", "draft-mtp", "--spec-draft-model", str(a.assistant.resolve()),
                 "--spec-draft-n-max", "4", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
                 "--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0",
                 "--spec-draft-ngl", "999"]
    else:
        args += ["--spec-type", "none"]
    args += extra
    if a.dry_run:
        print(
            json.dumps(
                {
                    "profile": a.preset or a.experimental_adaptive or selected,
                    "command": args,
                    "environment": {
                        k: v
                        for k, v in env.items()
                        if k.startswith("WINNOW_")
                        and k not in {"WINNOW_API_KEY", "WINNOW_API_KEY_FILE"}
                    },
                },
                indent=2,
            )
        )
        return
    print(
        f"Winnow profile: {a.preset or a.experimental_adaptive or selected}. API: http://{a.host}:{a.port}\nOnce ready, use the model-specific client commands in docs/QUICKSTART.md.",
        flush=True,
    )
    os.execve(args[0], args, env)


if __name__ == "__main__":
    main()
