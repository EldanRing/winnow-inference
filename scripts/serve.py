#!/usr/bin/env python3
"""Launch Winnow with full GPU residency and configurable context capacity."""

import argparse
import json
import os
import platform
from pathlib import Path

from profiles import PROFILES, resolve_profile

ROOT = Path(__file__).resolve().parents[1]


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
    p.add_argument("--mmproj", type=Path)
    p.add_argument("--text-only", action="store_true", help="Do not load a vision projector")
    p.add_argument(
        "--dry-run", action="store_true", help="Print the resolved command without loading a model"
    )
    p.add_argument("--context", type=int)
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
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--ubatch", type=int, default=1024)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--gpu", default="0")
    p.add_argument("--server", type=Path, default=ROOT / ".build/bin/winnow-server")
    a, extra = p.parse_known_args()
    try:
        selected, preset = resolve_profile(a.profile)
    except ValueError as error:
        p.error(str(error))
    for name, value in preset.items():
        if getattr(a, name) is None:
            setattr(a, name, value)
    if a.text_only and a.mmproj:
        p.error("Choose --text-only or --mmproj, not both")
    if a.model is None:
        release = json.loads((ROOT / "manifests/models.json").read_text())["release"]
        a.model = a.model_dir / release["model"]["file"]
        if not a.text_only and a.mmproj is None:
            a.mmproj = a.model_dir / release["projector"]["file"]
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
    env = os.environ.copy()
    env.update(
        WINNOW_CONTEXT=str(a.decision_context or a.context),
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
        "Winnow-12B",
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
    if a.mmproj:
        args += ["--mmproj", str(a.mmproj.resolve())]
    args += extra
    if a.dry_run:
        print(
            json.dumps(
                {
                    "profile": selected,
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
        f"Winnow profile: {selected}. API: http://{a.host}:{a.port}\nOnce ready, open a second terminal and run: python3 examples/client.py",
        flush=True,
    )
    os.execve(args[0], args, env)


if __name__ == "__main__":
    main()
