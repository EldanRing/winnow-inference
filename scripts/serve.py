#!/usr/bin/env python3
"""Launch Winnow with full GPU residency and configurable context capacity."""

import argparse
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--mmproj", type=Path)
    p.add_argument("--context", type=int, default=65536)
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
        default="f16" if platform.system() == "Darwin" else "q8_0",
    )
    p.add_argument("--memory", choices=["auto", "exclusive"], default="auto")
    p.add_argument(
        "--decision-parallel",
        type=int,
        default=1 if platform.system() == "Darwin" else 4,
    )
    p.add_argument("--chat-parallel", type=int, default=1)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--ubatch", type=int, default=1024)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--gpu", default="0")
    p.add_argument("--server", type=Path, default=ROOT / ".build/bin/winnow-server")
    a, extra = p.parse_known_args()
    if a.context < 512 or (a.decision_context and a.decision_context < 512):
        p.error("context must be at least 512")
    if not a.model.is_file() or (a.mmproj and not a.mmproj.is_file()):
        p.error("model/projector file not found")
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
    os.execve(args[0], args, env)


if __name__ == "__main__":
    main()
