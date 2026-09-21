# Exportable, allowlisted build context for packaging checks.
FROM scratch AS source
COPY . /source

# CUDA 13.0.2 Ubuntu 24.04, linux/amd64; no model downloads during the build.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c AS build
ARG CUDA_ARCH=120
ARG BUILD_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=source /source/CMakeLists.txt /source/runtime.lock.json /app/
COPY --from=source /source/native /app/native
COPY --from=source /source/patches /app/patches
COPY --from=source /source/tests /app/tests
COPY --from=source /source/scripts/build.py /app/scripts/build.py
RUN python3 scripts/build.py --jobs "${BUILD_JOBS}" --cuda-arch "${CUDA_ARCH}"

# Ship runtime libraries, the executable and public tools, not the compiler/build tree.
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ca-certificates libssl3t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
LABEL org.opencontainers.image.source="https://github.com/EldanRing/winnow-inference"
LABEL org.opencontainers.image.licenses="MIT"
WORKDIR /app
COPY --from=build /app/.build/bin/winnow-server /app/.build/bin/winnow-server
COPY --from=source /source/scripts /app/scripts
COPY --from=source /source/manifests /app/manifests
COPY --from=source /source/examples /app/examples
COPY --from=source /source/tests /app/tests
COPY --from=source /source/third_party /app/third_party
COPY --from=build /app/.runtime/llama.cpp/vendor /app/third_party/llama.cpp-vendor
COPY --from=build /app/.runtime/llama.cpp/LICENSE /app/third_party/llama.cpp-LICENSE
COPY --from=source /source/LICENSE /source/README.md /source/runtime.lock.json /app/
RUN useradd --uid 10001 --create-home winnow
USER winnow
EXPOSE 8091
HEALTHCHECK --interval=30s --start-period=120s --timeout=5s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=3)" || exit 1
ENTRYPOINT ["python3", "scripts/serve.py", "--host", "0.0.0.0", "--model-dir", "/models"]
