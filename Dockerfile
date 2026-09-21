# Exportable, allowlisted build context for packaging checks.
FROM scratch AS source
COPY . /source

# CUDA 13.0.2 Ubuntu 24.04, linux/amd64; no model downloads during the build.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c
ARG CUDA_ARCH=120
ARG BUILD_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=source /source/ .
RUN python3 scripts/build.py --jobs ${BUILD_JOBS} --cuda-arch ${CUDA_ARCH}
RUN useradd --uid 10001 --create-home winnow
USER winnow
EXPOSE 8091
ENTRYPOINT ["python3", "scripts/serve.py", "--host", "0.0.0.0"]
