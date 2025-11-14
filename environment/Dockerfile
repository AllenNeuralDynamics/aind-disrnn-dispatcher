# hash:sha256:4f567b195522ba414263860e03ae193d69e7c4114b288c652000066eab30db33
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

# Install Python dependencies
RUN pip install --no-cache-dir \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    pydantic>=2.0.0

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
