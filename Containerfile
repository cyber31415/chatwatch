# chatwatch's runtime: Python and the five packages the dashboard uses (no torch). start.sh builds it on first run:
#   podman build -f Containerfile -t "chatwatch:0.9-beta-$(sha256sum Containerfile | cut -c1-8)" .
# This folder is bind-mounted at /work; every file the tool writes lands under its data/ and models/.
FROM docker.io/library/python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/work/src \
    PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir aiohttp==3.12.15 websockets==13.1 numpy==2.1.2 onnxruntime==1.20.1 tokenizers==0.20.3
WORKDIR /work
