# One image for the SageMaker jobs (feature processing, training, live run). x86: SageMaker training has no ARM instances.
# Built from the committed tree only, so the image is exactly GIT_COMMIT (README, "Running on AWS"):
#   git archive HEAD | docker build --platform linux/amd64 --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t enefit -
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src src
COPY tests tests
RUN pip install --no-cache-dir ".[dev,aws]"
ARG GIT_COMMIT
# ENEFIT_DATA_DIR is set per job (the S3 mount); the caches go to the container's own disk
ENV GIT_COMMIT=$GIT_COMMIT ENEFIT_ROOT=/app ENEFIT_CACHE_DIR=/tmp/enefit-cache PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
