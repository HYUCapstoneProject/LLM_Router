# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
FROM ${PYTHON_IMAGE}

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="LLM Router Data Processing"
LABEL org.opencontainers.image.description="CPU-only initial data and feedback retraining runtime"
LABEL org.opencontainers.image.source="https://github.com/HYUCapstoneProject/LLM_Router"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/llm-router

COPY ml/requirements-lock.txt /tmp/requirements/requirements-lock.txt
RUN python -m pip install \
        --requirement /tmp/requirements/requirements-lock.txt \
    && python -m pip install "google-cloud-storage>=3.0,<4.0"

COPY ml/docker/data-processing.pyproject.toml /opt/llm-router/ml/pyproject.toml
RUN mkdir -p \
        /opt/llm-router/ml/src/llm_router_labeling/training \
        /opt/llm-router/ml/src/llm_router_labeling/evaluation \
        /opt/llm-router/ml/src/llm_router_labeling/shared \
        /workspace/ml/config

COPY ml/src/llm_router_labeling/__init__.py /opt/llm-router/ml/src/llm_router_labeling/__init__.py
COPY ml/src/llm_router_labeling/training/__init__.py /opt/llm-router/ml/src/llm_router_labeling/training/__init__.py
COPY ml/src/llm_router_labeling/training/routerbench_cli.py /opt/llm-router/ml/src/llm_router_labeling/training/routerbench_cli.py
COPY ml/src/llm_router_labeling/training/routerbench_labeling.py /opt/llm-router/ml/src/llm_router_labeling/training/routerbench_labeling.py
COPY ml/src/llm_router_labeling/training/routerbench_preprocessing.py /opt/llm-router/ml/src/llm_router_labeling/training/routerbench_preprocessing.py
COPY ml/src/llm_router_labeling/training/feedback_retraining_cli.py /opt/llm-router/ml/src/llm_router_labeling/training/feedback_retraining_cli.py
COPY ml/src/llm_router_labeling/training/feedback_retraining_preprocessing.py /opt/llm-router/ml/src/llm_router_labeling/training/feedback_retraining_preprocessing.py
COPY ml/src/llm_router_labeling/evaluation/__init__.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/__init__.py
COPY ml/src/llm_router_labeling/evaluation/routerarena_cli.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/routerarena_cli.py
COPY ml/src/llm_router_labeling/evaluation/routerarena_preprocessing.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/routerarena_preprocessing.py
COPY ml/src/llm_router_labeling/shared/__init__.py /opt/llm-router/ml/src/llm_router_labeling/shared/__init__.py
COPY ml/src/llm_router_labeling/shared/dataset_io.py /opt/llm-router/ml/src/llm_router_labeling/shared/dataset_io.py
COPY ml/src/llm_router_labeling/shared/text_normalization.py /opt/llm-router/ml/src/llm_router_labeling/shared/text_normalization.py
COPY ml/src/llm_router_labeling/shared/object_store.py /opt/llm-router/ml/src/llm_router_labeling/shared/object_store.py
COPY ml/src/llm_router_labeling/shared/observability.py /opt/llm-router/ml/src/llm_router_labeling/shared/observability.py
COPY ml/src/llm_router_labeling/shared/structured_text.py /opt/llm-router/ml/src/llm_router_labeling/shared/structured_text.py
COPY ml/config/routerbench_labeling.json /workspace/ml/config/routerbench_labeling.json
COPY ml/config/routerbench_labeling_v2.json /workspace/ml/config/routerbench_labeling_v2.json
COPY ml/config/routerarena_preprocessing.json /workspace/ml/config/routerarena_preprocessing.json
COPY ml/config/feedback_retraining_preprocessing.json /workspace/ml/config/feedback_retraining_preprocessing.json

RUN python -m pip install --no-deps /opt/llm-router/ml \
    && routerbench-labeling --help >/dev/null \
    && routerarena-preprocessing --help >/dev/null \
    && escalation-feedback-preprocessing --help >/dev/null \
    && groupadd --gid 10001 datajob \
    && useradd --uid 10001 --gid datajob --create-home --shell /usr/sbin/nologin datajob \
    && mkdir -p /workspace/input /workspace/output \
    && chown -R datajob:datajob /workspace

USER 10001:10001
WORKDIR /workspace

CMD ["escalation-feedback-preprocessing", "--help"]
