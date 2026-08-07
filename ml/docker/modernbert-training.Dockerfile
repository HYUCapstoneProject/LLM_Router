# syntax=docker/dockerfile:1.7

ARG PYTORCH_IMAGE=pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755
FROM ${PYTORCH_IMAGE}

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="LLM Router ModernBERT Training"
LABEL org.opencontainers.image.description="Shared ModernBERT training runtime for local smoke and Kubeflow/GKE"
LABEL org.opencontainers.image.source="https://github.com/HYUCapstoneProject/LLM_Router"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MODERNBERT_RUNTIME_CONTRACT=/opt/llm-router/config/modernbert_runtime_cu124_v1.json

WORKDIR /opt/llm-router

COPY ml/requirements-training.lock /tmp/requirements/requirements-training.lock

RUN python -m pip install \
        --requirement /tmp/requirements/requirements-training.lock

COPY ml/docker/modernbert-training.pyproject.toml /opt/llm-router/ml/pyproject.toml
RUN mkdir -p \
        /opt/llm-router/ml/src/llm_router_labeling/training \
        /opt/llm-router/ml/src/llm_router_labeling/shared \
        /opt/llm-router/config
COPY ml/src/llm_router_labeling/__init__.py /opt/llm-router/ml/src/llm_router_labeling/__init__.py
COPY ml/src/llm_router_labeling/training/__init__.py /opt/llm-router/ml/src/llm_router_labeling/training/__init__.py
COPY ml/src/llm_router_labeling/training/modernbert_training.py /opt/llm-router/ml/src/llm_router_labeling/training/modernbert_training.py
COPY ml/src/llm_router_labeling/training/modernbert_training_cli.py /opt/llm-router/ml/src/llm_router_labeling/training/modernbert_training_cli.py
COPY ml/src/llm_router_labeling/training/modernbert_runtime.py /opt/llm-router/ml/src/llm_router_labeling/training/modernbert_runtime.py
COPY ml/src/llm_router_labeling/shared/__init__.py /opt/llm-router/ml/src/llm_router_labeling/shared/__init__.py
COPY ml/src/llm_router_labeling/shared/dataset_io.py /opt/llm-router/ml/src/llm_router_labeling/shared/dataset_io.py
COPY ml/config/modernbert_runtime_cu124_v1.json /opt/llm-router/config/modernbert_runtime_cu124_v1.json

RUN python -m pip install --no-deps /opt/llm-router/ml \
    && python -c "import torch, transformers; from pathlib import Path; from llm_router_labeling.training.modernbert_runtime import collect_runtime_versions, load_runtime_contract, validate_runtime_contract; contract = load_runtime_contract(Path('/opt/llm-router/config/modernbert_runtime_cu124_v1.json')); validate_runtime_contract(contract, collect_runtime_versions(torch, transformers), device='cpu'); assert torch.version.cuda == '12.4'" \
    && modernbert-train --help >/dev/null \
    && groupadd --gid 10001 trainer \
    && useradd --uid 10001 --gid trainer --create-home --shell /usr/sbin/nologin trainer \
    && mkdir -p /workspace /cache/huggingface \
    && chown -R trainer:trainer /workspace /cache

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["modernbert-train"]
CMD ["--help"]
