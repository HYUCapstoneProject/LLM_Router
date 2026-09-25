# syntax=docker/dockerfile:1.7

ARG PYTORCH_IMAGE=pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755
FROM ${PYTORCH_IMAGE}

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="LLM Router ModernBERT Training"
LABEL org.opencontainers.image.description="Shared ModernBERT training runtime for local smoke and Vertex AI Custom Training"
LABEL org.opencontainers.image.source="https://github.com/HYUCapstoneProject/LLM_Router"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MODERNBERT_RUNTIME_CONTRACT=/opt/llm-router/config/modernbert_runtime_cu124_v1.json

WORKDIR /opt/llm-router

# ModernBERT enables torch.compile on CUDA, and TorchInductor generates C/C++
# kernel code that it compiles at run time. The PyTorch base image ships no
# toolchain, so without this the first forward pass dies with
# "RuntimeError: Failed to find C compiler." The failure only appears on GPU —
# MPS and eager CPU paths skip compilation entirely — so it cannot be caught
# outside a real CUDA job.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ml/requirements-training.lock /tmp/requirements/requirements-training.lock

RUN python -m pip install \
        --requirement /tmp/requirements/requirements-training.lock \
    && python -m pip install "google-cloud-storage>=3.0,<4.0"

COPY ml/docker/modernbert-training.pyproject.toml /opt/llm-router/ml/pyproject.toml
RUN mkdir -p \
        /opt/llm-router/ml/src/llm_router_labeling/training \
        /opt/llm-router/ml/src/llm_router_labeling/objective_comparison \
        /opt/llm-router/ml/src/llm_router_labeling/shared \
        /opt/llm-router/config
COPY ml/src/llm_router_labeling/__init__.py /opt/llm-router/ml/src/llm_router_labeling/__init__.py
COPY ml/src/llm_router_labeling/training/__init__.py /opt/llm-router/ml/src/llm_router_labeling/training/__init__.py
COPY ml/src/llm_router_labeling/training/modernbert_training.py /opt/llm-router/ml/src/llm_router_labeling/training/modernbert_training.py
COPY ml/src/llm_router_labeling/training/modernbert_training_cli.py /opt/llm-router/ml/src/llm_router_labeling/training/modernbert_training_cli.py
COPY ml/src/llm_router_labeling/training/modernbert_checkpointing.py /opt/llm-router/ml/src/llm_router_labeling/training/modernbert_checkpointing.py
COPY ml/src/llm_router_labeling/training/modernbert_runtime.py /opt/llm-router/ml/src/llm_router_labeling/training/modernbert_runtime.py
# Include only the loss module as a namespace package. The research package's
# __init__.py imports dataset preparation, which the training image does not use.
COPY ml/src/llm_router_labeling/objective_comparison/objectives.py /opt/llm-router/ml/src/llm_router_labeling/objective_comparison/objectives.py
COPY ml/src/llm_router_labeling/shared/__init__.py /opt/llm-router/ml/src/llm_router_labeling/shared/__init__.py
COPY ml/src/llm_router_labeling/shared/dataset_io.py /opt/llm-router/ml/src/llm_router_labeling/shared/dataset_io.py
COPY ml/src/llm_router_labeling/shared/object_store.py /opt/llm-router/ml/src/llm_router_labeling/shared/object_store.py
COPY ml/src/llm_router_labeling/shared/observability.py /opt/llm-router/ml/src/llm_router_labeling/shared/observability.py
COPY ml/config/modernbert_runtime_cu124_v1.json /opt/llm-router/config/modernbert_runtime_cu124_v1.json

RUN python -m pip install --no-deps /opt/llm-router/ml \
    && python -c "import torch, transformers; from pathlib import Path; from llm_router_labeling.training.modernbert_runtime import collect_runtime_versions, load_runtime_contract, validate_runtime_contract; contract = load_runtime_contract(Path('/opt/llm-router/config/modernbert_runtime_cu124_v1.json')); validate_runtime_contract(contract, collect_runtime_versions(torch, transformers), device='cpu'); assert torch.version.cuda == '12.4'" \
    && modernbert-train --help >/dev/null \
    && python -c "from llm_router_labeling.objective_comparison.objectives import primary_row_losses; assert callable(primary_row_losses)" \
    && cc --version >/dev/null \
    && c++ --version >/dev/null \
    && groupadd --gid 10001 trainer \
    && useradd --uid 10001 --gid trainer --create-home --shell /usr/sbin/nologin trainer \
    && mkdir -p /workspace /cache/huggingface \
    && chown -R trainer:trainer /workspace /cache

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["modernbert-train"]
CMD ["--help"]
