ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
FROM ${PYTHON_IMAGE}

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="LLM Router Model Promotion"
LABEL org.opencontainers.image.description="CPU-only model packaging and deploy gate runtime"
LABEL org.opencontainers.image.source="https://github.com/HYUCapstoneProject/LLM_Router"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/llm-router

COPY ml/docker/model-promotion.pyproject.toml /opt/llm-router/ml/pyproject.toml
RUN mkdir -p \
        /opt/llm-router/ml/src/llm_router_labeling/deployment \
        /opt/llm-router/ml/src/llm_router_labeling/shared
COPY ml/src/llm_router_labeling/__init__.py /opt/llm-router/ml/src/llm_router_labeling/__init__.py
COPY ml/src/llm_router_labeling/deployment/__init__.py /opt/llm-router/ml/src/llm_router_labeling/deployment/__init__.py
COPY ml/src/llm_router_labeling/deployment/model_package.py /opt/llm-router/ml/src/llm_router_labeling/deployment/model_package.py
COPY ml/src/llm_router_labeling/deployment/model_package_cli.py /opt/llm-router/ml/src/llm_router_labeling/deployment/model_package_cli.py
COPY ml/src/llm_router_labeling/deployment/deploy_gate.py /opt/llm-router/ml/src/llm_router_labeling/deployment/deploy_gate.py
COPY ml/src/llm_router_labeling/deployment/deploy_gate_cli.py /opt/llm-router/ml/src/llm_router_labeling/deployment/deploy_gate_cli.py
COPY ml/src/llm_router_labeling/shared/__init__.py /opt/llm-router/ml/src/llm_router_labeling/shared/__init__.py
COPY ml/src/llm_router_labeling/shared/dataset_io.py /opt/llm-router/ml/src/llm_router_labeling/shared/dataset_io.py

RUN python -m pip install --no-deps /opt/llm-router/ml \
    && modernbert-package --help >/dev/null \
    && modernbert-deploy-gate --help >/dev/null \
    && groupadd --gid 10001 promoter \
    && useradd --uid 10001 --gid promoter --create-home --shell /usr/sbin/nologin promoter \
    && mkdir -p /workspace/input /workspace/output \
    && chown -R promoter:promoter /workspace

USER 10001:10001
WORKDIR /workspace

CMD ["modernbert-package", "--help"]
