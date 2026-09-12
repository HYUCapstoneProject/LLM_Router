ARG PYTORCH_IMAGE=pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755
FROM ${PYTORCH_IMAGE}

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="LLM Router ModernBERT Evaluation"
LABEL org.opencontainers.image.description="RouterArena CPU evaluation runtime for Vertex AI Pipelines"
LABEL org.opencontainers.image.source="https://github.com/HYUCapstoneProject/LLM_Router"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /opt/llm-router

COPY ml/requirements-training.lock /tmp/requirements/requirements-evaluation.lock
RUN python -m pip install --requirement /tmp/requirements/requirements-evaluation.lock \
    && python -m pip install "google-cloud-storage>=3.0,<4.0"

COPY ml/docker/modernbert-evaluation.pyproject.toml /opt/llm-router/ml/pyproject.toml
RUN mkdir -p \
        /opt/llm-router/ml/src/llm_router_labeling/evaluation \
        /opt/llm-router/ml/src/llm_router_labeling/shared \
        /workspace/ml/config
COPY ml/src/llm_router_labeling/__init__.py /opt/llm-router/ml/src/llm_router_labeling/__init__.py
COPY ml/src/llm_router_labeling/evaluation/__init__.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/__init__.py
COPY ml/src/llm_router_labeling/evaluation/routerarena_evaluation_cli.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/routerarena_evaluation_cli.py
COPY ml/src/llm_router_labeling/evaluation/routerarena_routing_evaluator.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/routerarena_routing_evaluator.py
COPY ml/src/llm_router_labeling/evaluation/routerarena_visualization.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/routerarena_visualization.py
COPY ml/src/llm_router_labeling/evaluation/routerarena_preprocessing.py /opt/llm-router/ml/src/llm_router_labeling/evaluation/routerarena_preprocessing.py
COPY ml/src/llm_router_labeling/shared/__init__.py /opt/llm-router/ml/src/llm_router_labeling/shared/__init__.py
COPY ml/src/llm_router_labeling/shared/dataset_io.py /opt/llm-router/ml/src/llm_router_labeling/shared/dataset_io.py
COPY ml/src/llm_router_labeling/shared/text_normalization.py /opt/llm-router/ml/src/llm_router_labeling/shared/text_normalization.py
COPY ml/src/llm_router_labeling/shared/object_store.py /opt/llm-router/ml/src/llm_router_labeling/shared/object_store.py
COPY ml/src/llm_router_labeling/shared/observability.py /opt/llm-router/ml/src/llm_router_labeling/shared/observability.py
COPY ml/config/routerarena_evaluation.json /workspace/ml/config/routerarena_evaluation.json

RUN python -m pip install --no-deps /opt/llm-router/ml \
    && routerarena-evaluate --help >/dev/null \
    && groupadd --gid 10001 evaluator \
    && useradd --uid 10001 --gid evaluator --create-home --shell /usr/sbin/nologin evaluator \
    && mkdir -p /workspace/input /workspace/output /cache/huggingface \
    && chown -R evaluator:evaluator /workspace /cache

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["routerarena-evaluate"]
CMD ["--help"]
