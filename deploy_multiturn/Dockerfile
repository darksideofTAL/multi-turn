# Standalone serving image for the multi-turn conversation monitor
# (talmonitor_multiturn): frozen 12B turn encoder + 15M causal aggregator
# (cscale ALL winner, see experiments/multiturn/RESULTS.md).
#
# Self-contained — builds from the public Triton base, no dependency on the
# local talmonitor-unified image. Build from the repo root:
#
#   docker build -f deploy_multiturn/Dockerfile -t talmonitor-multiturn .
#
# The 24 GB turn-encoder checkpoint is NOT baked in — bind-mount it at run
# time at the path stamped in config.pbtxt (parameters.model_dir):
#
#   docker run -d --name talmonitor-multiturn \
#     --gpus '"device=<GPU-or-MIG-UUID>"' --shm-size 1g \
#     -p 8700:8000 -p 8701:8001 -p 8702:8002 \
#     -v /path/to/google-gemma-3-12b-it-v4-nofreeze-n7500:/models/google-gemma-3-12b-it-v4-nofreeze-n7500:ro \
#     talmonitor-multiturn
FROM nvcr.io/nvidia/tritonserver:25.04-py3

ENV PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    FLA_DISABLE_BACKEND_DISPATCH=1 \
    TALMONITOR_SRC=/opt/talmonitor/src \
    TALMONITOR_MULTITURN=/opt/talmonitor/experiments/multiturn \
    PYTHONPATH=/opt/talmonitor/src:/opt/talmonitor/experiments/multiturn \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN pip install --no-cache-dir uv

RUN uv pip install --system \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0+cu128 torchvision torchaudio

RUN uv pip install --system \
        "transformers==5.8.1" \
        "peft>=0.19" \
        "accelerate>=1.12.0" \
        "pandas>=3.0.1" \
        "jinja2>=3.1.6" \
        "pyyaml>=6.0.3" \
        "pydantic>=2.12.5" \
        numpy \
        "tritonclient[all]"

COPY src/ /opt/talmonitor/src/
# prompts/ must stay a sibling of src/ (build_prompter resolves ../../../prompts)
COPY prompts/ /opt/talmonitor/prompts/
COPY experiments/multiturn/mtlib/ /opt/talmonitor/experiments/multiturn/mtlib/
# 15M aggregator checkpoint (44 MB) + Triton config ship inside the model repo.
COPY model_repository_multiturn/ /models_repo_mt/

EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models_repo_mt"]
