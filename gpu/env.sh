# Sourced by every gpu/ script on the pod. /workspace is the pod's persistent disk on Runpod; keeping the
# Hugging Face cache and token there means a restart doesn't re-download Gemma or lose the login.
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
