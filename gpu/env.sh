# Sourced by every gpu/ script on the pod. /workspace is the pod's persistent disk on Runpod; keeping the
# Hugging Face cache and token there means a restart doesn't re-download Gemma or lose the login.
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
# Runpod writes pod env vars (e.g. HF_TOKEN from a Runpod secret) here; non-interactive ssh shells don't load it.
[ -f /etc/rp_environment ] && source /etc/rp_environment
# Cap BLAS/OpenMP/torch threads at the container's CPU quota: nproc reports the host's cores, and 100+ threads
# spinning under a ~14-CPU cgroup quota makes CPU linear algebra crawl (seen: a 3.5 h analysis step).
if [ -r /sys/fs/cgroup/cpu.max ] && read q p < /sys/fs/cgroup/cpu.max && [ "$q" != max ]; then
  export OMP_NUM_THREADS=$(( q / p > 0 ? q / p : 1 ))
  export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
fi
