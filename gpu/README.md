# Running the GPU queue on a Runpod pod

What it runs (`experiments/gpu_queue.py`, ~5–7 A100-hours, skips finished work on restart):

| step | what | runs |
|---|---|---|
| 1 | Cache 10M tokens of Gemma-2-2B activations at layers 3, 13, 23 | 3 passes, ~138 GB fp16 |
| 2 (A) | Phase 2 at Gemma scale: real / Gaussian / scrambled, widths 16k and 65k, k=40, 3 seeds, 3 epochs | 54 |
| 3 (B) | Small-k sweep at layer 13: real and Gaussian, k = 1, 2, 4, 8, 16, 40, 128, 2 seeds | 28 |
| 4 (C) | De-massive at layer 13: drop tokens with the top 0.1% / 1% of \|x₁₁₇₀\|, 3 seeds | 6 |
| 5 | Summarize every run into `results/gpu/runs.json` + `sigma.json` | — |

Pod: one A100 80 GB (or H100), Runpod PyTorch template, **≥ 250 GB persistent volume at `/workspace`**,
SSH over exposed TCP port 22 (the `ssh.runpod.io` proxy cannot do rsync).

## Steps

On this Mac:

```bash
bash gpu/push.sh <pod-ip> <ssh-port>
```

On the pod (`ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519`):

```bash
cd /workspace/SAE
bash gpu/setup.sh        # venv on the image's CUDA PyTorch, deps, GPU check
bash gpu/hf_login.sh     # you: paste your Hugging Face token (Gemma is gated)
bash gpu/run_queue.sh    # starts the queue in tmux session "sae"
tail -f results/gpu/queue.log
```

Back on this Mac, when `QUEUE DONE` appears in the log:

```bash
bash gpu/pull.sh <pod-ip> <ssh-port>
```

Then **stop or terminate the pod** in the Runpod console (or ask Claude to) — it bills while running.

## Notes

- Everything is restartable: re-running `gpu/run_queue.sh` skips finished activation caches and SAE runs.
- On CUDA, `train_sae.py` shuffles and scrambles on the GPU; on the Mac (MPS/CPU) its behaviour is unchanged.
- `results/gpu/runs.json` carries each run's decoder eigenvalues, slopes on the four pre-registered windows,
  γ against that layer's Σ, FVU, dead latents and eval L0, so W_dec files never need copying back.
