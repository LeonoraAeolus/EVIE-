# OpenMath-Nemotron on a Nebius GPU VM (the v3 math checker)

Tesla Lab v3 uses two NVIDIA models:

| Role | Model | Where |
|---|---|---|
| **Lead** (reads sources, extracts claims, plans, writes code, explains verdicts) | Nemotron 3 Super | Nebius **Token Factory** (pay per token) |
| **Math consultant** (derives the governing equation step by step) | **OpenMath-Nemotron** | **your own GPU VM** on Nebius AI Cloud (this folder) |
| **Calculator of record** | Python (SymPy / NumPy, sandboxed) | the app |

OpenMath derives, Python computes, and Tesla Lab compares the two. Each verdict shows
"math checker agreed" or "math checker DISAGREED". A disagreement is flagged for review; it never
changes the verdict on its own.

## 1. The model (checked on Hugging Face, 2026-10-13)

| Model id | Base | Weights (bf16) | Fits on |
|---|---|---|---|
| `nvidia/OpenMath-Nemotron-1.5B` | Qwen2.5-Math-1.5B | 3.1 GB | any GPU (weak at multi-step physics) |
| **`nvidia/OpenMath-Nemotron-7B`** (recommended) | Qwen2.5-Math-7B | 15.2 GB | **1× L40S 48 GB** |
| `nvidia/OpenMath-Nemotron-14B` | Qwen2.5-14B | 29.5 GB | 1× L40S 48 GB (less room for parallel requests) |
| `nvidia/OpenMath-Nemotron-32B` | Qwen2.5-32B | 65.5 GB | 1× H100 80 GB or 2 GPUs |

Weight sizes = parameter count from the Hugging Face API × 2 bytes. Not gated: no Hugging Face token needed.

- **Licence:** CC-BY-4.0 ("ready for commercial use"), plus Apache-2.0 from the Qwen base. Credit NVIDIA in your README/demo.
- **Prompt format (model card):** no system prompt, one user message:
  `Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.` + a blank line + the problem.
  Tesla Lab sends exactly that (`tesla_lab/openmath.py`).
- **Settings:** NVIDIA's own evaluation uses temperature **0.6** and allows up to 32,768 new tokens
  (NeMo-Skills eval docs). The card's example uses `max_new_tokens=4096`. Tesla Lab uses 0.6 and caps at
  12,000 (`OPENMATH_MAX_TOKENS`). top_p 0.95 is our choice; neither page states one.
- **It thinks first.** Tested on this box with the 1.5B model (CPU): the reply starts with `<think>` and
  reasons at length before the final answer. Too low a token cap = no `\boxed{}` answer, which Tesla Lab
  reports as "inconclusive: ran out of tokens".
- **Maths only.** The card says these models "have not been instruction tuned on general data and thus might
  not provide good answers outside of math domain". That is why Super writes the problem and OpenMath only solves it.
- Context length 131,072 tokens; we serve 16,384 to save memory.

Sources: https://huggingface.co/nvidia/OpenMath-Nemotron-7B ·
https://github.com/NVIDIA/NeMo-Skills/blob/main/docs/releases/openmathreasoning/evaluation.md ·
paper https://arxiv.org/abs/2504.16891

## 2. What it costs (Nebius prices read 2026-10-13, ESTIMATES)

From https://docs.nebius.com/compute/resources/pricing and https://nebius.com/prices (on-demand, USD).
Billing is per second while the VM is **running**. A stopped VM costs nothing for GPU/CPU/RAM,
but its disk is still billed.

| Setup | Price / hour | Hours for $100 (estimate) |
|---|---|---|
| **7B on L40S** (`gpu-l40s-a`, preset `1gpu-8vcpu-32gb`): $1.35 GPU + 8 × $0.012 vCPU + 32 × $0.0032 GiB RAM | **≈ $1.55** | **≈ 64 h** |
| 14B on L40S (`gpu-l40s-a`, `1gpu-16vcpu-64gb`) | ≈ $1.75 | ≈ 57 h |
| 7B on L40S **preemptible** (spot; can be stopped by Nebius at any time) | ≈ $0.75 | ≈ 130 h |
| H100 80 GB (`gpu-h100-sxm`, `1gpu-16vcpu-200gb`), from 2026-10-08 | $4.50 | ≈ 22 h |
| RTX PRO 6000 96 GB (`gpu-rtx6000`, us-central1 only) | $1.80 | ≈ 55 h |
| Boot disk, 100 GiB network SSD ($0.071 per GiB per month) | ≈ $0.24 / day, **even when stopped** | |

L40S and H100 are only in region `eu-north1`, so create your project there.
Not checked: whether your $100 credit covers every item (GPU, disk, public IP). Look at Billing in the console.
Public IP pricing isn't on the pricing page I read.

**What $100 really buys:** about 60 hours of the 7B model, which covers 18 days of building if you
only run it while you work (around 3 h a day) and **stop it every time**. Left running 24/7 it's gone in under 3 days.

**Speed (estimate, not measured):** one request on an L40S decodes roughly 40–55 tokens/s for 7B
(memory-bandwidth bound: 864 GB/s ÷ 15 GB of weights), about half that for 14B. OpenMath often writes
2,000–8,000 tokens of reasoning, so one check can take **1–3 minutes**. Tesla Lab sends all checks for a run
at once (vLLM batches them), so a whole run costs about as long as the slowest check. For demo recording day
an H100 is about 3–4× faster per request ($4.50/h, a few hours only).

## 3. Create the VM

On your laptop:

```bash
curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash     # Nebius CLI
nebius profile create                                              # log in, pick the eu-north1 project
ssh-keygen -t ed25519                                              # if you have no key yet
cd deploy/openmath
./create_vm.sh                                                     # 7B on L40S
# MODEL_SIZE=14B PRESET=1gpu-16vcpu-64gb ./create_vm.sh            # 14B
# PREEMPTIBLE=1 ./create_vm.sh                                     # spot: about half price, may be stopped
```

`create_vm.sh` follows Nebius' "Host an LLM" quickstart
(https://docs.nebius.com/compute/quickstart-host-model): it creates a 100 GiB boot disk from the
`ubuntu24.04-cuda12` image (NVIDIA drivers preinstalled), creates the VM with a public IP, and uses
cloud-init to install `serve.sh` as a systemd service. On first boot the service installs vLLM and downloads the
weights (allow 10–15 min), then serves the model:

```bash
vllm serve nvidia/OpenMath-Nemotron-7B \
  --served-model-name nvidia/OpenMath-Nemotron-7B \
  --host 127.0.0.1 --port 8000 \
  --api-key $OPENMATH_API_KEY \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --generation-config vllm
```

Web console instead of the CLI: Compute → Virtual machines → Create → platform "NVIDIA L40S PCIe with
Intel Ice Lake", preset 1 GPU / 8 vCPU / 32 GiB, image "Ubuntu 24.04 for NVIDIA GPUs (CUDA 12)",
disk 100 GiB, public IP on, your SSH key. Then SSH in and run `OPENMATH_API_KEY=... ./serve.sh`.

## 4. Connect Tesla Lab

```bash
ssh -N -L 8000:localhost:8000 tesla@<VM_IP>        # tunnel: the API never faces the internet
```

In `.env`:

```
OPENMATH_BASE_URL=http://localhost:8000/v1
OPENMATH_MODEL=nvidia/OpenMath-Nemotron-7B
OPENMATH_API_KEY=<printed by create_vm.sh>
```

```bash
python scripts/check_openmath.py      # lists served models, solves one problem, runs the cross-check
streamlit run app.py
```

Need a public endpoint (e.g. a hosted demo)? Recreate with `BIND_HOST=0.0.0.0` (the API key is
then required on every request), and restrict port 8000 to your demo host's IP.

## 5. Stop it when idle (this is what makes $100 last)

```bash
./stop_vm.sh            # = nebius compute instance stop --id <VM_ID>
./start_vm.sh           # next session (~1–3 min until vLLM has loaded the model again)
```

- **Don't** run `sudo shutdown` on the VM. Nebius treats it as a failure, restarts the VM and **keeps billing**
  (https://docs.nebius.com/compute/virtual-machines/stopped-with-linux-commands). Stop it from the console or the CLI.
- A dynamic public IP is released after the VM has been stopped for more than an hour, so the IP can change.
  `start_vm.sh` prints the new one.
- After the hackathon, **delete** the VM and its disk in the console, or the disk keeps costing ~$7/month.

**Optional auto-shutdown:** `IDLE_MIN=20 MAX_HOURS=4 ./idle_stop.sh` runs on your laptop next to the
tunnel. It reads vLLM's request counter from `/metrics` and stops the VM through the Nebius API after
20 idle minutes, or after 4 hours no matter what. A sturdier version runs the same loop on a tiny CPU VM
or as a cron job with a Nebius service account, so it works even when your laptop is closed.
The simplest safety net is a phone alarm.

## 6. What was and wasn't verified

Verified 2026-10-13: model ids, sizes, licence, base models and chat template (Hugging Face API and files);
the prompt format and temperature (model card + NeMo-Skills docs); that the model emits `<think>` first
(1.5B run on CPU on this box); Nebius CLI commands, platform/preset names, image family, region, stop/start
behaviour and prices (docs pages above); shell syntax of the scripts (`bash -n`).

Not verified (needs your Nebius account): the scripts have **not** been run against Nebius; vLLM has
not been run on a GPU here; speeds and hours are estimates; whether the credit covers disk/IP.
