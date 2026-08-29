# Compute disclosure

The reported final SFT continuation ran for 28 minutes 45 seconds on one NVIDIA
RTX 6000 Ada Generation GPU (46,068 MiB), with 12 CPU cores and 150 GiB RAM
allocated. The scheduler reported approximately 19.8 GiB peak process memory.
This excludes the three inherited LoRA stages and preliminary work.

Final-stage data materialization was CPU-only: 44 seconds with four CPU cores
and 64 GiB RAM allocated, with approximately 4.59 GiB peak process memory. This
excludes teacher-trajectory generation and earlier LoRA stages.

The completed HotpotQA post-trained and official-checkpoint families ran on
eight NVIDIA RTX A6000 GPUs (49,140 MiB each). Summed per-call telemetry records
36.69 active Gemma GPU-hours for the post-trained family and 27.07 active Gemma
GPU-hours for the official-checkpoint family. These totals exclude idle
model-residency time, canaries, and invalid preliminary runs.

GPT-5.6 Sol is externally hosted. The records expose calls, tokens, dollar cost,
and request latency, but not the provider's physical accelerators, memory,
energy, or GPU-hours. The release therefore does not claim a complete project
compute or energy accounting.
