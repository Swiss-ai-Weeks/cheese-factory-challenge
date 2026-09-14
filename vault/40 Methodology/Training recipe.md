---
tags: [methodology]
script: src/train.py
---

# Training recipe

## Backbone

`convnext_base.fb_in22k_ft_in1k_384` from **timm** — ConvNeXt-Base pretrained on
ImageNet-22k, fine-tuned on ImageNet-1k at 384 px. Chosen for the 384 px native
resolution, which matches the input size without re-interpolating position embeddings
(a ViT would need that).

| setting | value | rationale |
|---|---|---|
| input | 384 px | native for this checkpoint |
| optimiser | AdamW | standard for ConvNeXt fine-tuning |
| learning rate | 1e-4 | 3e-4 was too high for a pretrained backbone this size |
| weight decay | 0.05 | timm default for ConvNeXt |
| schedule | linear warmup 3 ep → cosine | |
| batch | 96 | fits 94 GB with bf16 |
| precision | **bf16 autocast** | H100 native, no loss scaler needed |
| memory format | `channels_last` | ~20% faster for convnets |
| grad clipping | 1.0 | |
| label smoothing | 0.1 | calibrates confidence, matters for the threshold |
| epochs | 20 | see below |

## Augmentation

```python
v2.RandomResizedCrop(384, scale=(0.55, 1.0), ratio=(0.75, 1.333))
v2.RandomHorizontalFlip()
v2.RandAugment(num_ops=2, magnitude=7)
v2.Normalize(mean, std)              # train-split statistics
v2.RandomErasing(p=0.25, scale=(0.02, 0.15))
```

Evaluation: `Resize(438) → CenterCrop(384)` — the 1.14 ratio is the standard
resize-then-crop convention.

> [!note] No vertical flip, no rotation
> The belt camera has a consistent up direction. Adding those would teach invariances
> that do not exist in deployment.

## Why 20 epochs and not 60

> [!important] Every head peaks early
> Best epochs across eight runs: **3, 9, 14, 17, 18, 19, 19, 38** — out of 20, 30 or 60.
> ConvNeXt-Base memorises 1.5-8k images in under twenty passes; the rest changes nothing
> in the retained checkpoint.
>
> ![[learning-curves.png]]
>
> The first runs used 60 epochs. Across five models that is over an hour of GPU for zero
> gain — see [[Decision log]].

## Hardware and throughput

2× H100 NVL (94 GB). Training is **GPU-bound**, not dataloader-bound: 99% GPU with a CPU
load of 12 on 128 cores. Converting PNG→JPEG and raising workers to 32 changed nothing.

| dataset | images | s/epoch |
|---|---|---|
| `bin` | 1,473 | 11.5 |
| `sim_bin` | 6,685 | 44 |
| `sim_type13` | 8,655 | 55 |

> [!tip] Only one GPU per run
> Two runs on two devices works, but a single run across both was never wired up. It
> would roughly halve the time. See [[Open questions]].
