---
tags: [todo]
priority: high
---

# Open questions

Ordered by value.

## 1. What does the model actually do on the real Isaac scene?

> [!danger] Unknown, and it is the largest remaining risk
> Every number in this vault comes from renders built out of photographs. **A hundred
> labelled frames** from the actual demo scene would measure the gap instead of assuming
> it; a few hundred would let a short fine-tune close it.
>
> Blocked on the demo scene existing. Nothing else here matters as much.

## 2. Will the demo use real 3D assets or textured primitives?

This decides whether [[Isaac Sim rendering]] is well matched. Textured primitives → the
flat-decal approximation is fine, train and inference share the same fakeness. Modelled
assets → the gap moves rather than disappears.

If real assets are needed, [[CHEESE-HIDB]]'s 42-view turntable sets are photogrammetry
material for a textured wheel.

## 3. Build the detector

The detect-then-crop stage of [[Robot pipeline]]. Background subtraction on a uniform
belt should be enough, and it is what would recover `bin_semi_hard`.

## 4. Strengthen `not_cheese`

Recall 0.828 — 17% of foreign objects pass as cheese, and that is the expensive
direction. Only 400 of 482 available non-cheese classes were used.
→ [[Reject performance]]

## 5. Would a 2D compositing baseline match the render pipeline?

~20 minutes to build. Would settle whether the hour of Isaac rendering bought anything
beyond cast shadows. → [[Decision log]]

## 6. Would ConvNeXt-Tiny match ConvNeXt-Base?

Every head peaks before epoch 20 on 1.5-8k images, suggesting wasted capacity. A quarter
of the weight would be easier to run in the perception loop. Untested.

## 7. Fix or drop `processed_cheese`

F1 0.000, 160 training images. Either gather more or fold it into `soft_cheese`.

## 8. Multi-GPU training

One GPU per run today. DDP would roughly halve the 20-minute cycle.

## 9. Recompute normalisation constants on renders

Minor, and arguably better left as is for comparability. → [[Normalisation]]
