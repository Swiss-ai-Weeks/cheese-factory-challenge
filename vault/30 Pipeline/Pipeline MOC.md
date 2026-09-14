---
tags: [moc, pipeline]
---

# Pipeline — map of content

```mermaid
flowchart LR
    N[[Normalisation]] --> C[[Cutouts]] --> R[[Isaac Sim rendering]] --> M[[Render manifest]]
    S[[Splits and data leakage]] -.governs.-> N
    S -.governs.-> M
    RJ[[Reject classes]] --> R
```

| note | what it covers |
|---|---|
| [[Normalisation]] | three sources → one format and one manifest |
| [[Cutouts]] | polygons → RGBA silhouettes, and the negatives |
| [[Isaac Sim rendering]] | the USD scene, domain randomisation, throughput |
| [[Reject classes]] | `empty` and `not_cheese`, and what they cost |
| [[Render manifest]] | renders → training manifest, JPEG conversion |
| [[Splits and data leakage]] | **the rule that governs all of it** |
