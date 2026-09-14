---
tags: [moc, dataset]
---

# Datasets — map of content

Three public sources, **three incompatible label spaces**, deliberately never merged.

| source | what it labels | used for |
|---|---|---|
| [[Food Recognition 2022]] | cheese type in a meal scene | **everything shipped** |
| [[CHEESE-HIDB]] | industrial wheel, product × ripeness | a cautionary tale, [[Splits and data leakage]] |
| [[Cheese Images]] | commercial variety name | [[variety]], demo flavour only |

> [!danger] Why they are not merged
> No image carries two of the three kinds of label. And merging would be *actively
> harmful*: every HIDB image is a wheel on cardboard under one lamp, every Food
> Recognition image is a plate. The background alone separates them, so the model would
> learn the dataset instead of the cheese.

- [[Data provenance]] — where every training pixel comes from, and why it was transformed
- [[Class taxonomy]] — the 13 classes and how bins group them
- [[Reject classes]] — the `empty` and `not_cheese` additions

## Volume

| stage | count |
|---|---|
| normalised real images | 6486 |
| RGBA cutouts | 2400 (400 of them non-cheese) |
| belt renders | 12355 |
| distinct pieces in renders | 3191 |
