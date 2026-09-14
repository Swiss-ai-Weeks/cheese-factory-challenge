---
tags: [dataset]
source: https://datasetninja.com/food-recognition
license: CC0 1.0
modality: polygon segmentation
status: primary
---

# Food Recognition 2022

> [!success] The source of everything shipped
> AIcrowd challenge dataset, mirrored on DatasetNinja in Supervisely format.
> ~44k meal-scene images, 498 food classes, annotated with **polygons**.

## Why polygons matter

This is not a classification dataset — it is a *segmentation* dataset. Each instance has
a human-drawn outline. That single fact makes [[Cutouts]] possible without any
segmentation model, and it is what lets a piece be placed convincingly in a 3D plate.

It also explains a known defect: when bread stays under the cheese, the annotator drew
it that way. Not fixable downstream.

## What was taken

| use | classes | instances |
|---|---|---|
| cheese types | 11 | 2000 cutouts |
| non-cheese negatives | 400 distinct | 400 cutouts |
| ignored | the rest of 498 | — |

See [[Class taxonomy]] for the exact mapping and [[Reject classes]] for the negatives.

## Two operational gotchas

> [!bug] The official `test` split ships with no annotations
> The challenge withheld the labels. It yields zero usable instances, so splits were
> rebuilt locally. The original assignment is kept in `extra.official_split`.

> [!bug] The DatasetNinja download link is dead
> `DOWNLOAD.md` points at a signed S3 URL that returns 404. The `dataset-tools` package
> regenerates a working one:
> ```python
> import dataset_tools as dtools
> dtools.download(dataset='Food Recognition 2022', dst_dir='data/raw/')
> ```

## Annotation format

Supervisely JSON, one file per image:

```json
{"size": {"height": 426, "width": 426},
 "objects": [{"classTitle": "hard-cheese",
              "geometryType": "polygon",
              "points": {"exterior": [[272, 418], [237, 412], ...],
                         "interior": []}}]}
```

`object_index` — the position in that `objects` list — is stored in every manifest row,
which is what makes the chain reversible. See [[Data provenance]].
