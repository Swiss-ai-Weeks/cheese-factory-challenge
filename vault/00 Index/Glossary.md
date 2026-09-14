---
tags: [moc, reference]
---

# Glossary

| term | meaning |
|---|---|
| **bin** | one of 5 sorting destinations the arm can target. A strict grouping of cheese types. See [[Class taxonomy]]. |
| **type** | one of 11 fine-grained cheese classes from [[Food Recognition 2022]], e.g. `emmental_cheese`. |
| **reject class** | `empty` or `not_cheese` — valid answers meaning "do not pick this up". See [[Reject classes]]. |
| **cutout** | an RGBA image of one piece with a real silhouette, from an annotation polygon. See [[Cutouts]]. |
| **render** | a cutout placed in a plate on the belt and rendered by Isaac Sim. See [[Isaac Sim rendering]]. |
| **group** | the split key: the physical object an image shows. Never the filename. See [[Splits and data leakage]]. |
| **marginalisation** | summing the probabilities of all types in a bin to decide the bin. See [[Probability marginalisation]]. |
| **macro-F1** | F1 averaged with equal weight per class. Chosen over accuracy because classes are imbalanced 18:1. See [[Evaluation protocol]]. |
| **head** | one trained model answering one question. Eight exist; see [[Models MOC]]. |
| **domain gap** | the difference between training images and what the robot will actually see. See [[What the numbers do not measure]]. |
| **status** | the field the robot must test first: `ok` / `empty` / `not_cheese` / `uncertain`. See [[Output contract]]. |
