---
tags: [dataset, gotcha]
source: https://github.com/andrealoddo/CHEESE-HIDB
license: CC BY-SA 4.0
status: cautionary
---

# CHEESE-HIDB

Industrial cheese wheels from Lattebusche-Conad, curated at the University of Cagliari.
378 images at 6016×4016, ~6 GB. Three products × two ripeness classes.

> [!danger] It contains 9 physical wheels, not 378 cheeses
> Each wheel was photographed **42 times on a turntable** — blocks of consecutive DSC
> numbers, exactly 42 per block.
>
> ```
> extra_hard__nottarget   84 img = 2 blocks : 1771-1812(42) 1814-1855(42)
> extra_hard__target      42 img = 1 block  : 1900-1941(42)
> hard__nottarget         84 img = 2 blocks : 1943-1984(42) 1986-2027(42)
> hard__target            42 img = 1 block  : 1857-1898(42)
> semi_hard__nottarget    84 img = 2 blocks : 2029-2070(42) 2072-2113(42)
> semi_hard__target       42 img = 1 block  : 1727-1766(40) 1768-1769(2)
> ```

![[hidb-leak.png]]

Zoom on the surface and the same pink ink stamp appears, rotated. Splitting per image
put the same wheel in train *and* test, and the first run reported **100% macro-F1 by
epoch 13** — a number with zero predictive value. Full story: [[Splits and data leakage]].

## What survives an honest protocol

| task | wheels per class | verdict |
|---|---|---|
| product × ripeness (6 cls) | 1-2 | **not measurable**, everything goes to train |
| product only (3 cls) | 3-4 | measurable, see [[hidb_product]] |

## The better use for it

> [!tip] 42 views around a wheel is photogrammetry material
> Reconstructing a textured 3D wheel for Omniverse would be far more valuable than the
> 0.758 classifier — and it would feed [[Isaac Sim rendering]] with real geometry
> instead of flat decals. See [[Open questions]].

## Download note

`git clone` is very slow — one 6 GB pack. Parallel per-file download from
`raw.githubusercontent.com` with `xargs -P 24` takes under a minute.
