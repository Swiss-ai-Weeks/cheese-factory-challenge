---
tags: [moc, model]
---

# Models — map of content

Eight heads were trained. **One is shipped.**

```mermaid
flowchart TD
    subgraph ship[SHIPPED]
        T13[[sim_type13]]
    end
    subgraph ablation[ABLATIONS - same splits]
        T11[[sim_type11]]
        SB[[sim_bin]]
    end
    subgraph baseline[REAL-PHOTO BASELINES]
        B[[bin]]
        FR[[fr_cheese_type]]
    end
    subgraph other[OTHER LABEL SPACES]
        HP[[hidb_product]]
        V[[variety]]
    end
    T11 -.what does reject cost?.-> T13
    SB -.does one model suffice?.-> T13
    B -.does the domain shift help?.-> T13
```

| head | classes | trained on | role | test top-1 |
|---|---|---|---|---|
| **[[sim_type13]]** | 13 | belt renders | **ship** | **0.728** |
| [[sim_type11]] | 11 | belt renders | ablation: cost of reject | 0.679 |
| [[sim_bin]] | 5 | belt renders | ablation: is one model enough | 0.763 |
| [[bin]] | 5 | real photos | baseline | 0.789 |
| [[fr_cheese_type]] | 12 | real photos | baseline | 0.591 |
| [[hidb_product]] | 3 | wheel photos | other label space | 0.758 |
| [[variety]] | 288 | web photos | other label space | 0.524 |
| `sim_type` | 11 | belt renders | **superseded**, old splits | — |

> [!danger] These top-1 values are NOT comparable to each other
> Different tasks, different class counts, different test sets. Only [[Model comparison]]
> puts matched pairs side by side.

![[all-models.png]]
