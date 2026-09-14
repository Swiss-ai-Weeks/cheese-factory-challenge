---
tags: [ops, reference]
---

# Reproduce everything

From an empty machine to the shipped model.

## 1 — Environment

See [[Environment]].

## 2 — Download the three datasets

```bash
# Food Recognition 2022 — the DatasetNinja link is dead, use dataset-tools
pip install dataset-tools
python -c "import dataset_tools as d; d.download(dataset='Food Recognition 2022', dst_dir='data/raw/')"

# cheese-images
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('NoeFlandre/cheese-images', repo_type='dataset', local_dir='data/raw/cheese_images')"

# CHEESE-HIDB — parallel per-file, git clone is far slower
curl -s 'https://api.github.com/repos/andrealoddo/CHEESE-HIDB/git/trees/main?recursive=1' \
  | python3 -c "import sys,json,urllib.parse; [print('https://raw.githubusercontent.com/andrealoddo/CHEESE-HIDB/main/'+urllib.parse.quote(e['path'])+'\t'+e['path']) for e in json.load(sys.stdin)['tree'] if e['type']=='blob' and e['path'].lower().endswith(('.jpg','.jpeg','.png'))]" \
  | xargs -P 24 -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r u p <<< "{}"; mkdir -p "$(dirname "$p")"; curl -sL -o "$p" "$u"'
```

## 3 — Normalise and cut out

```bash
.venv/bin/python src/normalize.py --workers 64
.venv/bin/python src/cutouts.py --negatives 400 --workers 48
```

## 4 — Render the belt

```bash
mkdir -p sim/out && chmod 777 sim/out          # the container is not root
for i in 0 1; do
  docker run -d --rm --gpus "device=$i" -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v $PWD:/workspace -v ~/.cache/ov/hub:/var/cache/hub \
    --name gen$i --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 \
    /workspace/sim/render_belt.py --all --views 5 --min-fill 0.35 --max-ar 3.0 \
      --elev-min 45 --elev-max 80 --shard $i/2 --seed $((200+i)) --skip-existing
done
# then the empty-belt class
docker run --rm --gpus "device=0" ... /workspace/sim/render_belt.py --empty 900 --elev-min 45 --elev-max 80
```

> [!important] Verify the counts before training
> ```bash
> ls sim/out/*.png | sed 's/.*__\(.*\)__.*/\1/' | sort | uniq -c | awk '$1 != 5'
> ```
> Should print nothing except the `empty` pieces. See [[Measurement pitfalls]].

## 5 — Manifest and training

```bash
.venv/bin/python src/render_manifest.py
.venv/bin/python src/train.py --task sim_type \
    --manifest data/processed/manifest_sim.csv \
    --model convnext_base.fb_in22k_ft_in1k_384 --img-size 384 \
    --epochs 20 --batch-size 96 --lr 1e-4 --workers 32 --out runs/sim_type13
```

## 6 — Export and publish

```bash
.venv/bin/python src/export.py runs/sim_type13/best.pt
./sync_jupyter.sh
```

Total: ~2 h wall clock, most of it rendering.
