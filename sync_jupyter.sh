#!/usr/bin/env bash
# Copie le projet dans /workspace du conteneur jupyter (bind sur
# /opt/nvidia/launchpad/jupyter-notebook), pour que la feuille hpe.ipynb soit
# executable depuis la webapp. Les deux repertoires sont sur des systemes de
# fichiers differents : pas de lien dur possible, c'est une vraie copie (~1 Go).
# A relancer apres chaque reentrainement.
set -euo pipefail
SRC=/home/nvidia/hpe/cheese
DST=/opt/nvidia/launchpad/jupyter-notebook/cheese

mkdir -p "$DST/src" "$DST/data/processed" "$DST/runs" "$DST/sim"
rsync -a --delete "$SRC/src/"            "$DST/src/"
rsync -a --delete "$SRC/data/processed/" "$DST/data/processed/"
rsync -a --delete "$SRC/runs/"           "$DST/runs/"
# Les PNG bruts de sim/out (3,6 Go) ne sont pas copies : la feuille lit les JPEG
# convertis dans data/processed/images/sim_belt. Les bruts restent sur l'hote.
rsync -a          "$SRC/sim/"*.py        "$DST/sim/"
# Les videos et leurs releves : les sections 8 et 9 de la feuille les lisent.
rsync -a          "$SRC/sim/cheese_sorting.mp4" "$SRC/sim/cheese_sorting.json" "$DST/sim/"
if [ -f "$SRC/sim/cheese_picking.mp4" ]; then
    rsync -a      "$SRC/sim/cheese_picking.mp4" "$SRC/sim/cheese_picking.json" "$DST/sim/"
fi
cp "$SRC/README.md" "$DST/README.md" 2>/dev/null || true
# Les feuilles sont servies a la racine de /workspace, pas dans cheese/
cp "$SRC/hpe.ipynb" "$SRC/hpe_fr.ipynb" "$DST/../"
# vault Obsidian (documentation), copie aussi a la racine de /workspace
rsync -a --delete "$SRC/vault/" "$DST/../vault/"

# La cellule qui prouve la fuite HIDB zoome dans les originaux 6016x4016, qui ne
# sont pas dans data/processed. On emporte seulement les 6 vues qu'elle affiche
# (~100 Mo) plutot que les 6 Go de data/raw.
mkdir -p "$DST/data/raw/cheese_hidb/Hard/Target"
ls "$SRC/data/raw/cheese_hidb/Hard/Target" | sort | awk 'NR % 7 == 1' | head -6 | while read -r f; do
    cp -n "$SRC/data/raw/cheese_hidb/Hard/Target/$f" "$DST/data/raw/cheese_hidb/Hard/Target/$f"
done

echo "copie -> $DST"
du -sh "$DST"
