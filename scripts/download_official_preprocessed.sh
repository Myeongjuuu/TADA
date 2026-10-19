#!/usr/bin/env bash
# Download only the official ds005574 derivatives and stimuli, not raw subject EDF data.
set -euo pipefail

destination="${1:-data/source/ds005574}"
tree_url="https://api.github.com/repos/OpenNeuroDatasets/ds005574/git/trees/main?recursive=1"
s3_base="https://s3.amazonaws.com/openneuro.org/ds005574"
manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

mkdir -p "$destination"

python3 - "$tree_url" "$s3_base" "$destination" > "$manifest" <<'PY'
import json
import sys
from pathlib import PurePosixPath
from urllib.request import urlopen

tree_url, s3_base, destination = sys.argv[1:]
with urlopen(tree_url) as response:
    tree = json.load(response)["tree"]

prefixes = ("derivatives/ecogprep/", "derivatives/ecogqc/", "stimuli/")
for item in tree:
    path = item["path"]
    if item["type"] == "blob" and path.startswith(prefixes):
        print(f"{s3_base}/{path}\t{PurePosixPath(destination) / path}")
PY

while IFS=$'\t' read -r url target; do
    mkdir -p "$(dirname "$target")"
    # -c resumes an interrupted real download. This target is deliberately
    # separate from the DataLad metadata checkout, whose files are annex pointers.
    wget -c --show-progress -O "$target" "$url"
done < "$manifest"

echo "Complete: $destination"
