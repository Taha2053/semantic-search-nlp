"""Exporte les graphiques du notebook exécuté vers figures/ (pour la présentation)."""
import base64
import json
import re
import sys
import unicodedata
from pathlib import Path

NB = Path(sys.argv[1] if len(sys.argv) > 1 else "Semantic_Search_Embeddings.ipynb")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

def slug(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50]

heading, count = "figure", 0
for cell in json.loads(NB.read_text(encoding="utf-8"))["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        titles = [line.lstrip("#").strip() for line in src.splitlines() if line.startswith("#")]
        heading = titles[-1] if titles else heading
        continue
    images = [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
    for k, img in enumerate(images):
        count += 1
        suffix = f"_{k + 1}" if len(images) > 1 else ""
        path = OUT / f"{count:02d}_{slug(heading)}{suffix}.png"
        path.write_bytes(base64.b64decode(img))
        print(path)
print(f"{count} figures exportées dans {OUT}/")
