"""Restore one launch-ledger source tree into an empty directory."""
from pathlib import Path
import argparse
import hashlib
import json

base = Path(__file__).resolve().parent
sources = json.loads((base / "source_map.json").read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("implementation", choices=sorted(sources))
parser.add_argument("destination", type=Path)
args = parser.parse_args()
if args.destination.exists() and any(args.destination.iterdir()):
    parser.error("Choose a new or empty destination.")
for rel, entry in sources[args.implementation].items():
    data = (base.parent / entry["source"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Hash mismatch for {rel}")
    target = args.destination / rel
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_bytes(data)
print(f"Restored {len(sources[args.implementation])} verified files to {args.destination}")
