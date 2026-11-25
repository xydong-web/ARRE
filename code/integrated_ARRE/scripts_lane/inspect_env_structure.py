"""Lane A helper: summarise the (protocol, block-type) structure of an env dir.

Reads only `workflow.yaml` and the line counts of `blocks/*.smi`; never loads the
12.7 GB `bb_feature.pt`, so it is cheap enough for the login node.

usage:  python scripts_lane/inspect_env_structure.py <env_dir>
"""

import sys
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf


def main(env_dir: str) -> None:
    env_path = Path(env_dir)
    cfg = OmegaConf.load(env_path / "workflow.yaml")

    n_block: dict[str, int] = {}
    for f in sorted((env_path / "blocks").iterdir()):
        with f.open("rb") as fh:
            n_block[f.stem] = sum(1 for _ in fh)
    total_blocks = sum(n_block.values())
    print(f"env_dir           = {env_path}")
    print(f"#block types      = {len(n_block)}")
    print(f"#blocks (sum N_t) = {total_blocks:,}")
    sizes = sorted(n_block.values())
    print(f"block type sizes  : min={sizes[0]:,} med={sizes[len(sizes)//2]:,} max={sizes[-1]:,}")

    first, birxn = [], []
    for action_type, group in cfg.items():
        for name, c in group.items():
            bts = list(c.block_types)
            entry = (name, bts, sum(n_block[b] for b in bts), c.get("state_pattern", None))
            (first if str(action_type) == "FirstBlock" else birxn).append(entry)

    print(f"\n#FirstBlock protocols = {len(first)}  (entries {sum(e[2] for e in first):,})")
    print(f"#BiRxn protocols      = {len(birxn)}  (entries {sum(e[2] for e in birxn):,})")
    n_pairs = sum(len(e[1]) for e in first + birxn)
    print(f"#(protocol, block type) pairs = {n_pairs}")

    pat_entries: Counter = Counter()
    pat_protocols: Counter = Counter()
    pat_pairs: Counter = Counter()
    for name, bts, n, pat in birxn:
        pat_entries[pat] += n
        pat_protocols[pat] += 1
        pat_pairs[pat] += len(bts)
    print(f"\n#state patterns = {len(pat_entries)}")
    vals = sorted(pat_entries.values())
    print(f"legal entries per BiRxn state: min={vals[0]:,} med={vals[len(vals)//2]:,} max={vals[-1]:,}")
    print(f"legal entries for the initial state (all FirstBlock): {sum(e[2] for e in first):,}")
    pp = sorted(pat_pairs.values())
    print(f"(protocol, type) pairs per BiRxn state: min={pp[0]} med={pp[len(pp)//2]} max={pp[-1]}")
    print("\ntop-5 state patterns by #legal entries:")
    for pat, n in pat_entries.most_common(5):
        print(f"  pattern {pat}: {n:,} entries, {pat_protocols[pat]} protocols, {pat_pairs[pat]} (proto,type) pairs")

    refs: Counter = Counter()
    for _, bts, _, _ in first + birxn:
        for b in bts:
            refs[b] += 1
    rv = sorted(refs.values())
    print(f"\n#protocols referencing a block type: min={rv[0]} med={rv[len(rv)//2]} max={rv[-1]}")
    print(f"total (protocol, block) actions = {sum(n_block[b] * c for b, c in refs.items()):,}")


if __name__ == "__main__":
    main(sys.argv[1])
