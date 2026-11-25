#!/usr/bin/env python3
"""The CGFlow reward as a plain function of a molecule (round 4: same oracle for SynGA/SynGBO, robustness re-scoring, retrospective enrichment).

In CGFlow with redocking=True (the paper setting and ours) the reward of a molecule is
    min(0, UniDock 'balance' score of an srETKDGv3 conformer (seed 1) docked into a 20 A box centred on the mean of the reference ligand's atoms)
(cgflow.tasks.semla_vina._run_redocking_caching -> cgflow.utils.unidock.docking); the SemlaFlow pose only conditions the generator, it is NOT the scored pose.
This module reproduces exactly that function (including the systems-only parallel-ETKDG patch of run/run_opt.py) so that any set of SMILES can be scored with the oracle of record.
usage as a library:  o = CGFlowOracle(pocket_dir); scores = o(smiles_list)      (failed docking -> 0.0; scores are <= 0)
usage as a script:   cgflow_oracle.py --pocket_dir data/LIT-PCBA/ADRB2 --smi in.smi --out out.tsv   (one SMILES per line, optional 2nd column = id)"""
import argparse
import os
import sys
from pathlib import Path

C = Path("/path/to/ARRE")


def assert_gpu_ok():
    """Fail loudly on a node whose GPU cannot be used (2026-10-22: eg046 GPU 0000:31:00.0 'Unable to determine the device handle'; Uni-Dock then returns the failure
    value 0 for every molecule and the job 'completes' with all-zero scores - 15 enrichment, 3 SynGA and 1 robustness units were silently corrupted that way)."""
    import subprocess

    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0 or "Unable to determine" in out or "NVIDIA" not in out:
        raise RuntimeError(f"no usable GPU on {os.uname().nodename}: {out[:300]!r}")


class CGFlowOracle:
    def __init__(self, pocket_dir, ncpu=None, pool=True):
        assert_gpu_ok()
        self._zero_streak = 0
        sys.path.insert(0, str(C / "run"))
        import run_opt as ro  # noqa: E402

        alloc = ro.limit_cpu_count()  # os.cpu_count() -> Slurm allocation (unidock_tools forks os.cpu_count() workers per call)
        n = alloc if ncpu is None else min(ncpu, alloc)
        if pool:
            ro.start_etkdg_pool(max(n - 1, 1))
            ro.patch_fast_etkdg()
        import cgflow.utils.unidock as ud
        from cgflow.utils.extract_pocket import get_mol_center

        self.ud = ud
        d = Path(pocket_dir)
        self.protein = d / "protein.pdb"
        x, y, z = get_mol_center(str(d / "ligand.mol2"))
        self.center = (round(x, 3), round(y, 3), round(z, 3))

    def dock(self, smiles):
        """-> [(docked RDKit mol or None, score)] for a list of SMILES"""
        from rdkit import Chem

        mols = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            mols.append(m if m is not None else Chem.Mol())
        res = self.ud.docking(mols, self.protein, self.center, search_mode="balance")
        # a healthy batch of >= 8 molecules always contains negative scores; an all-zero batch means that Uni-Dock itself failed (broken GPU / driver)
        if len(res) >= 8 and all(float(v) >= 0.0 for _, v in res):
            self._zero_streak += 1
            if self._zero_streak >= 2:
                raise RuntimeError(f"Uni-Dock returned only zero scores for {self._zero_streak} consecutive batches on {os.uname().nodename}: broken GPU?")
        else:
            self._zero_streak = 0
        return res

    def __call__(self, smiles, pmap=None):
        return [min(float(v), 0.0) for _, v in self.dock(smiles)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_dir", required=True)
    ap.add_argument("--smi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save_poses", default="")
    a = ap.parse_args()
    rows = [ln.split() for ln in Path(a.smi).read_text().splitlines() if ln.strip()]
    o = CGFlowOracle(a.pocket_dir)
    from rdkit import Chem

    w = Chem.SDWriter(a.save_poses) if a.save_poses else None
    with open(a.out, "w") as f:
        f.write("id\tsmiles\tvina\n")
        for i in range(0, len(rows), a.batch):
            chunk = rows[i : i + a.batch]
            res = o.dock([r[0] for r in chunk])
            for r, (mol, v) in zip(chunk, res):
                f.write(f"{r[1] if len(r) > 1 else ''}\t{r[0]}\t{min(float(v), 0.0):.4f}\n")
                if w is not None and mol is not None:
                    mol.SetProp("_Name", r[1] if len(r) > 1 else r[0])
                    mol.SetProp("smiles_in", r[0])
                    w.write(mol)
            f.flush()
            print(f"{min(i + a.batch, len(rows))}/{len(rows)}", flush=True)
    if w is not None:
        w.close()


if __name__ == "__main__":
    main()
