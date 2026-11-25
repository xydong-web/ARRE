"""Lane D - D2: cheap, deterministic, pose-aware features for the docking proxy.

The feature vector of a candidate is

    phi(x) = [ dense (94) | log1p(Morgan count fp, r=2, 1024 bits) ]

The dense block is

  * 64 protein-ligand contact counts  log1p(n[shell, lig_class, poc_class])
    with shells  (<3.0, 3.0-3.5, 3.5-4.5, 4.5-6.0) A,
    ligand classes (C, N, O, other) and pocket classes (C, N, O, S);
  * 4 per-shell contact densities (contacts per ligand heavy atom);
  * 12 geometric scalars (clashes, burial, min/mean contact distance, centroid
    offset, radius of gyration, polar / apolar contact counts);
  * 14 2D descriptors (MW, HBD, HBA, TPSA, cLogP, rotatable bonds, rings,
    aromatic rings, heavy atoms, QED, FractionCSP3, formal charge,
    heteroatoms, saturated rings).

Everything is a function of *distances* between ligand heavy atoms and pocket
heavy atoms, of ligand-internal geometry, and of the 2D graph, except the
centroid offset which is measured against ``pocket_center``.  Passing the
pocket atoms and the pocket centre explicitly therefore makes the whole
featuriser equivariant-by-construction: applying one rigid-body motion to
ligand + pocket + centre leaves phi(x) unchanged (tests/lane_d/test_features.py).

No RNG, no learned component: phi is deterministic given (mol, pocket).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, QED, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

# --------------------------------------------------------------------------------------
# layout constants
# --------------------------------------------------------------------------------------
SHELL_EDGES: tuple[float, ...] = (3.0, 3.5, 4.5, 6.0)
N_SHELL = len(SHELL_EDGES)
CONTACT_CUTOFF = SHELL_EDGES[-1]

LIG_CLASSES: tuple[str, ...] = ("C", "N", "O", "X")  # X = S, P, halogens, ...
POC_CLASSES: tuple[str, ...] = ("C", "N", "O", "S")
N_LIG_CLS, N_POC_CLS = len(LIG_CLASSES), len(POC_CLASSES)

CLASH_DIST = 2.8
POLAR_DIST = 3.5
APOLAR_DIST = 4.5
BURIED_DIST = 4.5

N_CONTACT = N_SHELL * N_LIG_CLS * N_POC_CLS  # 64
N_DENSITY = N_SHELL  # 4
N_GEOM = 12
N_DESC = 14
N_DENSE = N_CONTACT + N_DENSITY + N_GEOM + N_DESC  # 94

FP_RADIUS = 2
FP_BITS = 1024
FEATURE_DIM = N_DENSE + FP_BITS  # 1118

_LIG_CLS_IDX = {"C": 0, "N": 1, "O": 2}
_POC_CLS_IDX = {"C": 0, "N": 1, "O": 2, "S": 3}

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


# --------------------------------------------------------------------------------------
# pocket side
# --------------------------------------------------------------------------------------
class PocketAtoms:
    """Heavy atoms of the pocket PDB written by ``extract_pocket_crossdocked_v1``.

    Only the four element classes that carry the interaction signal are kept
    (C, N, O, S); anything else in the pocket file (rare hetero atoms) is
    dropped.  ``pos`` is (P, 3) float32, ``cls`` is (P,) int8 with the index
    into :data:`POC_CLASSES`.
    """

    __slots__ = ("pos", "cls", "center", "path")

    def __init__(self, pos: np.ndarray, cls: np.ndarray, center: np.ndarray, path: str = ""):
        assert pos.ndim == 2 and pos.shape[1] == 3
        assert cls.shape == (pos.shape[0],)
        self.pos = np.ascontiguousarray(pos, dtype=np.float32)
        self.cls = np.ascontiguousarray(cls, dtype=np.int8)
        self.center = np.asarray(center, dtype=np.float32).reshape(3)
        self.path = path

    def __len__(self) -> int:
        return int(self.pos.shape[0])

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "PocketAtoms":
        """Rigid-body image of the pocket (used by the invariance test)."""
        R = np.asarray(R, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32).reshape(3)
        return PocketAtoms(self.pos @ R.T + t, self.cls, self.center @ R.T + t, self.path)


def load_pocket_atoms(pocket_pdb: str | Path, center: tuple[float, float, float]) -> PocketAtoms:
    """Parse the pocket PDB with BioPython (already a dependency of the repo)."""
    from Bio.PDB.PDBParser import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", str(pocket_pdb))
    pos: list[tuple[float, float, float]] = []
    cls: list[int] = []
    for atom in structure.get_atoms():
        element = (atom.element or "").strip().upper()
        if not element:  # fall back to the PDB atom name
            element = atom.get_id().strip()[:1].upper()
        if element == "H" or element == "D":
            continue
        idx = _POC_CLS_IDX.get(element)
        if idx is None:
            continue
        c = atom.get_coord()
        pos.append((float(c[0]), float(c[1]), float(c[2])))
        cls.append(idx)
    if not pos:
        raise ValueError(f"no C/N/O/S heavy atoms parsed from pocket file {pocket_pdb}")
    return PocketAtoms(np.asarray(pos, dtype=np.float32), np.asarray(cls, dtype=np.int8),
                       np.asarray(center, dtype=np.float32), str(pocket_pdb))


# --------------------------------------------------------------------------------------
# ligand side
# --------------------------------------------------------------------------------------
def _ligand_arrays(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    """(pos, cls) of the ligand heavy atoms taken from the predicted-pose conformer."""
    conf = mol.GetConformer()
    all_pos = np.asarray(conf.GetPositions(), dtype=np.float32)
    keep: list[int] = []
    cls: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        keep.append(atom.GetIdx())
        cls.append(_LIG_CLS_IDX.get(atom.GetSymbol(), 3))
    if not keep:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int8)
    return all_pos[keep], np.asarray(cls, dtype=np.int8)


def morgan_counts(mol: Chem.Mol) -> np.ndarray:
    fp = _MORGAN_GEN.GetCountFingerprintAsNumPy(mol)
    return np.log1p(np.asarray(fp, dtype=np.float32))


def _descriptors_2d(mol: Chem.Mol) -> np.ndarray:
    def _safe(fn, default: float = 0.0) -> float:
        try:
            return float(fn(mol))
        except Exception:
            return default

    return np.asarray(
        [
            _safe(rdMolDescriptors.CalcExactMolWt) / 100.0,
            _safe(rdMolDescriptors.CalcNumHBD),
            _safe(rdMolDescriptors.CalcNumHBA),
            _safe(rdMolDescriptors.CalcTPSA) / 100.0,
            _safe(Crippen.MolLogP),
            _safe(rdMolDescriptors.CalcNumRotatableBonds),
            _safe(rdMolDescriptors.CalcNumRings),
            _safe(rdMolDescriptors.CalcNumAromaticRings),
            float(mol.GetNumHeavyAtoms()) / 50.0,
            _safe(QED.qed),
            _safe(rdMolDescriptors.CalcFractionCSP3),
            float(Chem.GetFormalCharge(mol)),
            _safe(rdMolDescriptors.CalcNumHeteroatoms),
            _safe(rdMolDescriptors.CalcNumSaturatedRings),
        ],
        dtype=np.float32,
    )


def _pose_block(lig_pos: np.ndarray, lig_cls: np.ndarray, pocket: PocketAtoms) -> np.ndarray:
    """64 contact counts + 4 densities + 12 geometric scalars."""
    out = np.zeros(N_CONTACT + N_DENSITY + N_GEOM, dtype=np.float32)
    n_lig = int(lig_pos.shape[0])
    if n_lig == 0 or len(pocket) == 0:
        return out

    # Restrict the pocket to a box around the ligand first: that turns the
    # 40 x 2000 distance matrix into roughly 40 x 400 and keeps the whole
    # featuriser at a few tenths of a millisecond per molecule.
    lo = lig_pos.min(axis=0) - CONTACT_CUTOFF
    hi = lig_pos.max(axis=0) + CONTACT_CUTOFF
    box = np.all((pocket.pos >= lo) & (pocket.pos <= hi), axis=1)
    poc_pos = pocket.pos[box]
    poc_cls = pocket.cls[box]

    lig_centroid = lig_pos.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((lig_pos - lig_centroid) ** 2, axis=1))))
    max_r = float(np.max(np.linalg.norm(lig_pos - lig_centroid, axis=1)))
    centroid_off = float(np.linalg.norm(lig_centroid - pocket.center))

    if poc_pos.shape[0] == 0:
        out[N_CONTACT + N_DENSITY + 4] = 10.0  # min_dist (clipped)
        out[N_CONTACT + N_DENSITY + 5] = 10.0  # mean of per-atom min distance
        out[N_CONTACT + N_DENSITY + 6] = centroid_off
        out[N_CONTACT + N_DENSITY + 7] = rg
        out[N_CONTACT + N_DENSITY + 8] = max_r
        return out

    d = np.linalg.norm(lig_pos[:, None, :] - poc_pos[None, :, :], axis=-1)  # (L, P)

    shell = np.digitize(d, SHELL_EDGES)  # 0..N_SHELL (N_SHELL == beyond cutoff)
    within = shell < N_SHELL
    if within.any():
        li, pj = np.nonzero(within)
        flat = (shell[li, pj].astype(np.int64) * N_LIG_CLS + lig_cls[li].astype(np.int64)) * N_POC_CLS
        flat = flat + poc_cls[pj].astype(np.int64)
        counts = np.bincount(flat, minlength=N_CONTACT).astype(np.float32)
    else:
        counts = np.zeros(N_CONTACT, dtype=np.float32)
    out[:N_CONTACT] = np.log1p(counts)
    out[N_CONTACT:N_CONTACT + N_DENSITY] = counts.reshape(N_SHELL, -1).sum(axis=1) / n_lig

    per_atom_min = d.min(axis=1)
    g = N_CONTACT + N_DENSITY
    n_clash = float(np.count_nonzero(d < CLASH_DIST))
    out[g + 0] = np.log1p(n_clash)
    out[g + 1] = float(np.count_nonzero(per_atom_min < CLASH_DIST)) / n_lig
    out[g + 2] = float(np.count_nonzero(per_atom_min < BURIED_DIST)) / n_lig  # buried fraction
    # distinct pocket atoms in contact, per ligand heavy atom (rotation invariant;
    # a box-normalised density would not be, the pre-filter box is axis aligned)
    out[g + 3] = float(np.count_nonzero(d.min(axis=0) < CONTACT_CUTOFF)) / n_lig
    out[g + 4] = float(min(per_atom_min.min(), 10.0))
    out[g + 5] = float(np.mean(np.minimum(per_atom_min, 10.0)))
    out[g + 6] = centroid_off
    out[g + 7] = rg
    out[g + 8] = max_r

    lig_polar = (lig_cls == 1) | (lig_cls == 2)
    poc_polar = (poc_cls == 1) | (poc_cls == 2)
    if lig_polar.any() and poc_polar.any():
        dp = d[np.ix_(lig_polar, poc_polar)]
        n_polar = float(np.count_nonzero(dp < POLAR_DIST))
    else:
        n_polar = 0.0
    out[g + 9] = np.log1p(n_polar)
    out[g + 10] = n_polar / n_lig

    lig_c = lig_cls == 0
    poc_c = poc_cls == 0
    if lig_c.any() and poc_c.any():
        dc = d[np.ix_(lig_c, poc_c)]
        n_apolar = float(np.count_nonzero(dc < APOLAR_DIST))
    else:
        n_apolar = 0.0
    out[g + 11] = np.log1p(n_apolar)
    return out


def featurize(mol: Chem.Mol, pocket: PocketAtoms) -> np.ndarray:
    """Full feature vector, shape (FEATURE_DIM,), float32.  Never raises."""
    x = np.zeros(FEATURE_DIM, dtype=np.float32)
    # the pose block degrades to zeros when there is no usable conformer, but the
    # 2D block is always filled: a candidate is never silently featureless
    try:
        lig_pos, lig_cls = _ligand_arrays(mol)
        x[: N_CONTACT + N_DENSITY + N_GEOM] = _pose_block(lig_pos, lig_cls, pocket)
    except Exception:
        pass
    try:
        x[N_CONTACT + N_DENSITY + N_GEOM: N_DENSE] = _descriptors_2d(mol)
    except Exception:
        pass
    try:
        x[N_DENSE:] = morgan_counts(mol)
    except Exception:
        pass
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def featurize_batch(mols: list[Chem.Mol | None], pocket: PocketAtoms) -> np.ndarray:
    out = np.zeros((len(mols), FEATURE_DIM), dtype=np.float32)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        out[i] = featurize(mol, pocket)
    return out


# index of the exact QED descriptor inside the dense block (used by the gate so
# that the acquisition can mix the predicted Vina reward with the *exact* QED)
IDX_QED = N_CONTACT + N_DENSITY + N_GEOM + 9
