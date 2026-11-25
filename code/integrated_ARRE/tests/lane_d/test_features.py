"""T1 - the pose-aware features are deterministic and rigid-motion invariant."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cgflow.lanes.lane_d_features import (
    FEATURE_DIM,
    IDX_QED,
    N_CONTACT,
    N_DENSE,
    PocketAtoms,
    featurize,
    featurize_batch,
)


def _mol_with_pose(smiles: str = "CC(=O)Nc1ccc(O)cc1", seed: int = 7) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    return Chem.RemoveHs(mol)


def _fake_pocket(mol: Chem.Mol, rng: np.random.Generator, n: int = 400) -> PocketAtoms:
    """Shell of C/N/O/S atoms scattered around the ligand."""
    lig = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    centre = lig.mean(axis=0)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = rng.uniform(3.0, 12.0, size=(n, 1))
    pos = (centre + direction * radius).astype(np.float32)
    cls = rng.integers(0, 4, size=n).astype(np.int8)
    return PocketAtoms(pos, cls, centre)


def _rigid(rng: np.random.Generator):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q.astype(np.float32), rng.normal(scale=5.0, size=3).astype(np.float32)


def _move_mol(mol: Chem.Mol, R: np.ndarray, t: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    pos = np.asarray(conf.GetPositions(), dtype=np.float64)
    conf.SetPositions(np.ascontiguousarray(pos @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64)))
    return out


def test_shape_and_finiteness():
    rng = np.random.default_rng(0)
    mol = _mol_with_pose()
    pocket = _fake_pocket(mol, rng)
    x = featurize(mol, pocket)
    assert x.shape == (FEATURE_DIM,)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert 0.0 <= x[IDX_QED] <= 1.0
    assert x[:N_CONTACT].sum() > 0.0, "the fake pocket must produce some contacts"


def test_deterministic():
    rng = np.random.default_rng(1)
    mol = _mol_with_pose()
    pocket = _fake_pocket(mol, rng)
    a = featurize(mol, pocket)
    b = featurize(mol, pocket)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_rigid_motion_invariance(seed):
    """One rigid motion applied to ligand + pocket + pocket centre changes nothing."""
    rng = np.random.default_rng(100 + seed)
    mol = _mol_with_pose(seed=seed + 1)
    pocket = _fake_pocket(mol, rng)
    R, t = _rigid(rng)

    before = featurize(mol, pocket)
    after = featurize(_move_mol(mol, R, t), pocket.transformed(R, t))
    assert np.abs(before - after).max() < 1e-3


def test_translation_of_ligand_alone_changes_the_features():
    """Sanity: the features are *not* trivially constant - moving only the ligand matters."""
    rng = np.random.default_rng(5)
    mol = _mol_with_pose()
    pocket = _fake_pocket(mol, rng)
    moved = _move_mol(mol, np.eye(3, dtype=np.float32), np.array([8.0, 0.0, 0.0], dtype=np.float32))
    assert np.abs(featurize(mol, pocket) - featurize(moved, pocket)).max() > 1e-2


def test_2d_block_is_pose_independent():
    """The descriptor + fingerprint block must not depend on the conformer at all."""
    rng = np.random.default_rng(6)
    mol = _mol_with_pose()
    pocket = _fake_pocket(mol, rng)
    moved = _move_mol(mol, np.eye(3, dtype=np.float32), np.array([30.0, -10.0, 4.0], dtype=np.float32))
    a, b = featurize(mol, pocket), featurize(moved, pocket)
    start = N_DENSE - 14
    assert np.abs(a[start:] - b[start:]).max() < 1e-5


def test_batch_handles_none_and_broken_molecules():
    rng = np.random.default_rng(7)
    mol = _mol_with_pose()
    pocket = _fake_pocket(mol, rng)
    no_conf = Chem.MolFromSmiles("CCO")  # no conformer at all
    X = featurize_batch([mol, None, no_conf], pocket)
    assert X.shape == (3, FEATURE_DIM)
    assert np.isfinite(X).all()
    assert np.abs(X[1]).max() == 0.0
    assert np.abs(X[2][:N_CONTACT]).max() == 0.0  # no pose -> no contacts
    assert np.abs(X[2][N_DENSE:]).max() > 0.0     # but the 2D block is still there
