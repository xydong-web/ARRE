"""Paper-named, *lazy* compatibility exports for the verified ARRE classes.

Use as a --trainer module:Class path with the integrated source on PYTHONPATH.
Module import itself avoids initializing PyTorch/CUDA on the login node.
The class objects remain the original classes to keep checkpoint compatibility.
"""
import importlib

PAPER_TO_IMPLEMENTATION = {
    'ARRE': 'cgflow.lanes.combo:VinaMOOTrainer_ARRE',
    'RHC': 'cgflow.lanes.rhc:VinaMOOTrainer_RHC',
    'RFG+SPP': 'cgflow.lanes.rfg_spp:VinaMOOTrainer_RFGSPP',
}

_ALIAS = {
    'ARRETrainer': PAPER_TO_IMPLEMENTATION['ARRE'],
    'RHCTrainer': PAPER_TO_IMPLEMENTATION['RHC'],
    'RFGSPPTrainer': PAPER_TO_IMPLEMENTATION['RFG+SPP'],
}

__all__ = ['ARRETrainer', 'RHCTrainer', 'RFGSPPTrainer', 'PAPER_TO_IMPLEMENTATION']


def __getattr__(name):
    spec = _ALIAS.get(name)
    if spec is None:
        raise AttributeError(name)
    module, symbol = spec.split(':', 1)
    return getattr(importlib.import_module(module), symbol)
