#!/usr/bin/env python3
"""Paper-named CGFlow/ARRE experiment command generator (dry-run by default).

Uses the public RHC and RFG/SPP configuration keys. Use --execute on a
scheduled compute node to run the printed command.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'code' / 'frozen_prior' / 'syn_prior_v1.frozen.json'
ORIGINAL_RUNNER = ROOT / 'code' / 'entrypoints' / 'run_opt.py'
PYTHONPATH = ':'.join((str(ROOT / 'code' / 'ARRE_PaperAPI'),
                       str(ROOT / 'code' / 'integrated_ARRE' / 'src')))

TRAINER = {
    'ARRE': 'arre_paper_names:ARRETrainer',
    'RHC': 'arre_paper_names:RHCTrainer',
    'RFG': 'arre_paper_names:RFGSPPTrainer',
    'SPP': 'arre_paper_names:RFGSPPTrainer',
    'F': 'arre_paper_names:ARRETrainer',
    'FZ': 'arre_paper_names:ARRETrainer',
}


def switches(profile):
    if profile in ('ARRE', 'F', 'FZ'):
        result = ['rhc.enable=true', 'rfg_spp.enable=true',
                  'rfg_spp.tilt=1.0', 'rfg_spp.group_norm=true',
                  'rfg_spp.syn_kappa=4.0', 'rfg_spp.syn_lookahead=true',
                  f'rfg_spp.syn_prior={PRIOR}']
    elif profile == 'RHC':
        result = ['rhc.enable=true', 'rfg_spp.enable=false']
    elif profile == 'RFG':
        result = ['rhc.enable=false', 'rfg_spp.enable=true',
                  'rfg_spp.tilt=1.0', 'rfg_spp.group_norm=true',
                  'rfg_spp.syn_kappa=0.0', 'rfg_spp.syn_lookahead=false']
    elif profile == 'SPP':
        result = ['rhc.enable=false', 'rfg_spp.enable=true',
                  'rfg_spp.tilt=0.0', 'rfg_spp.syn_kappa=4.0',
                  'rfg_spp.syn_lookahead=true',
                  f'rfg_spp.syn_prior={PRIOR}']
    else:
        raise ValueError(profile)
    if profile == 'FZ':
        # Archived F/FZ delta is exactly a frozen generator and logZ update.
        result += ['opt.learning_rate=0.0', 'algo.tb.Z_learning_rate=0.0']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=TRAINER, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    p.add_argument('--env-dir', required=True)
    p.add_argument('--pocket-dir', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--log-dir', required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--execute', action='store_true',
                   help='EXPERIMENTAL: run a full expensive molecular search')
    args = p.parse_args()
    cmd = [args.python, str(ORIGINAL_RUNNER),
           '--trainer',TRAINER[args.profile],
           '--target',args.target,'--seed',str(args.seed),
           '--env_dir',args.env_dir,'--pocket_dir',args.pocket_dir,
           '--ckpt',args.ckpt,'--log_dir',args.log_dir,
           '--steps',str(args.steps)]
    for flag in switches(args.profile):
        cmd += ['--set', flag]
    print(json.dumps({'paper_profile':args.profile,
                      'legacy_config_example':{
                          'ARRE':'r2CE4L4', 'F':'x4nw2F', 'FZ':'x4nw2FZ'
                      }.get(args.profile), 'command':cmd,
                      'purpose':'new run; not an exact replay of every historical scheduler flag',
                      'run_requested':bool(args.execute)},indent=2))
    if args.execute:
        env = os.environ.copy()
        env['PYTHONPATH'] = PYTHONPATH + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        print('Launching user-requested expensive experiment:',shlex.join(cmd),flush=True)
        subprocess.run(cmd,check=True,env=env)


if __name__=='__main__':main()
