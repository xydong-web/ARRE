#!/usr/bin/env python3
"""Recalculate central ARRE claims from frozen per-pocket contrast vectors.

This independently recalculates summary statistics from the 20 pocket-level
differences archived in trial.json and portability.json; it does NOT rerun the
64,000-molecule GPU trials or independently recompute raw docking scores.
"""
from pathlib import Path
import argparse
import json
import math
import statistics

from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'manuscript' / 'source' / 'data'


def calculate(rows: list[float], level: float):
    n = len(rows)
    mean = statistics.mean(rows)
    sd = statistics.stdev(rows)
    sem = sd / math.sqrt(n)
    quantile = student_t.ppf((1 + level) / 2, n - 1)
    return dict(n=n, mean=mean, sd=sd, ci=[mean-quantile*sem, mean+quantile*sem])


def compare(label, record, confidence, saved_ci, expected_mean):
    assert len(record['diffs']) == 20, f'{label}: expected 20 pocket differences'
    result = calculate(record['diffs'], confidence)
    # Compare using the precision actually printed in the paper (0.57 pp
    # is rounded to two decimals, whereas docking has three decimals).
    decimals = len(str(expected_mean).split('.')[-1])
    rounding_tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    assert abs(result['mean'] - expected_mean) <= rounding_tolerance, label
    assert abs(result['mean'] - record['mean']) < 1e-9, label
    assert max(abs(a-b) for a,b in zip(result['ci'], record[saved_ci])) < 1e-8, label
    result.update(claim=label, confidence=confidence, expected_paper=expected_mean,
                  n_favorable=sum(x > 0 if 'planning' in label else x < 0 for x in record['diffs']))
    return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--json',action='store_true')
    args=a.parse_args()
    trial=json.loads((DATA/'trial.json').read_text())
    portable=json.loads((DATA/'portability.json').read_text())
    records=[
        compare('ARRE minus CGFlow size-adjusted Vina',
                trial['P1_m1_adj_R1']['F-B'], 0.95,'ci95',-0.981),
        compare('ARRE minus RxnFlow size-adjusted Vina',
                portable['held20']['contrasts']['m1_adj']['RXCE-RX'],0.95,'ci95',-1.009),
        compare('Policy-frozen ARRE minus SynGBO size-adjusted Vina',
                trial['P1_m1_adj_R1']['FZ-SynGBO'],0.95,'ci95',-0.292),
        compare('Policy-frozen ARRE minus SynGBO Retro*--MHNreact planning, percentage points',
                trial['P2']['FZ_vs_SynGBO_noninferiority'],0.90,'ci90',0.57),
    ]
    assert records[-1]['ci'][0] > -5.0, 'Prespecified noninferiority bound not met'
    if args.json:
        print(json.dumps({'source':'frozen 20-pocket difference arrays',
                          'independent_gpu_rerun':False,
                          'claims':records,
                          'noninferiority_margin_pp':-5.0},indent=2))
    else:
        for r in records:
            lo,hi=r['ci']
            print(f"{r['claim']}: n={r['n']}, mean={r['mean']:+.4f}, "
                  f"{int(r['confidence']*100)}% CI [{lo:+.4f}, {hi:+.4f}]")
        print('Policy-frozen ARRE independent planning noninferiority against -5 pp: PASSED')


if __name__=='__main__':
    main()
