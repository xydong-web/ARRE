#!/usr/bin/env python3
"""Rebuild every quantitative panel from frozen evidence.

Runs Figure 2–8 and Supporting Figures S1–S3, retaining the original
PowerPoint-rendered Figure 1 and the journal-sized custom graphical abstract.
No model training, docking, molecule generation or planning takes place.
"""
import json

import figures as f


def main():
    source_qa = f.PROJECT / 'figure_qa.json'
    source_panels = f.PROJECT / 'panel_sources.json'
    earlier_qa = json.loads(source_qa.read_text())
    earlier_panels = json.loads(source_panels.read_text())
    f.benchmark()
    f.search()
    f.pocket()
    f.mechanism()
    f.transfer()
    f.properties()
    f.validation()
    f.supplementary()
    generated_qa = {q['figure'] for q in f.QA}
    generated_panels = {str(q['figure']) for q in f.PANEL}
    retained_qa = [q for q in earlier_qa if q['figure'] not in generated_qa]
    for q in retained_qa:
        if q.get('figure') == 'Graphical_Abstract':
            q.update(width_mm=110, height_mm=20)
    retained_panels = [q for q in earlier_panels if str(q['figure']) not in generated_panels]
    source_qa.write_text(json.dumps(retained_qa + f.QA, indent=2) + '\n')
    source_panels.write_text(json.dumps(retained_panels + f.PANEL, indent=2) + '\n')
    print('QUANTITATIVE PANELS REBUILT: 7 main + 3 SI; original Figure 1/ToC preserved')


if __name__ == '__main__':
    main()
