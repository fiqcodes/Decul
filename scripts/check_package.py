#!/usr/bin/env python3
"""Static preflight, never opens licensed raw match tracking or unpickles a model."""
from pathlib import Path
import hashlib, json, sys
ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'models/model.joblib':'69db493a71c99b53cfec4032f5b27429be67d1d9ba349af725ec28796bfb5282',
            'src/run_v6.py':'a5ae204fd1a7db1ba629068a598eb4aee96fa37cded7d4b2e9f39c56420569ff',
            'src/models_v6.py':'45bc4ca15b3a9e1b35fbda1f3492675d2b551d9f55718e6fdff952bbdd0dfe42',
            'local_modules/possession_features.py':'fe0281a72ff88414a4dfbd917677284b512fb4abfecd28661f28c567dd55309f'}
REQUIRED = ['games.py','starter/baseline.py','starter/load_atd.py','local_modules/possession_features.py',
            'evaluation/possession_metrics.py','evaluation/validate_submission.py',
            'third_party/METRICA_LICENSE.txt','configs/bar_ath_feature_options.json',
            'configs/feature_metadata_public.json',
            'data/derived/idsse/J03WPY/idsse_feature_contract.json']
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def check():
    problems=[]
    for rel,expected in EXPECTED.items():
        p=ROOT/rel
        if not p.is_file() or digest(p)!=expected: problems.append('Missing or changed frozen asset: '+rel)
    p=ROOT/'models/experiment.json'
    if not p.is_file(): problems.append('Missing model experiment manifest')
    else:
        ex=json.loads(p.read_text())
        if ex['submission']['sha256']!='020ebd0cdac4f6bd8228cf1f27193c123b7c73fe872d5655a182300c4556c250':
            problems.append('Unexpected submission digest in manifest')
    missing=[p for p in REQUIRED if not (ROOT/p).is_file()]
    # Public license notice must be distributed along with copied Metrica source.
    if (ROOT/'third_party/METRICA_LICENSE.txt').is_file():
        notice=(ROOT/'third_party/METRICA_LICENSE.txt').read_text()
        if 'MIT' not in notice or 'Metrica Sports' not in notice:
            problems.append('Original Metrica MIT license notice is invalid')
    print('Frozen model/source hashes:', 'PASS' if not any('frozen' in x for x in problems) else 'FAIL')
    print('Exact shared source dependencies:', 'COMPLETE' if not missing else 'INCOMPLETE')
    if missing:
        print('Missing original files:')
        for path in missing: print('  -',path)
        print('Run scripts/complete_repository.py against the original local project.')
    for error in problems: print('ERROR:',error)
    return 0 if not problems and not missing else 1
if __name__=='__main__': sys.exit(check())
