#!/usr/bin/env python3
"""Generate Decul.csv from permitted raw tracking and frozen model, with exact digest."""
from pathlib import Path
import sys, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_package import check

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    if check(): raise SystemExit('Repository is missing original dependencies or a frozen hash does not match.')
    sys.path.insert(0,str(ROOT/'src'))
    import models_v6 as M
    import run_v6 as R
    # The original V6 implementation is preserved without renaming its Python classes.
    cfg=M.config(ROOT)
    target=ROOT/'output/Decul.csv'
    if target.exists(): raise FileExistsError('Refuse to overwrite an existing target: '+str(target))
    temporary=ROOT/'output/_original_run'
    temporary.mkdir(parents=True,exist_ok=True)
    for name in ('model.joblib','experiment.json'):
        path=temporary/name
        if path.exists(): raise FileExistsError('Remove stale work directory first: '+str(path))
        shutil.copy2(ROOT/'models'/name,path)
    R.predict(temporary,cfg,reproduce=True)
    produced=temporary/'Decul_V6_reproduced.csv'
    expected=json.loads((ROOT/'models/experiment.json').read_text())['submission']['sha256']
    if sha(produced)!=expected: raise RuntimeError('Predictions do not match frozen SHA256')
    shutil.copyfile(produced,target)
    if sha(target)!=expected: raise RuntimeError('Output copy did not match')
    print('PASS: Decul.csv matches frozen SHA256:', expected)
    print('Output:',target)
if __name__=='__main__': main()
