#!/usr/bin/env python3
"""Complete the public Decul repository with verified original organiser source modules.

No raw match data, tracking caches, labels, or private experiment files are copied.
Static assembly alone is not proof of successful raw-ATD inference reproduction.
"""
from pathlib import Path
import argparse, shutil, hashlib, zipfile, sys
THIS=Path(__file__).resolve().parents[1]
OFFICIAL_REQUIRED=['games.py','starter/baseline.py','starter/load_atd.py','evaluation/possession_metrics.py',
                   'evaluation/validate_submission.py']
CUSTOM_REQUIRED=[
 'local_modules/possession_features.py',
 'configs/bar_ath_feature_options.json',
 'configs/feature_metadata_public.json',
 'data/derived/idsse/J03WPY/idsse_feature_contract.json'
]
ORIGINAL_CUSTOM_SHA={
 'local_modules/possession_features.py':'fe0281a72ff88414a4dfbd917677284b512fb4abfecd28661f28c567dd55309f',
 'configs/bar_ath_feature_options.json':'3c8737a23604babde067494d37ed877a12fec8c5c07ac2011bd7a7abb34f18f5',
 'configs/feature_metadata_public.json':'8a74147b89d76a1233f4baa054c7f3eadde700016516a7ada42b629850004e29',
 'data/derived/idsse/J03WPY/idsse_feature_contract.json':'460baacde0b0f6b51be845c639381b1bf49838e5de29ff34c7854822d87d7238'
}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--project-root',required=True,help='Original more-than-a-hack-2027 repository')
 args=ap.parse_args()
 project=Path(args.project_root).expanduser().resolve()
 if not project.is_dir(): raise SystemExit('Source project directory not found: '+str(project))
 # Fail on a different version of the custom feature builder and feature contract.
 for rel in CUSTOM_REQUIRED:
  installed=THIS/rel;source=project/rel
  if not installed.is_file() or sha(installed)!=ORIGINAL_CUSTOM_SHA[rel]:
   raise SystemExit('Packaged custom dependency is missing/changed: '+rel)
  if source.is_file() and sha(source)!=sha(installed):
   raise SystemExit('Custom dependency differs from original local project; review: '+rel)
 sources=list(OFFICIAL_REQUIRED)
 if (project/'starter').is_dir():
  sources+=sorted(str(p.relative_to(project)) for p in (project/'starter').rglob('*.py') if '__pycache__' not in p.parts)
 missing=sorted(set(rel for rel in sources if not (project/rel).is_file()))
 if missing:
  print('Missing official project source files:')
  for m in missing:print(' -',m)
  raise SystemExit(2)
 files=[]
 for rel in sorted(set(sources)):
  inp=project/rel
  dest=THIS/rel
  dest.parent.mkdir(parents=True,exist_ok=True)
  if inp.resolve()!=dest.resolve():shutil.copy2(inp,dest)
  files.append(rel)
 print('Copied official Python source:',len(files),'files')
 # Original licence notice must accompany copied Metrica source. It is stored
 # under third_party/ to avoid silently choosing a licence for Decul code.
 licence=project/'LICENSE'
 if not licence.is_file():raise SystemExit('Missing official source LICENSE: '+str(licence))
 licence_dest=THIS/'third_party/METRICA_LICENSE.txt'
 licence_dest.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(licence,licence_dest)
 if 'MIT' not in licence_dest.read_text() or 'Metrica Sports' not in licence_dest.read_text():
  raise SystemExit('Official license notice looks unexpected; review before publication.')
 print('Copied original organiser MIT license notice')
 sys.path.insert(0,str(THIS/'scripts'))
 from check_package import check
 if check(): raise SystemExit('Static dependency/hash check failed; do not publish.')
 out=project/'Decul_Source_Package.zip'
 if out.exists():raise SystemExit('Output ZIP already exists, preserving it: '+str(out))
 public_dirs={'src','models','results','docs','scripts','tests','starter','local_modules','evaluation','configs','third_party'}
 top_files={'README.md','requirements.txt','requirements-optional.txt','.gitignore','.gitattributes','games.py'}
 items=[p for p in THIS.iterdir() if p.is_file() and p.name in top_files]
 for folder in public_dirs:
  d=THIS/folder
  if d.is_dir():items += [p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts and p.suffix not in {'.pyc','.pyo'}]
 items += [THIS/p for p in CUSTOM_REQUIRED if str(p).startswith('data/')]
 # Keep the exact canonical custom feature-contract location even though it lives under data/.
 items=sorted(set(items))
 for p in items:
  if not p.is_file():raise SystemExit('Missing file: '+str(p))
  if p.suffix in {'.npz','.npy','.parquet','.csv'} and not str(p).startswith(str(THIS/'results')):
   raise SystemExit('Unexpected non-aggregate data file in package: '+str(p))
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in items:z.write(p,arcname='Decul/'+str(p.relative_to(THIS)))
 print('ASSEMBLY COMPLETE — STATIC CHECKS PASSED:',out)
 print('NOT YET VERIFIED FOR PUBLICATION: run exact raw ATD reproduction in local licensed environment, and review copied official source for license and private data.')
 print('Next: cd Decul && python scripts/reproduce_submission.py')

if __name__=='__main__':main()
