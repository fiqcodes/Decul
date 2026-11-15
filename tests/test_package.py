from pathlib import Path
import hashlib,json,ast
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def test_original_model_digest():
    assert sha('models/model.joblib')=='69db493a71c99b53cfec4032f5b27429be67d1d9ba349af725ec28796bfb5282'
def test_original_source_digests():
    assert sha('src/run_v6.py')=='a5ae204fd1a7db1ba629068a598eb4aee96fa37cded7d4b2e9f39c56420569ff'
    assert sha('src/models_v6.py')=='45bc4ca15b3a9e1b35fbda1f3492675d2b551d9f55718e6fdff952bbdd0dfe42'
def test_original_manifest():
    exp=json.loads((ROOT/'models/experiment.json').read_text())
    assert exp['final_choice']['model']=='context_deep'
    assert exp['final_choice']['decoder']=='prior_4'
    assert exp['submission']['rows']==199319
    assert exp['submission']['sha256']=='020ebd0cdac4f6bd8228cf1f27193c123b7c73fe872d5655a182300c4556c250'
def test_scripts_syntax():
    for p in (ROOT/'scripts').glob('*.py'): ast.parse(p.read_text())


def test_custom_dependencies_exact():
    assert sha('local_modules/possession_features.py')=='fe0281a72ff88414a4dfbd917677284b512fb4abfecd28661f28c567dd55309f'
    assert sha('configs/bar_ath_feature_options.json')=='3c8737a23604babde067494d37ed877a12fec8c5c07ac2011bd7a7abb34f18f5'
    assert sha('configs/feature_metadata_public.json')=='8a74147b89d76a1233f4baa054c7f3eadde700016516a7ada42b629850004e29'
    assert sha('data/derived/idsse/J03WPY/idsse_feature_contract.json')=='460baacde0b0f6b51be845c639381b1bf49838e5de29ff34c7854822d87d7238'

def test_feature_contract_schema():
    names=json.loads((ROOT/'data/derived/idsse/J03WPY/idsse_feature_contract.json').read_text())['feature_columns']
    original=json.loads((ROOT/'configs/feature_metadata_public.json').read_text())['original_feature_columns']
    assert len(names)==84 and len(names)==len(set(names))
    assert not any(n.startswith('tag_') for n in names)
    assert len([c for c in original if not c.startswith('tag_')])==76


def test_publication_essentials():
    root=Path(__file__).resolve().parents[1]
    ignore=(root/'.gitignore').read_text()
    assert '!/data/derived/idsse/J03WPY/idsse_feature_contract.json' in ignore
    assert (root/'evaluation/validate_submission.py').is_file()
    assert (root/'third_party/METRICA_LICENSE.txt').is_file()
