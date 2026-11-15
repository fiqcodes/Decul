#!/usr/bin/env python3
"""Compact V6 workflow: inspect -> nested development CV -> full fit -> inference.

Only existing local IDSSE labels are used for learning. No downloads, no target
labels, no dependency on another experimental version, and no overwriting v1-v5.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,sys,time,gc,os,shutil
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
import models_v6 as M
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'games.py').exists())
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def load_source(mid):
 d=ROOT/'data/derived/idsse'/mid
 with np.load(d/'idsse_tracking_adapter.npz',allow_pickle=False) as t:p,b,teams,st=[t[k].copy() for k in ['players','ball','team_slots','period_start']]
 y=pd.read_parquet(d/'idsse_training_labels.parquet',columns=['target_candidate']).target_candidate.map({'H':0,'A':1,'D':2});assert y.notna().all()
 return p,b,teams,st.astype(bool),y.to_numpy(dtype=np.int8)

def protocol():
 return dict(version='V6_sequence_context',created_utc=datetime.now(timezone.utc).isoformat(),data='Existing local IDSSE seven matches only; no new external data acquired',matches=M.MATCHES,outer_folds=M.OUTER_FOLDS,
  selection='Three outer match-group folds. Within each outer-training set, two disjoint inner match-group folds select learner and decoder. Outer labels never select that outer model or preprocessing. A fixed causal84 reference is independently refitted on the same outer training matches.',
  final_selection='Mean inner validation objective across the six inner folds chooses the final all-seven-match fit; this is a final development choice, not an independently evaluated fixed-model score.',
  historical_caveat='All seven matches have been inspected in earlier versions. Nested grouping prevents within-run fit leakage but cannot erase research-history selection bias. Results are development evidence, not a pristine holdout.',
  configs=M.CONFIGS,decoder_options=M.DECODERS,unsupervised='PCA 8 components plus 8 tracking-regime distances fitted only on current training-fold tracking; no target adaptation',
  feature_selection='Use all 84 compatible input measurements; V1 top-60 selection is not inherited. Model family/regularization/context/decoder selected inside inner folds; no global label-based feature ranking.',
  target_features='Raw target tracking is first read for final inference only. No target possession labels or Smart Tagging teacher channels are used.',
  offline_context='Before/after tracking within each half, symmetric 1s/5s summaries and last/next observed ball context. This is offline inference, not a real-time or causal model.',
  scenarios=M.SCENARIOS,scenario_seeds='6006 + 1009 * match_index + 101 * scenario_index',training_view_weights={'clean':.4,'original_dropout':.2,'short_gaps':.2,'long_gaps':.2},
  sampling_hz=5,evaluation_hz=25,hmm='New transition/initial/prior estimates from current training-fold labels at 5 Hz; no frozen V1 HMM. Frame predictions are expanded within each period before official evaluation.',
  objective='.5 * mean clean SpellF1 + .5 * mean of three dropout conditions, equal match weighting',
  screening='Compare nested V6 with the independently refitted causal84 reference; also show frozen v1 only on the three matches excluded from v1 training. Never score frozen v1 on its own training fixtures as an unseen reference.',
  no_automatic_submission=True)

def inspect_inputs():
 cfg=M.config(ROOT)
 for mid in M.MATCHES:
  d=ROOT/'data/derived/idsse'/mid
  for f in ['idsse_features_84.parquet','idsse_tracking_adapter.npz','idsse_training_labels.parquet','idsse_feature_contract.json']:assert (d/f).is_file(),d/f
 assert len(cfg['names'])==84 and not any(n.startswith('tag_') for n in cfg['names'])
 print('INPUTS PASS: 7 existing IDSSE fixtures, 84 tracking features, no downloads.',flush=True)
 return cfg

def prepare_cache(out,cfg):
 folder=out/'.cache';folder.mkdir(exist_ok=True);cache_path=folder/'features.joblib'
 if cache_path.exists():return joblib.load(cache_path)
 baseline_dir=ROOT/'data/derived/idsse/trained_model_v2';recipe=json.loads((baseline_dir/'training_recipe.json').read_text());frozen=XGBClassifier();frozen.load_model(baseline_dir/'xgboost_model.ubj')
 with np.load(baseline_dir/'hmm_parameters.npz',allow_pickle=False) as h:old_hmm=dict(transition=h['transition_matrix'],initial=h['initial_probabilities'],prior=h['class_prior'])
 data={}
 for j,mid in enumerate(M.MATCHES):
  p,b,teams,starts,y=load_source(mid);data[mid]={}
  for si,scenario in enumerate(M.SCENARIOS):
   if scenario=='clean':base=pd.read_parquet(ROOT/'data/derived/idsse'/mid/'idsse_features_84.parquet').astype(np.float32);missing=~np.isfinite(b).all(1)
   else:
    ap,ab=M.augment(p,b,starts,scenario,M.SEED+1009*j+101*si);base=M.build84(ap,ab,teams,starts,cfg);missing=~np.isfinite(ab).all(1);del ap,ab
   assert base.columns.tolist()==cfg['names']
   if j==0 and scenario=='clean':
    recomputed=M.build84(p,b,teams,starts,cfg);assert np.array_equal(recomputed.to_numpy(),base.to_numpy(),equal_nan=True);del recomputed
   grid,ix,st,expand=M.sample_grid(base,starts);case=dict(X=grid,y_grid=y[ix],y_full=y,starts_grid=st,starts_full=starts,expand=expand,missing_full=missing)
   if mid in recipe['validation_matches']+recipe['test_matches']:
    prob=frozen.predict_proba(base[recipe['selected_feature_columns_ordered']]);pred=M.decode(prob,starts,old_hmm,'prior_1');case['frozen_v1']=M.score_full(case,pred,pm)
    if mid=='J03WR9' and scenario=='clean':assert abs(case['frozen_v1']['spellF1']-.5923896499238965)<1e-10
   data[mid][scenario]=case;del base;gc.collect();print('CACHE',mid,scenario,grid.shape,flush=True)
  del p,b;gc.collect()
 joblib.dump(data,cache_path,compress=3);return data

def fit_model(data,matches,name,cfg):
 chunks=[];ys=[];weights=[];sequence=[]
 for mid in matches:
  for scenario in M.SCENARIOS:
   c=data[mid][scenario];chunks.append(c['X']);ys.append(c['y_grid']);w=.4 if scenario=='clean' else .2;weights.append(np.full(len(c['X']),w/len(c['X']),dtype=np.float32))
  c=data[mid]['clean'];sequence.append((c['y_grid'],c['starts_grid']))
 X=pd.concat(chunks,ignore_index=True);y=np.concatenate(ys);w=np.concatenate(weights);w*=len(y)/w.sum()
 learner=M.Learner(name,cfg['names']).fit(X,y,w);hmm=M.estimate_hmm(sequence)
 return learner,hmm

def evaluate_models(data,matches,learner,hmm,decoders,stage,outer,inner,rows):
 for mid in matches:
  for scenario in M.SCENARIOS:
   c=data[mid][scenario];prob=learner.predict_proba(c['X'])
   for decoder in decoders:rows.append(dict(stage=stage,outer=outer,inner=inner,match=mid,scenario=scenario,model=learner.name,decoder=decoder,**M.evaluate_case(c,prob,hmm,decoder,pm)))

def ranking(rows):
 df=pd.DataFrame(rows);scores=[]
 for (name,decoder),g in df.groupby(['model','decoder']):
  clean=float(g[g.scenario=='clean'].spellF1.mean());drop=float(g[g.scenario!='clean'].spellF1.mean());scores.append(dict(model=name,decoder=decoder,objective=.5*(clean+drop),clean=clean,dropout=drop))
 return sorted(scores,key=lambda r:(-r['objective'],r['model'],r['decoder']))

def nested_train(out,cfg,data,exp):
 rows=[];outer_choices=[];runtime=time.time()
 for oi,heldout in enumerate(M.OUTER_FOLDS):
  train=[mid for mid in M.MATCHES if mid not in heldout];inner_sets=[train[::2],train[1::2]];inner_rows=[]
  for ii,valid in enumerate(inner_sets):
   fit_ids=[mid for mid in train if mid not in valid];assert set(fit_ids).isdisjoint(valid) and set(train).isdisjoint(heldout)
   for name in M.CONFIGS:
    cached=out/'.cache'/f'inner_{oi}_{ii}_{name}.joblib'
    start=time.time()
    if cached.exists():learner,hmm=joblib.load(cached)
    else:learner,hmm=fit_model(data,fit_ids,name,cfg);joblib.dump((learner,hmm),cached,compress=3)
    evaluate_models(data,valid,learner,hmm,M.DECODERS,'inner',oi,ii,inner_rows)
    print('INNER',oi,ii,name,'train',fit_ids,'validate',valid,'seconds',round(time.time()-start,1),flush=True)
    pd.DataFrame(rows+inner_rows).to_csv(out/'validation.csv',index=False)
    del learner;gc.collect()
  choice=ranking(inner_rows)[0];outer_choices.append(dict(outer=oi,train=train,heldout=heldout,selection=choice));rows.extend(inner_rows)
  print('OUTER CHOICE',oi,choice,flush=True)
  # Outer scores are recorded after inner selection and never fed back into it.
  for role,name,decoder in [('nested_v6',choice['model'],choice['decoder']),('causal_reference','causal84','prior_1')]:
   learner,hmm=fit_model(data,train,name,cfg);local=[];evaluate_models(data,heldout,learner,hmm,[decoder],role,oi,-1,local);rows.extend(local);del learner;gc.collect()
  for mid in heldout:
   for scenario in M.SCENARIOS:
    if 'frozen_v1' in data[mid][scenario]:rows.append(dict(stage='frozen_v1_reference',outer=oi,inner=-1,match=mid,scenario=scenario,model='frozen_v1',decoder='original_25Hz',**data[mid][scenario]['frozen_v1']))
  exp['outer_choices']=outer_choices;save(out/'experiment.json',exp);pd.DataFrame(rows).to_csv(out/'validation.csv',index=False)
 final_choice=ranking([r for r in rows if r['stage']=='inner'])[0]
 exp['final_choice']=final_choice;exp['nested_runtime_seconds']=time.time()-runtime;save(out/'experiment.json',exp)
 learner,hmm=fit_model(data,M.MATCHES,final_choice['model'],cfg)
 payload=dict(learner=learner,hmm=hmm,decoder=final_choice['decoder'],feature_names=cfg['names'],training_matches=M.MATCHES,version='V6',sampling_stride=M.STRIDE)
 joblib.dump(payload,out/'model.joblib',compress=3);exp['model_sha256']=sha(out/'model.joblib');save(out/'experiment.json',exp);write_report(out)
 return payload

def raw_target(cfg):
 """Read target tracking ONLY for final inference, then reset features per half."""
 import games
 from starter import baseline,load_atd
 from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata
 from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data
 paths=games.asset_paths(source='tactical');teams=np.asarray(baseline.team_of_each_track(paths['atd_meta']),dtype=np.int8);n=games.source_frames('tactical');pitch=np.asarray(games.GAMES[games.GAME]['pitch'],dtype=np.float32)
 p=np.full((n,len(teams),2),np.nan,dtype=np.float32);b=np.full((n,2),np.nan,dtype=np.float32)
 with open(paths['atd_meta'],'rb') as f:meta=load_metadata(f)
 with open(paths['atd_raw'],'rb') as f:
  for i,row in enumerate(read_raw_data(f,meta)):
   assert load_atd.video_frame_of(row['frame_id'])==i
   p[i]=[[row[f'player_{k}_x'],row[f'player_{k}_y']] for k in range(1,len(teams)+1)];b[i]=[row['ball_x'],row['ball_y']]
 assert i+1==n;p[~np.isfinite(p).all(2)]=np.nan;b[~np.isfinite(b).all(1)]=np.nan;p*=pitch;b*=pitch
 ids=[];starts=[]
 for half,left,right in games.half_video_blocks(games.GAME):ids.extend(range(left,right+1));starts.extend(np.arange(right-left+1)==0)
 ids=np.asarray(ids);starts=np.asarray(starts,dtype=bool);features=M.build84(p[ids],b[ids],teams,starts,cfg)
 grid,ix,st,expand=M.sample_grid(features,starts)
 return grid,st,expand,ids,n

def predict(out,cfg,reproduce=False):
 exp=json.loads((out/'experiment.json').read_text());assert sha(out/'model.joblib')==exp['model_sha256'];payload=joblib.load(out/'model.joblib');grid,st,expand,ids,n=raw_target(cfg);prob=payload['learner'].predict_proba(grid);pred=M.decode(prob,st,payload['hmm'],payload['decoder'])[expand]
 labels=np.full(n,'X',dtype='<U1');labels[ids]=np.array(['H','A','D'])[pred];submission=pd.DataFrame({'frame':np.arange(n,dtype=np.int32),'label':labels});assert pm.validate_submission(submission,n) is None
 name='Decul_V6_reproduced.csv' if reproduce else 'Decul_V6.csv';path=out/name
 if path.exists():raise FileExistsError(f'Preserve existing output: {path}')
 submission.to_csv(path,index=False);digest=sha(path)
 if reproduce:assert digest==exp['submission']['sha256']
 else:exp['submission']=dict(file=name,sha256=digest,rows=n,label_counts=submission.label.value_counts().to_dict(),target_labels_used=False,target_accuracy_unknown=True);save(out/'experiment.json',exp);write_report(out)
 print('PREDICTION PASS',path,digest,flush=True)

def write_report(out):
 df=pd.read_csv(out/'validation.csv');exp=json.loads((out/'experiment.json').read_text());outer=df[df.stage.isin(['nested_v6','causal_reference'])];summaries=[]
 for stage,g in outer.groupby('stage'):
  clean=g[g.scenario=='clean'].spellF1.mean();drop=g[g.scenario!='clean'].spellF1.mean();summaries.append(dict(model=stage,clean_spellF1=clean,dropout_spellF1=drop,objective=.5*(clean+drop),macroF1=g.macroF1.mean(),short_recall=g.short_recall.mean()))
 summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False)
 def table(frame):
  text=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
  for row in frame.itertuples(index=False,name=None):text.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
  return '\n'.join(text)
 subset=df[df.stage.isin(['nested_v6','frozen_v1_reference']) & df.match.isin(['J03WOY','J03WQQ','J03WR9'])];paired=[]
 for stage,g in subset.groupby('stage'):paired.append(dict(model=stage,clean_spellF1=g[g.scenario=='clean'].spellF1.mean(),dropout_spellF1=g[g.scenario!='clean'].spellF1.mean()))
 bymatch=outer.groupby(['match','stage']).apply(lambda g:pd.Series({'objective':.5*(g[g.scenario=='clean'].spellF1.mean()+g[g.scenario!='clean'].spellF1.mean()),'clean':g[g.scenario=='clean'].spellF1.mean(),'dropout':g[g.scenario!='clean'].spellF1.mean()}),include_groups=False).reset_index()
 selected=exp.get('final_choice',{})
 text=f'''# Decul V6 — offline sequence-context research

## Completed experiment

V6 trains new classifiers and temporal models from existing IDSSE data. It does
not inherit V1's selected 60 features, XGBoost hyperparameters or HMM parameters.
All 84 common tracking measurements are available. Offline before/after context
is optional; another ablation adds training-fold-only unsupervised PCA/regimes.
No new external data was downloaded. Smart Tagging and BAR–ATH labels are unused.

## Nested match-group development results

{table(summary)}

Each outer match appears exactly once in evaluation. Its labels cannot fit that
outer model, feature normalizer, regime encoder, HMM or inner choice. The causal
reference is independently retrained on the same outer training matches. It is
not the frozen V1 model. Configurations and decoder weights are selected only from
inner match-group validation, using equal clean/dropout and match weights.
All seven fixtures were explored in earlier project versions, so these remain
development results, not a pristine external test or a guarantee of BAR–ATH accuracy.

## Per-match results

{table(bymatch)}

## Comparison with frozen v1 on its three non-training fixtures

{table(pd.DataFrame(paired))}

Only J03WOY, J03WQQ and J03WR9 are used for this frozen-v1 comparison. The frozen
model is never treated as unseen on its own four training matches. V6's outer
training groups differ from V1's original four-match training pool, so this is a
paired input comparison with different training pools, not a pure architecture
ablation. The separate causal-reference comparison controls the training pool.

## Final all-seven-match model

Selected configuration: `{selected.get('model')}`. Decoder: `{selected.get('decoder')}`.
The final choice uses mean inner-validation scores, then refits on all seven
external matches. Outer scores estimate the nested selection procedure, not an
untouched test of this exact final fit. Full configuration, folds, source hashes,
checks and model hash are recorded in experiment.json.

Predictions run at a 5 Hz grid and expand to 25 Hz within each half. HMM estimates
use the same 5 Hz time scale. Offline context uses future tracking observations,
not future possession labels, and cannot cross a half or match boundary. This is
suitable for completed-match inference; do not describe it as a live model.

The final CSV, when present, is Decul_V6.csv. It is a separate candidate and does
not overwrite any earlier submission. Raw target reconstruction is performed only
after model selection and reads no target possession labels or tag teachers.
Target prediction accuracy remains unknown. Nothing is uploaded or submitted.
'''
 (out/'report.md').write_text(text)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','inspect','train','predict','report'],default='inspect');parser.add_argument('--run-dir');args=parser.parse_args()
 if args.mode=='smoke':print(M.smoke());return
 cfg=inspect_inputs()
 if args.mode=='inspect':return
 if args.mode in ['predict','report']:
  assert args.run_dir,'Provide --run-dir';out=Path(args.run_dir).resolve()
  if args.mode=='predict':predict(out,cfg,reproduce=True)
  else:write_report(out)
  return
 out=Path(args.run_dir).resolve() if args.run_dir else HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_utc');out.mkdir(parents=True,exist_ok=False)
 exp=protocol();exp['smoke_checks']=M.smoke();exp['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'models_v6.py',ROOT/'local_modules/possession_features.py']};exp['preserved_versions']={str(p.relative_to(ROOT)):sha(p) for folder in [ROOT/'notebook_2/v3_dual_expert',ROOT/'notebook_2/v4_ssl_smarttag'] for p in folder.glob('*.py')};save(out/'experiment.json',exp);print('RUN DIRECTORY',out,flush=True)
 data=prepare_cache(out,cfg);nested_train(out,cfg,data,exp);del data;gc.collect();predict(out,cfg)
 for relative,digest in exp['preserved_versions'].items():assert sha(ROOT/relative)==digest
 print('V6 COMPLETE',out,flush=True)

if __name__=='__main__':main()
