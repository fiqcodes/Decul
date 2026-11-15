"""V6: IDSSE-only offline context features, train-fold regimes and new decoders.

No target possession labels, Smart Tagging teachers or pretrained v1 components
are used by V6. The common 84 tracking measurements are retained as input units.
"""
from pathlib import Path
import json,sys,warnings,itertools
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from numba import njit

FPS=25
STRIDE=5
SEED=6006
MATCHES=['J03WMX','J03WN1','J03WOH','J03WOY','J03WPY','J03WQQ','J03WR9']
SCENARIOS=['clean','original_dropout','short_gaps','long_gaps']
OUTER_FOLDS=[['J03WMX','J03WOY'],['J03WN1','J03WQQ'],['J03WOH','J03WPY','J03WR9']]
BASE_PARAMS=dict(objective='multi:softprob',num_class=3,eval_metric='mlogloss',tree_method='hist',random_state=SEED,n_jobs=4,subsample=.85,colsample_bytree=.9,reg_alpha=.1)
CONFIGS={
 'causal84':dict(context=False,regimes=False,n_estimators=400,max_depth=5,learning_rate=.05,min_child_weight=8,reg_lambda=5.),
 'context_regularized':dict(context=True,regimes=False,n_estimators=300,max_depth=4,learning_rate=.05,min_child_weight=10,reg_lambda=8.),
 'context_deep':dict(context=True,regimes=False,n_estimators=500,max_depth=6,learning_rate=.04,min_child_weight=5,reg_lambda=5.),
 'context_regimes':dict(context=True,regimes=True,n_estimators=400,max_depth=5,learning_rate=.04,min_child_weight=8,reg_lambda=5.),
}
DECODERS=['raw','prior_0.25','prior_1','prior_4','posterior_1']
SIGNALS=['ball_x','ball_y','ball_speed','proximity_advantage','control_dominance','home_distance_1','away_distance_1','visible_player_count','home_median_speed_mps','away_median_speed_mps','home_team_spread_m','away_team_spread_m']
ANCHORS=['ball_x','ball_y','ball_speed','proximity_advantage','control_dominance','home_distance_1','away_distance_1']

def config(root):
 root=Path(root);sys.path.insert(0,str(root/'local_modules'))
 from possession_features import FeatureBuilder
 names=json.loads((root/'data/derived/idsse/J03WPY/idsse_feature_contract.json').read_text())['feature_columns']
 original=json.loads((root/'configs/feature_metadata_public.json').read_text())['original_feature_columns']
 opts=json.loads((root/'configs/bar_ath_feature_options.json').read_text())['features']
 return dict(names=names,original_columns=original,tracking_columns=[n for n in original if not n.startswith('tag_')],opts=opts,FeatureBuilder=FeatureBuilder)

def eight_features(builder,ball,teams):
 n=len(ball);ix=np.arange(n);seen=np.isfinite(ball).all(1);last=np.maximum.accumulate(np.where(seen,ix,-1));out=pd.DataFrame(index=range(n))
 out['ball_missing_streak_s']=(np.where(seen,0,ix-last)/FPS).astype(np.float32)
 for side,code in [('home',0),('away',1)]:
  slots=np.flatnonzero(teams==code);speed=np.linalg.norm(builder.player_velocity[:,slots],axis=2);valid=np.isfinite(speed);count=valid.sum(1)
  with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(speed,axis=1)
  med[count<3]=np.nan;slow=np.divide(((speed<.6)&valid).sum(1),count,out=np.full(n,np.nan),where=count>=3)
  pos=builder.players[:,slots];validp=np.isfinite(pos).all(2);counts=validp.sum(1);centers=np.where(validp[...,None],pos,0).sum(1)/np.maximum(counts[:,None],1);spread=np.sqrt(np.nansum(np.sum((pos-centers[:,None])**2,2),1)/np.maximum(counts,1));spread[counts<3]=np.nan
  out[f'{side}_median_speed_mps']=med.astype(np.float32);out[f'{side}_slow_player_fraction']=slow.astype(np.float32);out[f'{side}_team_spread_m']=spread.astype(np.float32)
 visible=np.isfinite(builder.players).all(2).sum(1);change=np.full(n,np.nan,dtype=np.float32);change[125:]=visible[125:]-visible[:-125];out['player_count_change_5s']=change
 return out

def build84(players,ball,teams,starts,cfg):
 parts=[];edges=np.r_[np.flatnonzero(starts),len(ball)]
 for a,b in zip(edges[:-1],edges[1:]):
  builder=cfg['FeatureBuilder'](players[a:b],ball[a:b],teams,FPS,cfg['opts']);builder.spatial();builder.motion();builder.temporal()
  # These observation-only measurements lived in the legacy tagging stage.
  # Reconstruct them explicitly without calling or reading any tagging source.
  stationary=builder.ball_visible & np.isfinite(builder.ball_speed) & (builder.ball_speed<.4)
  builder.features['stationary_ball_fraction_2s']=builder.roll(stationary,50)
  rx=builder.roll(builder.ball[:,0],FPS,'max')-builder.roll(builder.ball[:,0],FPS,'min')
  ry=builder.roll(builder.ball[:,1],FPS,'max')-builder.roll(builder.ball[:,1],FPS,'min')
  builder.features['ball_span_1s_m']=np.hypot(rx,ry)
  builder.features['ball_coverage_1s']=builder.roll(builder.ball_visible,FPS)
  originals=builder.features[cfg['tracking_columns']].reset_index(drop=True);extra=eight_features(builder,ball[a:b],teams)
  parts.append(pd.concat([originals,extra],axis=1)[cfg['names']].astype(np.float32))
 return pd.concat(parts,ignore_index=True)

def dropout_mask(n,fraction,rng,ranges,weights):
 missing=np.zeros(n,dtype=bool);target=round(n*fraction);attempts=0
 while missing.sum()<target and attempts<max(1000,n//25):
  start=int(rng.integers(n));lo,hi=ranges[int(rng.choice(len(weights),p=weights))];end=min(n,start+max(1,int(rng.uniform(lo,hi)*25)));free=np.flatnonzero(~missing[start:end]);missing[start+free[:target-int(missing.sum())]]=True;attempts+=1
 assert missing.sum()==target
 return missing

def augment(players,ball,starts,scenario,seed):
 rng=np.random.default_rng(seed);p=players.copy();b=ball.copy();edges=np.r_[np.flatnonzero(starts),len(ball)]
 settings={'original_dropout':(.43,[(.2,5),(5,40),(40,220)],[.45,.4,.15]),'short_gaps':(.25,[(.2,2),(2,8)],[.65,.35]),'long_gaps':(.60,[(10,60),(60,180)],[.6,.4])}
 fraction,ranges,weights=settings[scenario]
 for start,end in zip(edges[:-1],edges[1:]):
  b[start:end][dropout_mask(end-start,fraction,rng,ranges,weights)]=np.nan
  for slot in range(p.shape[1]):p[start:end,slot][dropout_mask(end-start,.12,rng,[(.2,3),(3,20),(20,65)],[.45,.45,.1])]=np.nan
 return p,b

def context_features(base,starts):
 """Offline only: use unlabeled before/after tracking, never cross a period."""
 parts=[];edges=np.r_[np.flatnonzero(starts),len(base)]
 for a,b in zip(edges[:-1],edges[1:]):
  x=base.iloc[a:b].reset_index(drop=True);out={}
  for offset in [-15,-5,5,15]:
   shifted=x[SIGNALS].shift(offset)
   for name in SIGNALS:out[f'{name}_lag_{offset}']=shifted[name].to_numpy()
  for window in [5,25]:
   smooth=x[SIGNALS].rolling(window,center=True,min_periods=1).mean()
   for name in SIGNALS:out[f'{name}_symmetric_{window}']=smooth[name].to_numpy()
  visible=(x.ball_visible.to_numpy()>0);ix=np.arange(len(x));prev=np.maximum.accumulate(np.where(visible,ix,-1));nxt=np.minimum.accumulate(np.where(visible,ix,len(x))[::-1])[::-1]
  for label,index,valid in [('previous',prev,prev>=0),('next',nxt,nxt<len(x))]:
   safe=np.clip(index,0,len(x)-1)
   for name in ANCHORS:
    vals=x[name].to_numpy()[safe].copy();vals[~valid]=np.nan;out[f'{label}_observed_{name}']=vals
   out[f'{label}_observation_distance_s']=np.where(valid,np.abs(index-ix)/(FPS/STRIDE),np.nan)
  parts.append(pd.concat([x,pd.DataFrame(out)],axis=1).astype(np.float32))
 result=pd.concat(parts,ignore_index=True);assert len(result)==len(base);return result

def sample_grid(base,starts):
 ix=[];flags=[];expand=np.empty(len(base),dtype=np.int32);offset=0;edges=np.r_[np.flatnonzero(starts),len(base)]
 for a,b in zip(edges[:-1],edges[1:]):
  sample=np.arange(a,b,STRIDE);ix.extend(sample);flags.extend(np.arange(len(sample))==0);expand[a:b]=offset+np.minimum((np.arange(b-a)//STRIDE),len(sample)-1);offset+=len(sample)
 ix=np.array(ix);st=np.array(flags,dtype=bool);grid=base.iloc[ix].reset_index(drop=True)
 return context_features(grid,st),ix,st,expand

class RegimeEncoder:
 """Train-fold-only, label-free PCA and distances to tracking regimes."""
 def fit(self,X):
  rng=np.random.default_rng(SEED);sample=X[rng.choice(len(X),min(30000,len(X)),replace=False)]
  with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);self.median=np.nanmedian(sample,axis=0)
  self.median=np.nan_to_num(self.median);sample=np.where(np.isfinite(sample),sample,self.median);self.mean=sample.mean(0);self.std=np.maximum(sample.std(0),.05);z=np.clip((sample-self.mean)/self.std,-6,6)
  self.pca=PCA(n_components=8,random_state=SEED).fit(z);self.kmeans=MiniBatchKMeans(n_clusters=8,batch_size=2048,n_init=3,random_state=SEED).fit(z);return self
 def transform(self,X):
  z=np.clip((np.where(np.isfinite(X),X,self.median)-self.mean)/self.std,-6,6)
  return np.concatenate([self.pca.transform(z),self.kmeans.transform(z)],axis=1).astype(np.float32)

class Learner:
 def __init__(self,name,base_names):self.name=name;self.base_names=list(base_names);self.encoder=None
 def fit(self,X,y,weights):
  cfg=CONFIGS[self.name];self.columns=list(X.columns) if cfg['context'] else self.base_names
  design=X[self.columns].to_numpy(dtype=np.float32)
  if cfg['regimes']:
   self.encoder=RegimeEncoder().fit(X[self.base_names].to_numpy(dtype=np.float32));design=np.c_[design,self.encoder.transform(X[self.base_names].to_numpy(dtype=np.float32))]
  self.model=XGBClassifier(**BASE_PARAMS,**{k:v for k,v in cfg.items() if k not in ['context','regimes']});self.model.fit(design,y,sample_weight=weights);return self
 def predict_proba(self,X):
  design=X[self.columns].to_numpy(dtype=np.float32)
  if self.encoder is not None:design=np.c_[design,self.encoder.transform(X[self.base_names].to_numpy(dtype=np.float32))]
  return self.model.predict_proba(design)

def estimate_hmm(sequences):
 transition=np.ones((3,3));initial=np.ones(3);prior=np.ones(3)
 for y,starts in sequences:
  initial+=np.bincount(y[starts],minlength=3);valid=~starts[1:];np.add.at(transition,(y[:-1][valid],y[1:][valid]),1);prior+=np.bincount(y,minlength=3)
 return dict(transition=transition/transition.sum(1,keepdims=True),initial=initial/initial.sum(),prior=prior/prior.sum())

@njit(cache=True)
def viterbi(emission,transition,initial,starts):
 n=len(emission);out=np.empty(n,np.int8);back=np.zeros((n,3),np.int8);score=np.empty(3);last=0
 for t in range(n):
  if starts[t]:
   if t>0:
    out[t-1]=np.argmax(score)
    for j in range(t-1,last,-1):out[j-1]=back[j,out[j]]
   last=t;score=np.log(initial)+emission[t]
  else:
   new=np.empty(3)
   for k in range(3):
    best=0;value=score[0]+np.log(transition[0,k])
    for j in range(1,3):
     candidate=score[j]+np.log(transition[j,k])
     if candidate>value:best=j;value=candidate
    back[t,k]=best;new[k]=value+emission[t,k]
   score=new
 out[n-1]=np.argmax(score)
 for j in range(n-1,last,-1):out[j-1]=back[j,out[j]]
 return out

def decode(prob,starts,hmm,name):
 if name=='raw':return prob.argmax(1).astype(np.int8)
 method,weight=name.split('_');emission=np.log(np.clip(prob,1e-8,1))
 if method=='prior':emission-=np.log(hmm['prior'])
 return viterbi(float(weight)*emission,hmm['transition'],hmm['initial'],starts)

def evaluate_case(case,prob,hmm,decoder,pm):
 pred=decode(prob,case['starts_grid'],hmm,decoder)[case['expand']]
 return score_full(case,pred,pm)

def score_full(case,pred,pm):
 split=int(np.flatnonzero(case['starts_full'])[1]);truth=np.insert(case['y_full'],split,3);yp=np.insert(pred,split,3);r=pm.evaluate(truth,yp,taus=(25,),bootstrap_n=0)
 runs=pm.valid_runs(truth);gs=pm.possession_spells(truth,runs);ps=pm.possession_spells(yp,runs);short=np.array([b-a<50 for a,b,c in gs]);flags=pm.matched_gt_flags(pm.spell_overlaps(gs,ps),len(gs),.5);tr=r['transitions'][25];missing=case['missing_full']
 return dict(spellF1=r['spell_f1_mean'],macroF1=r['macro_f1'],accuracy=r['accuracy'],short_recall=float(flags[short].mean()) if short.any() else np.nan,short_count=int(short.sum()),transition_precision=tr['precision'],transition_recall=tr['recall'],missed_transitions=tr['n_missed'],spurious_transitions=tr['n_spurious'],missing_accuracy=float((pred[missing]==case['y_full'][missing]).mean()) if missing.any() else np.nan)

def smoke():
 rng=np.random.default_rng(71);h=dict(transition=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]),initial=np.array([.4,.35,.25]),prior=np.array([.3,.3,.4]))
 count=0
 for n in [1,3,6]:
  prob=rng.dirichlet(np.ones(3),size=n);st=np.arange(n)==0
  for name in ['prior_0.25','prior_1','prior_4','posterior_1']:
   method,weight=name.split('_');em=np.log(prob)-(np.log(h['prior']) if method=='prior' else 0);em*=float(weight);paths=list(itertools.product(range(3),repeat=n));scores=[np.log(h['initial'][p[0]])+sum(em[t,k] for t,k in enumerate(p))+sum(np.log(h['transition'][p[t-1],p[t]]) for t in range(1,n)) for p in paths];assert tuple(decode(prob,st,h,name))==paths[int(np.argmax(scores))];count+=1
 # Changing the other half cannot alter this half's look-ahead features.
 cols=list(set(SIGNALS+ANCHORS+['ball_visible']));base=pd.DataFrame(rng.normal(size=(40,len(cols))),columns=cols);base['ball_visible']=1.;st=np.zeros(40,dtype=bool);st[[0,20]]=True
 first=context_features(base,st).iloc[:20];base.iloc[20:]=999;second=context_features(base,st).iloc[:20];pd.testing.assert_frame_equal(first,second)
 return dict(exhaustive_decoder_cases=count,half_context_isolation=True,status='passed')
