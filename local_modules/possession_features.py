"""Shared observation-only features for the notebook and independent scenario tests.

No target labels, simulator state, match schedule, or frame IDs enter the model features.
"""
import re
import numpy as np
import pandas as pd


def rolling(values, blocks, window, statistic='mean'):
    rolled = pd.Series(values, dtype=float).groupby(blocks, sort=False).rolling(window, min_periods=1)
    return getattr(rolled, statistic)().reset_index(level=0, drop=True).sort_index().to_numpy()


def observation_age(mask, reset, fps, cap_s=30):
    out = np.full(len(mask), np.nan)
    last = None
    for i, observed in enumerate(mask):
        if reset[i]: last = None
        if observed: last = i
        if last is not None: out[i] = min((i-last)/fps, cap_s)
    return out


class FeatureBuilder:
    """Five inspectable feature stages. Call spatial, motion, temporal, tagging in order."""
    def __init__(self, players, ball, team_slots, fps=25, options=None):
        self.options = dict(deduplicate_exact=True, max_player_speed=12., max_ball_speed=45.)
        self.options.update(options or {})
        self.players, self.ball = players.copy(), ball.copy()
        self.team_slots, self.fps, self.n = np.asarray(team_slots), fps, len(ball)
        # Partial coordinate pairs are unavailable, not usable single-axis observations.
        self.players[~np.isfinite(self.players).all(2)] = np.nan
        self.ball[~np.isfinite(self.ball).all(1)] = np.nan
        self.duplicate_count = np.zeros(self.n)
        if self.options['deduplicate_exact']:
            # Only machine-identical, same-team coordinates; near players and cross-team duels stay.
            for team in [0,1]:
                slots = np.flatnonzero(self.team_slots == team)
                for j, slot in enumerate(slots):
                    previous = slots[:j]
                    if len(previous):
                        duplicate = (self.players[:,previous] == self.players[:,slot,None,:]).all(2).any(1)
                        self.players[duplicate,slot] = np.nan
                        self.duplicate_count += duplicate
        self.ball_visible = np.isfinite(self.ball).all(1)
        self.player_visible = np.isfinite(self.players).all(2)
        self.player_count = self.player_visible.sum(1)
        active = self.player_count > 0
        last = np.maximum.accumulate(np.where(active, np.arange(self.n), -1))
        previous = np.r_[-1,last[:-1]]
        self.reset = active & ((previous < 0) | (np.arange(self.n)-previous > round(5*fps)))
        self.reset[0] = True
        self.block_id = np.cumsum(self.reset)
        self.features = pd.DataFrame(index=pd.RangeIndex(self.n,name='frame'))
        self.docs = []
        self.player_velocity = np.diff(self.players,axis=0,prepend=self.players[:1])*fps
        self.player_velocity[self.reset] = np.nan
        self.slot_jumps = np.linalg.norm(self.player_velocity,axis=2) > self.options['max_player_speed']
        self.player_velocity[self.slot_jumps] = np.nan
        self.add('ball_x',self.ball[:,0],'spatial','m','Observed ball x; no filling.')
        self.add('ball_y',self.ball[:,1],'spatial','m','Observed ball y; no filling.')
        self.add('ball_visible',self.ball_visible,'quality','binary','Both ball coordinates observed.')
        self.add('ball_tracking_age_s',self.age(self.ball_visible),'quality','s','Time since observed ball; capped at 30 s; unknown before first observation.')
        self.add('visible_player_count',self.player_count,'quality','count','Usable track slots, not stable identities.')
        self.add('slot_jump_count',self.slot_jumps.sum(1),'quality','count','Same-slot speed beyond configured plausible limit.')
        self.add('duplicate_tracks_removed',self.duplicate_count,'quality','count','Machine-identical same-team coordinates removed; close players retained.')

    def add(self,name,values,group,unit,meaning):
        if name in self.features: raise ValueError(f'Duplicate feature {name}')
        if len(self.features.columns) and len(self.features.columns)%64==0:
            self.features=self.features.copy()
        self.features[name] = np.asarray(values,dtype=np.float32)
        self.docs.append(dict(feature=name,group=group,unit=unit,definition=meaning))

    def age(self,mask): return observation_age(mask,self.reset,self.fps)
    def roll(self,values,window,statistic='mean'): return rolling(values,self.block_id,window,statistic)

    def spatial(self):
        distance = np.linalg.norm(self.players-self.ball[:,None,:],axis=2)
        self.nearest_slots = {}
        for side, team in [('home',0),('away',1)]:
            slots = np.flatnonzero(self.team_slots == team)
            dist = np.where(np.isfinite(distance[:,slots]),distance[:,slots],np.inf)
            ordered = np.sort(dist,axis=1)
            visible = self.player_visible[:,slots].sum(1)
            valid = self.ball_visible & (visible > 0)
            for k in range(3):
                value = ordered[:,k].copy(); value[~np.isfinite(value)] = np.nan
                self.add(f'{side}_distance_{k+1}',value,'spatial','m',f'Distance to nearest {side} track rank {k+1}; unavailable is NaN.')
            self.nearest_slots[side] = slots[dist.argmin(1)]
            for radius in [2,5]:
                count = (dist<=radius).sum(1).astype(float); count[~valid] = np.nan
                self.add(f'{side}_within_{radius}m',count,'spatial','count','Observed nearby tracks; NaN when ball or entire team is unobserved.')
            self.add(f'{side}_visible_count',visible,'quality','count','Usable observed tracks for this team.')
            self.add(f'{side}_observation_valid',valid,'quality','binary','Ball and at least one team track are observed.')
            mass = np.exp(-dist/2.).sum(1); mass[~valid] = np.nan
            self.add(f'{side}_control_mass',mass,'spatial','score','Sum exp(-distance/2m); soft proximity evidence, not possession truth.')
        f = self.features
        self.add('proximity_advantage',f.away_distance_1-f.home_distance_1,'spatial','m','Away minus home distance; positive favors home.')
        self.add('control_dominance',(f.home_control_mass-f.away_control_mass)/(f.home_control_mass+f.away_control_mass+1e-8),'spatial','ratio','Normalized soft proximity advantage; missing if either team is unobserved.')
        return self

    def motion(self):
        v = np.diff(self.ball,axis=0,prepend=self.ball[:1])*self.fps; v[self.reset] = np.nan
        speed = np.linalg.norm(v,axis=1)
        bad = speed > self.options['max_ball_speed']; v[bad] = np.nan; speed[bad] = np.nan
        self.ball_speed = speed
        self.add('ball_motion_anomaly',bad,'quality','binary','Observed ball speed exceeds configured plausible limit.')
        self.add('ball_speed',speed,'motion','m/s','Consecutive observed displacement; gaps, resets, anomalies are NaN.')
        acceleration = np.r_[np.nan,np.diff(speed)]*self.fps; acceleration[self.reset] = np.nan
        self.add('ball_acceleration',acceleration,'motion','m/s^2','Signed change in consecutive reliable speeds.')
        prev = np.vstack([np.full((1,2),np.nan),v[:-1]])
        den = np.linalg.norm(prev,axis=1)*speed
        cosine = np.divide((prev*v).sum(1),den,out=np.full(self.n,np.nan),where=den>.01)
        angle = np.arccos(np.clip(cosine,-1,1)); angle[self.reset] = np.nan
        self.add('ball_turn_angle',angle,'motion','radians','Change between reliable nonstationary ball velocities.')
        for side in ['home','away']:
            slots = self.nearest_slots[side]
            velocity = self.player_velocity[np.arange(self.n),slots]
            vector = self.ball-self.players[np.arange(self.n),slots]
            d = self.features[f'{side}_distance_1'].to_numpy()
            player_speed = np.linalg.norm(velocity,axis=1); player_speed[~np.isfinite(d)] = np.nan
            closing = np.divide((velocity*vector).sum(1),d,out=np.full(self.n,np.nan),where=d>.05)
            changed = (slots != np.r_[slots[0],slots[:-1]]).astype(float)
            pair = np.isfinite(d) & np.r_[False,np.isfinite(d[:-1])] & ~self.reset
            changed[~pair] = np.nan
            consistency = np.linalg.norm(v-velocity,axis=1); consistency[~pair | (changed == 1)] = np.nan
            self.add(f'{side}_nearest_speed',player_speed,'motion','m/s','Same-slot velocity of current nearest track; not differences between nearest identities.')
            self.add(f'{side}_closing_speed',closing,'motion','m/s','Projection of same-slot player velocity toward observed ball.')
            self.add(f'{side}_nearest_slot_changed',changed,'quality','binary','Nearest slot changed between consecutive observations.')
            self.add(f'{side}_ball_velocity_difference',consistency,'motion','m/s','Ball minus player velocity norm; unavailable across nearest-slot changes.')
        return self

    def temporal(self):
        f = self.features
        for window in [5,13,25,50,125]:
            for col in ['proximity_advantage','ball_speed','ball_visible','home_distance_1','away_distance_1']:
                self.add(f'{col}_mean_{window}f',self.roll(f[col],window),'temporal','fraction' if col=='ball_visible' else ('m/s' if col=='ball_speed' else 'm'),f'Trailing {window}-frame mean; ignores NaN within observation block.')
            self.add(f'proximity_advantage_std_{window}f',self.roll(f.proximity_advantage,window,'std'),'temporal','m',f'Trailing {window}-frame sample standard deviation.')
        self.strong_home = ((f.home_distance_1<=2)&(f.proximity_advantage>=1)).to_numpy()
        self.strong_away = ((f.away_distance_1<=2)&(f.proximity_advantage<=-1)).to_numpy()
        for side,signal in [('home',self.strong_home),('away',self.strong_away)]:
            self.add(f'since_{side}_control_s',self.age(signal),'temporal','s','Age of strong observed proximity evidence; capped at 30s; not target history.')
        strong = self.strong_home | self.strong_away
        age = self.age(strong)
        last_control = np.full(self.n,np.nan)
        last = np.nan
        persistence = np.zeros(self.n)
        previous, length = 0,0
        for i in range(self.n):
            if self.reset[i]: last,previous,length = np.nan,0,0
            candidate = 1 if self.strong_home[i] else (-1 if self.strong_away[i] else 0)
            length = length+1 if candidate and candidate==previous else int(candidate!=0)
            previous = candidate
            persistence[i] = min(length/self.fps,30.)
            if candidate: last = float(candidate)
            if age[i] <= 5: last_control[i] = last
        self.add('last_strong_team',last_control,'temporal','signed indicator','Last observed strong side (+1 home/-1 away), expires after 5 seconds.')
        self.add('time_since_last_confident_control',age,'temporal','s','Time since either strong proximity observation; no labels.')
        self.add('candidate_control_persistence_s',persistence,'temporal','s','Continuous strong-side evidence duration; not a possession duration.')
        self.add('control_evidence_coverage',self.roll(strong,50),'quality','fraction','Fraction of last two seconds with strong control evidence.')
        return self

    def tagging(self,events,team_names):
        # Fixed semantic columns provide stable features even when a type is absent from a fixture.
        vocabulary = ['PLAYERS IN THE BOX','BALL IN FINAL THIRD','BALL IN THE BOX','PROGRESSION',
            'DEFENDING IN MIDDLE THIRD','SET PIECES','CREATING CHANCES','DEFENDING IN DEFENSIVE THIRD',
            'LONG BALL','ATTACKING TRANSITION','DEFENSIVE TRANSITION','BUILD UP','DEFENDING IN ATTACKING THIRD','GOALS']
        slug = lambda x: re.sub(r'[^a-z0-9]+','_',x.lower()).strip('_')
        arrays = {'tag_code_'+slug(code):np.zeros(self.n) for code in vocabulary}
        for name in ['home_count','away_count','unknown_team_count','active_count','unrecognized_code_count','restart_count']:
            arrays['tag_'+name] = np.zeros(self.n)
        for side in ['home','away']:
            for role in ['attack','defense','restart']:
                arrays[f'tag_{side}_{role}_count'] = np.zeros(self.n)
        restart_types = ['throw_in','corner_kick','goal_kick','free_kick','kick_off_start','kick_off_after_goal']
        for kind in restart_types: arrays['tag_restart_'+kind] = np.zeros(self.n)
        boundary = np.zeros(self.n,dtype=bool)
        for event in events:
            a,b = event['start_frame'],event['end_frame']
            if not 0<=a<b<=self.n: raise ValueError('Tag interval outside feature timeline')
            code = str(event['code']).strip().upper()
            column = 'tag_code_'+slug(code)
            arrays[column if column in arrays else 'tag_unrecognized_code_count'][a:b] += 1
            team = event['tags'].get('Team')
            side = 'home' if team==team_names[0] else ('away' if team==team_names[1] else 'unknown_team')
            arrays['tag_'+side+'_count'][a:b] += 1
            arrays['tag_active_count'][a:b] += 1
            if code == 'SET PIECES':
                arrays['tag_restart_count'][a:b] += 1
                kind = slug(event['tags'].get('Type',''))
                if kind in restart_types: arrays['tag_restart_'+kind][a:b] += 1
                role = 'restart'
            elif code.startswith('DEFEND') or code=='DEFENSIVE TRANSITION': role='defense'
            elif code in {'BUILD UP','PROGRESSION','ATTACKING TRANSITION','CREATING CHANCES','LONG BALL'}: role='attack'
            else: role=None
            if side!='unknown_team' and role: arrays[f'tag_{side}_{role}_count'][a:b] += 1
            boundary[a]=True
            if b<self.n: boundary[b]=True
        for name,value in arrays.items(): self.add(name,value,'tagging','count','Active noisy event intervals [start,end): '+name.removeprefix('tag_'))
        self.add('tag_team_conflict',(arrays['tag_home_count']>0)&(arrays['tag_away_count']>0),'tagging','binary','Team overlap only; attack-home and defense-away may agree, not conflict.')
        self.add('tag_role_home_evidence',arrays['tag_home_attack_count']+arrays['tag_away_defense_count'],'tagging','count','Complementary home attack and away defense evidence.')
        self.add('tag_role_away_evidence',arrays['tag_away_attack_count']+arrays['tag_home_defense_count'],'tagging','count','Complementary away attack and home defense evidence.')
        self.add('tag_since_boundary_s',self.age(boundary),'tagging','s','Age since any observed tag boundary; capped at 30 seconds.')
        # No missing ball -> D implication. Stationarity alone can also be a live ball.
        stationary = self.ball_visible & np.isfinite(self.ball_speed) & (self.ball_speed<.4)
        fraction = self.roll(stationary,50)
        self.add('stationary_ball_fraction_2s',fraction,'temporal','fraction','Observed near-stationary ball evidence in a trailing two-second window.')
        self.add('tag_restart_stationary_evidence',fraction*(arrays['tag_restart_count']>0),'tagging','score','Stationarity with noisy restart context; not a copied dead-ball label.')
        # A one-frame derivative amplifies position noise at 25 FPS. Use a trailing
        # observed spatial range as a separate stillness signal; never fill long gaps.
        range_x=self.roll(self.ball[:,0],self.fps,'max')-self.roll(self.ball[:,0],self.fps,'min')
        range_y=self.roll(self.ball[:,1],self.fps,'max')-self.roll(self.ball[:,1],self.fps,'min')
        coverage=self.roll(self.ball_visible,self.fps)
        span=np.hypot(range_x,range_y)
        self.add('ball_span_1s_m',span,'motion','m','Bounding-box diagonal of observed ball positions in the last second; no interpolation.')
        self.add('ball_coverage_1s',coverage,'quality','fraction','Observed ball fraction in the last second.')
        stillness=np.exp(-span/.75)*coverage
        stillness[~self.ball_visible | (coverage<.6)]=0.
        stillness=np.nan_to_num(stillness,nan=0.)
        self.add('tag_restart_stillness_score',stillness*(arrays['tag_restart_count']>0),'tagging','score',
                 'Noise-tolerant observed stillness with restart context; threshold fitted only on training targets.')
        self.boundary = boundary
        # Consolidate storage after staged additions.
        self.features = self.features.copy()
        return self


def build_features(players,ball,team_slots,events,team_names,fps=25,options=None):
    return FeatureBuilder(players,ball,team_slots,fps,options).spatial().motion().temporal().tagging(events,team_names)
