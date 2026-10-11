from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/20260909-extended-sensitivities'
OUT=ROOT/'revision_v28_audit'; OUT.mkdir(exist_ok=True)
ANNUAL=RUN/'stage09_current_full_history/annual_matched_effects.parquet'
MONTH=RUN/'stage09_current_full_history/monthly_matched_effects.parquet'
SUP=RUN/'analysis_current_full_history/common_support_weights.parquet'
PAIRS=RUN/'analysis_current_full_history/fixed_matched_pairs.parquet'

def first_event_meta():
    pairs=pd.read_parquet(PAIRS,columns=['treatment_grid_id','event_year','event_type','censor_relative_year'])
    return pairs.drop_duplicates(['treatment_grid_id','event_year']).copy()

def load_base():
    ann=pd.read_parquet(ANNUAL)
    sup=pd.read_parquet(SUP)
    c=first_event_meta()
    # verify event keys and merge frozen standardization and censoring
    sup=sup[['treatment_grid_id','event_year','event_type','macroregion','event_area_m2','disturbance_share','event_composition_purity','baseline_undist_share','standardized_event_weight','standardization_multiplier']]
    ann=ann.merge(sup,on=['treatment_grid_id','event_year','event_type'],how='inner',validate='many_to_one',suffixes=('','_sup'))
    ann=ann.merge(c,on=['treatment_grid_id','event_year','event_type'],how='left',validate='many_to_one')
    assert ann['standardized_event_weight'].notna().all()
    return ann,sup,c

def weighted_stats(df,weight='standardized_event_weight'):
    out=[]
    for r,dr in df.groupby('relative_year'):
      for et,d in dr.groupby('event_type'):
        w=d[weight].to_numpy(float); y=d['annual_matched_effect'].to_numpy(float)
        ok=np.isfinite(y)&np.isfinite(w)&(w>0); d=d.loc[ok]; w=w[ok]; y=d['annual_matched_effect'].to_numpy(float)
        if len(d)==0: continue
        mu=np.sum(w*y)/np.sum(w); ess=np.sum(w)**2/np.sum(w*w)
        vals={'relative_year':int(r),'event_type':et,'n':len(d),'estimate':mu,'ess':ess}
        for col in ['disturbance_share','event_composition_purity','baseline_undist_share']:
          x=d[col].to_numpy(float); m=np.sum(w*x)/np.sum(w); sd=np.sqrt(np.sum(w*(x-m)**2)/np.sum(w)); vals[col+'_mean']=m; vals[col+'_sd']=sd
        vals['event_year_mean']=np.sum(w*d.event_year.to_numpy(float))/np.sum(w)
        vals['event_year_sd']=np.sqrt(np.sum(w*(d.event_year.to_numpy(float)-vals['event_year_mean'])**2)/np.sum(w))
        vals['weight_sum']=np.sum(w)
        out.append(vals)
    return pd.DataFrame(out)

def contrast(stats):
  piv=stats.pivot(index='relative_year',columns='event_type',values='estimate')
  return pd.DataFrame({'relative_year':piv.index,'direct_contrast':piv.get('deforestation')-piv.get('degradation')}).reset_index(drop=True)

def month_thresholds():
    m=pd.read_parquet(MONTH)
    key=['treatment_grid_id','event_year','event_type','macroregion','event_area_m2','treatment_lon','treatment_lat','relative_year']
    # finite matched effect is the current monthly usable effect. calendar_month is unique by event-year-month.
    m=m[np.isfinite(m.matched_effect)].copy()
    monthly=m.groupby(key,as_index=False).agg(usable_months=('calendar_month','nunique'),annual_matched_effect=('matched_effect','mean'))
    annmeta,sup,c=load_base()
    monthly=monthly.merge(sup[['treatment_grid_id','event_year','event_type','standardized_event_weight','disturbance_share','event_composition_purity','baseline_undist_share']],on=['treatment_grid_id','event_year','event_type'],how='inner',validate='many_to_one')
    monthly=monthly.merge(c,on=['treatment_grid_id','event_year','event_type'],how='left',validate='many_to_one')
    rows=[]
    for threshold in [6,8,9,10,12]:
      x=monthly[monthly.usable_months>=threshold].copy()
      x=x[x.censor_relative_year.isna() | (x.relative_year < x.censor_relative_year)]
      x=x[np.isfinite(x.annual_matched_effect)&np.isfinite(x.standardized_event_weight)&(x.standardized_event_weight>0)]
      st=weighted_stats(x)
      st['month_threshold']=threshold
      st['direct_contrast']=st.groupby('relative_year',group_keys=False).apply(lambda z: z.set_index('event_type')['estimate'].get('deforestation',np.nan)-z.set_index('event_type')['estimate'].get('degradation',np.nan)) if False else np.nan
      for r,g in x.groupby('relative_year'):
        q=g.groupby('event_type').apply(lambda d: np.sum(d.standardized_event_weight*d.annual_matched_effect)/np.sum(d.standardized_event_weight))
        rows.append({'month_threshold':threshold,'relative_year':int(r),'degradation_n':int((g.event_type=='degradation').sum()),'deforestation_n':int((g.event_type=='deforestation').sum()),'degradation_estimate':q.get('degradation',np.nan),'deforestation_estimate':q.get('deforestation',np.nan),'direct_contrast':q.get('deforestation',np.nan)-q.get('degradation',np.nan),'degradation_ess':(g.loc[g.event_type=='degradation','standardized_event_weight'].sum()**2/(g.loc[g.event_type=='degradation','standardized_event_weight']**2).sum()),'deforestation_ess':(g.loc[g.event_type=='deforestation','standardized_event_weight'].sum()**2/(g.loc[g.event_type=='deforestation','standardized_event_weight']**2).sum())})
    out=pd.DataFrame(rows);out.to_csv(OUT/'month_threshold_sensitivity.csv',index=False)
    # save threshold input event-year table for reproducibility
    return out,monthly

def year_specific():
    ann, sup, c=load_base()
    ann=ann[np.isfinite(ann.annual_matched_effect)&np.isfinite(ann.standardized_event_weight)&(ann.standardized_event_weight>0)]
    ann=ann[ann.censor_relative_year.isna() | (ann.relative_year < ann.censor_relative_year)].copy()
    # each relative year: observable sample first; qcut on pooled current-year events, duplicates drop if needed
    rows=[]; bal=[]; event_tables=[]
    for r,g in ann.groupby('relative_year'):
      g=g.copy()
      for col in ['disturbance_share','event_composition_purity','baseline_undist_share']:
        try:g[col+'_bin']=pd.qcut(g[col],5,labels=False,duplicates='drop')
        except ValueError:g[col+'_bin']=0
      g['yr_stratum']=g.macroregion.astype(str)+'|year='+g.event_year.astype(str)+'|i='+g.disturbance_share_bin.astype(str)+'|p='+g.event_composition_purity_bin.astype(str)+'|f='+g.baseline_undist_share_bin.astype(str)
      counts=g.groupby(['yr_stratum','event_type']).size().unstack(fill_value=0)
      keep=counts.index[(counts.get('degradation',0)>0)&(counts.get('deforestation',0)>0)]
      g=g[g.yr_stratum.isin(keep)].copy()
      # raw event area shares within stratum; equal target of two type shares
      g['raw_area']=g.event_area_m2
      sh=g.groupby(['yr_stratum','event_type']).raw_area.sum().unstack(fill_value=0)
      sh=sh.loc[sh.index.intersection(keep)]
      s=sh.div(sh.sum(axis=0),axis=1)
      q=(s.get('degradation',0)+s.get('deforestation',0))/2
      mult=pd.concat([(q/s['degradation']).rename('degradation'),(q/s['deforestation']).rename('deforestation')]).rename('mult').reset_index()
      # easier map
      mapd={(idx,'degradation'):qv/s.loc[idx,'degradation'] for idx,qv in q.items()}
      mapd.update({(idx,'deforestation'):qv/s.loc[idx,'deforestation'] for idx,qv in q.items()})
      g['year_multiplier']=[mapd[(a,b)] for a,b in zip(g.yr_stratum,g.event_type)]
      g['year_weight']=g.event_area_m2*g.year_multiplier
      event_tables.append(g.assign(relative_year=int(r))[['treatment_grid_id','event_year','event_type','macroregion','relative_year','annual_matched_effect','year_weight','disturbance_share','event_composition_purity','baseline_undist_share','treatment_lon','treatment_lat']])
      for et,d in g.groupby('event_type'):
        w=d.year_weight.to_numpy(float); y=d.annual_matched_effect.to_numpy(float);mu=np.sum(w*y)/np.sum(w);ess=np.sum(w)**2/np.sum(w*w)
        rows.append({'relative_year':int(r),'event_type':et,'n':len(d),'ess':ess,'estimate':mu,'multiplier_min':g.year_multiplier.min(),'multiplier_p99':np.quantile(g.year_multiplier,.99),'multiplier_max':g.year_multiplier.max(),'support_strata':len(keep)})
      for et,d in g.groupby('event_type'):
        w=d.year_weight.to_numpy(float); vals={'relative_year':int(r),'event_type':et,'n':len(d),'weighted_event_year_mean':np.sum(w*d.event_year)/np.sum(w),'weighted_event_year_sd':np.sqrt(np.sum(w*(d.event_year-np.sum(w*d.event_year)/np.sum(w))**2)/np.sum(w))}
        for col in ['disturbance_share','event_composition_purity','baseline_undist_share']:
          vals[col+'_mean']=np.sum(w*d[col])/np.sum(w)
        vals['weight_sum']=np.sum(w)
        vals['region_fractions']=';'.join(f'{k}={np.sum(w[d.macroregion==k])/np.sum(w):.6f}' for k in sorted(d.macroregion.unique()))
        bal.append(vals)
      # Store exact weighted standardized mean differences for the year-specific risk set.
      for col in ['disturbance_share','event_composition_purity','baseline_undist_share','event_year']:
        moments={}
        for et,d in g.groupby('event_type'):
          w=d.year_weight.to_numpy(float); x=d[col].to_numpy(float); mu=np.sum(w*x)/np.sum(w); var=np.sum(w*(x-mu)**2)/np.sum(w); moments[et]=(mu,var)
        if set(moments)>= {'degradation','deforestation'}:
          den=np.sqrt((moments['degradation'][1]+moments['deforestation'][1])/2)
          bal[-1][col+'_smd']=(moments['deforestation'][0]-moments['degradation'][0])/den if den else np.nan
          bal[-2][col+'_smd']=bal[-1][col+'_smd']
      qest=g.groupby('event_type').apply(lambda d: np.sum(d.year_weight*d.annual_matched_effect)/np.sum(d.year_weight))
      rows[-1]['direct_contrast']=qest.get('deforestation',np.nan)-qest.get('degradation',np.nan)
      if len(rows)>=2: rows[-2]['direct_contrast']=rows[-1]['direct_contrast']
    out=pd.DataFrame(rows);out.to_csv(OUT/'year_specific_standardization.csv',index=False)
    bdf=pd.DataFrame(bal)
    for r,g in bdf.groupby('relative_year'):
      if set(g.event_type)>= {'degradation','deforestation'}:
        a=g.set_index('event_type')
        for v in ['disturbance_share','event_composition_purity','baseline_undist_share','event_year']:
          den=np.sqrt((a.loc['degradation',v+'_mean' if v!='event_year' else 'weighted_event_year_sd']*0+1)) if False else np.nan
        # SMD based on raw event tables below; placeholder filled by exact weighted moments.
    bdf.to_csv(OUT/'year_specific_balance.csv',index=False)
    pd.concat(event_tables,ignore_index=True).to_parquet(OUT/'year_specific_event_table.parquet',index=False)
    return out

def event_year_balance():
    ann, sup,c=load_base();ann=ann[np.isfinite(ann.annual_matched_effect)&np.isfinite(ann.standardized_event_weight)&(ann.standardized_event_weight>0)];ann=ann[ann.censor_relative_year.isna() | (ann.relative_year < ann.censor_relative_year)]
    rows=[]
    for r,g in ann.groupby('relative_year'):
      for et,d in g.groupby('event_type'):
        w=d.standardized_event_weight.to_numpy(float); mu=np.sum(w*d.event_year)/np.sum(w);sd=np.sqrt(np.sum(w*(d.event_year-mu)**2)/np.sum(w));rows.append({'relative_year':int(r),'event_type':et,'n':len(d),'weighted_event_year_mean':mu,'weighted_event_year_sd':sd,'event_year_min':int(d.event_year.min()),'event_year_max':int(d.event_year.max()),'weighted_shares':';'.join(f'{int(y)}={np.sum(w[d.event_year==y])/np.sum(w):.8f}' for y in sorted(d.event_year.unique()))})
    out=pd.DataFrame(rows);p=out.pivot(index='relative_year',columns='event_type',values='weighted_event_year_mean');smd=[]
    for r in p.index:
      d=out[(out.relative_year==r)&(out.event_type=='deforestation')].iloc[0];g=out[(out.relative_year==r)&(out.event_type=='degradation')].iloc[0];# pooled sd unavailable; direct weighted sd approximation
      den=np.sqrt((d.weighted_event_year_sd**2+g.weighted_event_year_sd**2)/2);smd.append({'relative_year':int(r),'event_year_smd_approx':(d.weighted_event_year_mean-g.weighted_event_year_mean)/den if den else np.nan})
    out=out.merge(pd.DataFrame(smd),on='relative_year');out.to_csv(OUT/'event_year_balance.csv',index=False)
    return out

if __name__=='__main__':
  th,_=month_thresholds();ys=year_specific();eb=event_year_balance();print(json.dumps({'threshold_rows':len(th),'year_specific_rows':len(ys),'event_year_rows':len(eb)},ensure_ascii=False))
