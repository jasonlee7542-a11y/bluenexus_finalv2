"""Statistical screening, with explicit unscored states. No physical limits."""
import numpy as np
import pandas as pd

def screen(series, days=14, multiplier=1.5, mode='Prior rolling window', reference_days=31):
    s=series.sort_index()
    if not isinstance(s.index,pd.DatetimeIndex) or s.index.has_duplicates:
        raise ValueError('Unique datetime index required')
    # Missing calendar dates stay missing; baseline windows are truly days.
    s=s.reindex(pd.date_range(s.index.min(),s.index.max(),freq='D'))
    min_obs=max(3,int(np.ceil(days*.8)))
    if mode=='Prior rolling window':
        window=s.shift(1).rolling(days,min_periods=min_obs)
        q1=window.quantile(.25);q3=window.quantile(.75);med=window.median();count=window.count()
        eligible=pd.Series(True,index=s.index)
    else:
        end=s.index.min()+pd.Timedelta(days=reference_days)
        ref=s.loc[s.index<end].dropna();enough=len(ref)>=max(3,int(np.ceil(reference_days*.8)))
        q1=pd.Series(ref.quantile(.25) if enough else np.nan,index=s.index)
        q3=pd.Series(ref.quantile(.75) if enough else np.nan,index=s.index)
        med=pd.Series(ref.median() if enough else np.nan,index=s.index)
        count=pd.Series(len(ref),index=s.index);eligible=pd.Series(s.index>=end,index=s.index)
    iqr=q3-q1
    tolerance=np.maximum(1.,med.abs())*1e-12
    spread_ok=iqr.gt(tolerance)
    valid=s.notna()&q1.notna()&eligible&spread_ok
    low=q1-multiplier*iqr;high=q3+multiplier*iqr
    flag=valid&(s.lt(low)|s.gt(high))
    status=pd.Series('Within statistical band',index=s.index)
    status.loc[flag]='Statistical deviation'
    status.loc[~eligible]='Reference period'
    status.loc[eligible&q1.isna()]='Insufficient history'
    status.loc[eligible&q1.notna()&~spread_ok]='Flat reference: no IQR decision'
    status.loc[s.isna()]='Missing reading'
    run=flag.groupby((~flag).cumsum()).cumsum().astype(int)
    return pd.DataFrame({'date':s.index,'value':s.values,'reference_median':med.values,
        'lower_fence':low.where(valid).values,'upper_fence':high.where(valid).values,
        'baseline_observations':count.values,'status':status.values,'flag':flag.values,
        'consecutive_flagged_days':run.values})
