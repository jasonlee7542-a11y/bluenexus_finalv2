"""Plant A exploratory operating review. Python 3.10+, pandas, numpy, matplotlib.
Run: python plant_a_review.py --folder "D:\\Documents\\Hack & Hire"
No API key, AI service, or plant-control connection is used.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EVENT_COLS = ['date', 'category', 'signal', 'observation', 'possible_explanation', 'next_check', 'action_route']

def add_event(events, date, category, signal, observation, explanation, check):
    events.append(dict(zip(EVENT_COLS, [pd.Timestamp(date).strftime('%Y-%m-%d'), category, signal,
        observation, explanation, check, 'Remote engineer/data review; on-site need undetermined'])))

def load_table(path, source, logs):
    """Preserve raw CSV externally; log coercions, remove only exact duplicate records.
    Ambiguous duplicate daily keys stop the run rather than silently choosing a record.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'date' not in raw:
        raise ValueError(f'{source}: missing date column')
    df = raw.copy()
    for c in df:
        s = df[c].str.strip()
        missing = s.str.upper().isin(['', 'NULL', 'NAN', 'NA', 'N/A'])
        df[c] = s.mask(missing)
    # Only unambiguous year-first date strings are accepted.
    dates = df['date'].str.replace('/', '-', regex=False)
    valid_format = dates.str.match(r'^\d{4}-\d{1,2}-\d{1,2}$', na=False)
    parsed = pd.to_datetime(dates.where(valid_format), format='%Y-%m-%d', errors='coerce')
    if parsed.isna().any():
        raise ValueError(f'{source}: invalid/ambiguous dates at CSV rows {(np.flatnonzero(parsed.isna()) + 2).tolist()}')
    df['date'] = parsed
    for c in df.columns.difference(['date', 'plant']):
        n = pd.to_numeric(df[c], errors='coerce')
        invalid = df[c].notna() & (n.isna() | ~np.isfinite(n))
        for i in df.index[invalid]:
            logs.append({'source':source,'date':df.loc[i,'date'].strftime('%Y-%m-%d'),
                         'column':c,'action':'invalid numeric -> missing','original':df.loc[i,c]})
        df[c] = n.where(np.isfinite(n))
    duplicate_count = int(df.duplicated().sum())
    logs.append({'source':source,'column':'ALL','action':'exact duplicate rows removed',
                 'original':duplicate_count,'date':''})
    df = df.drop_duplicates()
    keys = ['plant','date'] if 'plant' in df else ['date']
    if df.duplicated(keys).any():
        raise ValueError(f'{source}: conflicting duplicate keys; review {df.loc[df.duplicated(keys,keep=False),keys].to_dict("records")}')
    return df.sort_values(keys).reset_index(drop=True), len(raw)

def audit_table(df, source):
    records=[]
    for c in df.columns.difference(['date','plant']):
        s=df[c]
        status='all missing' if s.notna().sum()==0 else 'constant observed value' if s.nunique()==1 else 'varying'
        records.append({'source':source,'column':c,'rows':len(s),'missing':int(s.isna().sum()),
            'missing_pct':round(s.isna().mean()*100,2),'zeros':int(s.eq(0).sum()),
            'unique_observed':s.nunique(),'min':s.min(),'median':s.median() if s.notna().any() else np.nan,'max':s.max(),'status':status})
    return pd.DataFrame(records)

def equipment_review(df, events):
    df=df.copy().set_index('date').sort_index()
    for c in list(df.select_dtypes('number').columns):
        s=df[c]
        if 'cumulative' in c or c.endswith('_ep'):
            decrease=s.diff().lt(0)
            df[c+'__counter_decrease']=decrease
            for date in df.index[decrease]:
                add_event(events,date,'Data quality',c,f'Counter-like value decreased to {s.loc[date]:.4g}.',
                    'Reset, rollover, invalid zero, or recording change; field meaning unconfirmed.',
                    'Verify counter definition, scaling and meter history. Do not derive consumption across this interval.')
    for k in (1,2):
        f=f'{k}_rohigh_pressure_pump_frequency_feedback_o'; p=f'zz_{k}_rohigh_pressure_pump_power'
        if f in df and p in df:
            flag=df[f].eq(0)&df[p].gt(0)
            df[f'pump_{k}__zero_frequency_nonzero_power']=flag
            for date in df.index[flag]:
                add_event(events,date,'Signal consistency',f'Pump {k}',
                    f'Frequency = 0; power-like reading = {df.loc[date,p]:.4g} (unit unconfirmed).',
                    'Aggregation mismatch, stale reading, different tag mapping or genuine operating condition.',
                    'Confirm both tags refer to the same asset and sampling period before inferring shutdown or wasted energy.')
        if p in df:
            # Calendar-time rolling mean, with minimum coverage visible in output.
            df[f'pump_{k}_power_7d_count']=df[p].rolling('7D').count()
            df[f'pump_{k}_power_7d_mean']=df[p].rolling('7D',min_periods=4).mean()
    return df.reset_index()

def performance_review(perf, events, config):
    p=perf.copy().sort_values('date').reset_index(drop=True)
    required=['daily_electricity_kwh','effluent_volume_m3','influent_volume_m3']
    if not set(required).issubset(p.columns):
        raise ValueError('Missing required performance columns: '+str(set(required)-set(p.columns)))
    energy=p.daily_electricity_kwh; vol=p.effluent_volume_m3; feed=p.influent_volume_m3
    p['calculated_sec_kwh_m3']=(energy/vol).where(energy.ge(0)&vol.gt(0))
    p['calculated_apparent_recovery']=(vol/feed).where(vol.ge(0)&feed.gt(0))
    if {'influent_conductivity_us_cm','effluent_conductivity_us_cm'}.issubset(p):
        ci=p.influent_conductivity_us_cm; co=p.effluent_conductivity_us_cm
        p['calculated_conductivity_reduction']=(1-co/ci).where(ci.gt(0)&co.ge(0))
    for c in required:
        bad=p[c].lt(0)
        for _,row in p.loc[bad].iterrows():
            add_event(events,row.date,'Data quality',c,f'Negative daily quantity: {row[c]}.',
                'Reporting correction or invalid measurement.', 'Confirm the reporting convention; affected ratios are unavailable.')
    for _,row in p.loc[vol.le(0)|feed.le(0)].iterrows():
        add_event(events,row.date,'Coverage / operating context','Water volumes','Nonpositive product or feed volume.',
            'Shutdown, missing-as-zero or a reporting issue.', 'Review operating state; undefined ratios are left missing.')
    if 'reported_specific_energy_kwh_m3' in p:
        p['sec_report_difference']=p.calculated_sec_kwh_m3-p.reported_specific_energy_kwh_m3
        for _,row in p.loc[p.sec_report_difference.abs().gt(config['sec_agreement_tolerance'])].iterrows():
            add_event(events,row.date,'Data quality','Specific energy',f'Recalculated minus reported SEC = {row.sec_report_difference:.4f} kWh/m3.',
                'Rounding, different denominator or reporting mismatch.', 'Verify meter boundaries and the reported SEC formula.')
    for _,row in p.loc[p.calculated_apparent_recovery.gt(1)].iterrows():
        add_event(events,row.date,'Reporting context','Apparent recovery','Daily product/feed ratio exceeds 100%.',
            'Tank inventory changes, reporting cutoffs or measurement problem.', 'Check storage changes and aligned feed/product reporting periods.')
    # Chronological, fixed baseline. Exploratory screening, not validated diagnosis.
    n=config['reference_rows']; p['period']=np.where(np.arange(len(p))<n,'Reference','Evaluation')
    ref=p.iloc[:n]; mask=ref.daily_electricity_kwh.ge(0)&ref.effluent_volume_m3.gt(0)
    train=ref.loc[mask].dropna(subset=required[:2]); x=train.effluent_volume_m3.to_numpy(); y=train.daily_electricity_kwh.to_numpy()
    model={'status':'unavailable','reason':'Need >=20 valid reference rows, later data and varying production.'}
    if len(train)>=20 and len(p)>n and np.unique(x).size>=5 and np.ptp(x)>0:
        slope,intercept=np.polyfit(x,y,1); residual=y-(intercept+slope*x)
        center=float(np.median(residual)); scale=float(1.4826*np.median(np.abs(residual-center)))
        if scale<1e-9: scale=float(np.std(residual,ddof=2))
        threshold=max(config['residual_scale_multiplier']*scale,config['energy_floor_fraction']*float(np.median(y)))
        model={'status':'fitted','reference_start':str(ref.date.min().date()),'reference_end':str(ref.date.max().date()),
            'reference_valid_rows':len(train),'intercept_kwh':float(intercept),'slope_kwh_per_m3':float(slope),
            'reference_r2':float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)) if np.ptp(y)>0 else None,
            'residual_median_kwh':center,'residual_scale_kwh':scale,'screening_margin_kwh':threshold,
            'note':'Exploratory OLS association; reference operation is not certified healthy. Residual is not avoidable energy.'}
        p['expected_electricity_kwh']=intercept+slope*vol
        p['electricity_residual_kwh']=energy-p.expected_electricity_kwh
        p['within_reference_production']=vol.between(x.min(),x.max())
        valid=energy.ge(0)&vol.gt(0)&p.expected_electricity_kwh.gt(0)
        p['high_energy_deviation']=(p.period.eq('Evaluation')&p.within_reference_production&valid&
            p.electricity_residual_kwh.gt(center+threshold))
        # Require consecutive calendar days, not merely adjacent rows.
        consecutive=p.date.diff().dt.days.eq(1)
        p['persistent_high_energy']=p.high_energy_deviation & p.high_energy_deviation.shift(1,fill_value=False)&consecutive
        for _,row in p.loc[p.high_energy_deviation].iterrows():
            add_event(events,row.date,'Performance screening','Plant electricity',
                f'Electricity {row.daily_electricity_kwh:.0f} kWh; baseline {row.expected_electricity_kwh:.0f}; residual {row.electricity_residual_kwh:.0f} kWh.'+
                (' Second or later consecutive flagged day.' if row.persistent_high_energy else ''),
                'Operating schedule, feed conditions, cleaning, measurement issue or possible inefficiency.',
                'Review production, product conductivity, logs and validated equipment signals. This is not confirmed waste or failure.')
        for _,row in p.loc[p.period.eq('Evaluation')&~p.within_reference_production&vol.notna()].iterrows():
            add_event(events,row.date,'Baseline coverage','Product volume','Production lies outside reference range; energy alert withheld.',
                'Different operating conditions; baseline would extrapolate.', 'Review separately or obtain representative reference data.')
    return p,model

def make_charts(equip,p,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    paths=[]
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    items=[('effluent_volume_m3','Product volume (m3/day)'),('calculated_sec_kwh_m3','Specific energy (kWh/m3)'),
           ('calculated_apparent_recovery','Apparent recovery (fraction)'),('effluent_conductivity_us_cm','Product conductivity (uS/cm)')]
    for ax,(c,label) in zip(axes,items):
        if c in p: ax.plot(p.date,p[c],color='#176b87',marker='.',ms=3)
        ax.set_ylabel(label);ax.grid(alpha=.2)
        if c=='calculated_apparent_recovery':
            ax.set_ylim(0,max(1.05,float(p[c].max())*1.05) if p[c].notna().any() else 1.05)
        if 'high_energy_deviation' in p:
            for date in p.loc[p.high_energy_deviation,'date']: ax.axvline(date,color='#d35d38',alpha=.3)
    axes[0].set_title('Plant A performance | orange lines: exploratory high-energy flags')
    fig.autofmt_xdate();fig.tight_layout();path=out/'performance.png';fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for k,ax in enumerate(axes,1):
        c=f'zz_{k}_rohigh_pressure_pump_power'
        if c in equip:
            ax.plot(equip.date,equip[c],alpha=.4,label='Observed')
            ax.plot(equip.date,equip[f'pump_{k}_power_7d_mean'],label='Trailing 7 calendar days (>=4 readings)')
        ax.set_ylabel(f'Pump {k} power-like signal\n(unit unconfirmed)');ax.legend();ax.grid(alpha=.2)
    axes[0].set_title('Full equipment history | smoothing does not diagnose fouling')
    fig.autofmt_xdate();fig.tight_layout();path=out/'equipment.png';fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for period,color in [('Reference','#176b87'),('Evaluation','#d35d38')]:
        q=p.loc[p.period.eq(period)];axes[0].scatter(q.effluent_volume_m3,q.daily_electricity_kwh,label=period,color=color,s=22)
    if 'expected_electricity_kwh' in p:
        q=p.loc[p.period.eq('Reference')].sort_values('effluent_volume_m3')
        axes[0].plot(q.effluent_volume_m3,q.expected_electricity_kwh,color='black',label='Reference fit')
    axes[0].set(xlabel='Product volume (m3/day)',ylabel='Plant electricity (kWh/day)');axes[0].legend()
    axes[1].hist(p.calculated_sec_kwh_m3.dropna(),bins=12,color='#176b87',edgecolor='white')
    axes[1].set(xlabel='Specific energy (kWh/m3)',ylabel='Days',title='Observed distribution, not an optimal range')
    fig.tight_layout();path=out/'relationships.png';fig.savefig(path,dpi=140);plt.close(fig);paths.append(path)
    return paths

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,help='Optional JSON overriding documented exploratory settings')
    args=parser.parse_args();out=args.output or args.folder/'plant_A_review_output';out.mkdir(parents=True,exist_ok=True)
    config={'reference_rows':31,'residual_scale_multiplier':3.0,'energy_floor_fraction':0.05,'sec_agreement_tolerance':0.01}
    if args.config:
        overrides=json.loads(args.config.read_text());unknown=set(overrides)-set(config)
        if unknown: raise ValueError(f'Unknown settings: {unknown}')
        config.update(overrides)
    if not isinstance(config['reference_rows'],int) or config['reference_rows']<20: raise ValueError('reference_rows must be an integer >=20')
    if any(not np.isfinite(config[k]) or config[k]<=0 for k in config if k!='reference_rows'): raise ValueError('Threshold settings must be positive finite numbers')
    paths=[args.folder/'plant_A_equipment_daily.csv',args.folder/'plant_daily_performance.csv']
    if any(p.resolve().parent==out.resolve() for p in paths): raise ValueError('Use a separate output directory')
    logs=[];events=[]
    equip,erows=load_table(paths[0],'Equipment',logs);allperf,prows=load_table(paths[1],'Performance',logs)
    if 'plant' not in allperf: raise ValueError('Performance file requires plant column')
    perf=allperf.loc[allperf.plant.eq('Plant A')].copy()
    if perf.empty: raise ValueError('No Plant A performance records')
    audits=pd.concat([audit_table(equip,'Equipment'),audit_table(perf,'Plant A performance')],ignore_index=True)
    for label,table in [('Equipment',equip),('Performance',perf)]:
        for date in pd.date_range(table.date.min(),table.date.max()).difference(table.date):
            add_event(events,date,'Missing date',label,'No daily record.', 'Unreported or unavailable observation.', 'Check source coverage; no row has been fabricated.')
    e=equipment_review(equip,events);p,model=performance_review(perf,events,config)
    # Full timeline retained; performance columns remain missing outside their coverage.
    combined=e.merge(p,on='date',how='outer',validate='one_to_one',indicator=True).sort_values('date')
    ev=pd.DataFrame(events,columns=EVENT_COLS).sort_values(['date','category'])
    for frame,name in [(e,'equipment_prepared.csv'),(p,'performance_analysis.csv'),(combined,'joined_review.csv'),
                       (audits,'column_audit.csv'),(ev,'flagged_events.csv'),(pd.DataFrame(logs),'cleaning_log.csv')]:
        frame.to_csv(out/name,index=False,date_format='%Y-%m-%d',na_rep='')
    summary={'equipment_raw_rows':erows,'equipment_prepared_rows':len(e),'performance_raw_rows_all_plants':prows,
        'plant_a_performance_rows':len(p),'overlap_rows':int(combined._merge.eq('both').sum()),
        'event_counts':ev.category.value_counts().to_dict(),'flagged_unique_dates':ev.date.nunique(),
        'configuration':config,'model':model,
        'inputs_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    charts=make_charts(e,p,out)
    content='''<!doctype html><html><head><meta charset="utf-8"><title>Plant A operating review</title>
    <style>body{font:15px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#203340}h1,h2{color:#176b87}table{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #dbe4e8;text-align:left}th{background:#edf4f6}img{width:100%}.note{padding:18px;background:#edf4f6}pre{white-space:pre-wrap}</style></head><body>
    <h1>Plant A operating review</h1><p>Exploratory daily-data screening • Reproducible calculations • No confirmed failure labels</p>
    <div class="note">Read flags as investigation priorities. Reference operation is not certified healthy or optimal.
    Equipment units and daily aggregation are unconfirmed. No fault diagnosis, automatic control, maintenance dispatch,
    savings claim or AquaxGo integration is performed. Missing values and questionable readings are retained.</div>'''
    content+=f'<h2>Coverage and findings</h2><p>{len(e)} equipment days; {len(p)} Plant A performance days; {summary["overlap_rows"]} matching dates. {len(ev)} event entries across {ev.date.nunique()} dates. Multiple entries can refer to one date; data-quality flags are not inefficiency events.</p>'
    content+='<h2>Event types</h2>'+ev.category.value_counts().rename_axis('Category').reset_index(name='Entries').to_html(index=False)
    content+='<h2>Method</h2><p>Exact duplicate records removed; conflicting daily keys stop processing. Dates parsed and sorted. Numeric coercions logged. All source columns retained; no interpolation, arbitrary temperature cap or cumulative-energy differencing. Calendar-time rolling means are descriptive only.</p>'
    content+=f'<p>Electricity baseline: first {config["reference_rows"]} performance records, electricity = intercept + slope × product volume. Later days are screened only within the reference production range. A high deviation exceeds reference median residual plus max({config["residual_scale_multiplier"]} × robust residual scale, {config["energy_floor_fraction"]:.0%} of reference median electricity). This configurable heuristic is not a calibrated failure probability. Two consecutive calendar-day flags receive a persistence note. Fit quality is reported below; omitted operating conditions can explain residuals.</p>'
    content+='<h2>Baseline details</h2><p>The fit is descriptive. A low R-squared means production alone explains little electricity variation; lack of alerts is not evidence of efficient operation. No threshold was calibrated against confirmed faults.</p><pre>'+html.escape(json.dumps(model,indent=2))+'</pre>'
    for chart in charts: content+='<img alt="'+chart.stem+'" src="data:image/png;base64,'+base64.b64encode(chart.read_bytes()).decode()+'">'
    content+='<h2>Flagged events</h2>'+ev.to_html(index=False,escape=True)+'<h2>Column audit</h2>'+audits.to_html(index=False,escape=True)
    content+='<h2>Before operational use</h2><p>Confirm tag mapping, units, scaling, aggregation, tank/meter boundaries, equipment specifications and healthy reference periods. Review flags against maintenance/cleaning logs. Validate remote versus on-site action rules with BlueNexus. Conductivity is a limited water-quality indicator, not proof of compliance. Sparse chemical data are audited but not used for causal dosing analysis.</p></body></html>'
    (out/'report.html').write_text(content,encoding='utf-8')
    print(json.dumps(summary,indent=2));print(f'Report: {out / "report.html"}')

if __name__=='__main__':
    main()
