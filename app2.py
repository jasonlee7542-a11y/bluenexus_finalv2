from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from detector import screen
from review_core import load_table, audit_table, performance_review, equipment_review, EVENT_COLS

st.set_page_config(page_title='BlueNexus | Operating Review',layout='wide')
st.markdown('''<style>[data-testid="stAppViewContainer"]{background:#f3f7f8}h1,h2,h3{color:#153e50} [data-testid="stMetric"]{background:white;border:1px solid #d9e6ea;border-radius:12px;padding:18px}div.stButton>button{border-radius:8px}</style>''',unsafe_allow_html=True)
st.title('BlueNexus | Daily Operating Review')
st.caption('Challenge 1 prototype • Historical CSV analysis • Evidence for engineering review')
plant=st.sidebar.selectbox('Plant',['Plant A','Plant B'])
folder=Path(st.sidebar.text_input('Folder containing the three CSV files',str(Path(__file__).resolve().parent)))
logs=[]
try:
    equipment,_=load_table(folder/f'plant_{plant[-1]}_equipment_daily.csv','Equipment',logs)
    allperf,_=load_table(folder/'plant_daily_performance.csv','Performance',logs)
    if 'plant' not in allperf:raise ValueError('Performance file needs a plant column')
    perf=allperf.loc[allperf.plant.eq(plant)].copy()
    if equipment.empty or perf.empty:raise ValueError('Selected plant has no equipment or performance records')
except (FileNotFoundError,ValueError) as exc:
    st.error(str(exc));st.info('Place both equipment CSVs and plant_daily_performance.csv in the chosen folder.');st.stop()

audit=pd.concat([audit_table(equipment,'Equipment'),audit_table(perf,'Performance')],ignore_index=True)
events=[]
if plant=='Plant A':equipment=equipment_review(equipment,events)
settings={'reference_rows':31,'residual_scale_multiplier':3.,'energy_floor_fraction':.05,'sec_agreement_tolerance':.01}
perf,energy_model=performance_review(perf,events,settings)
tabs=st.tabs(['1 · Performance','2 · Equipment investigation','3 · Data reliability','4 · Method & export'])

def chart(fig):
    fig.update_layout(template='plotly_white',paper_bgcolor='rgba(0,0,0,0)',font_color='#173746',hovermode='x unified',margin=dict(t=45,b=35))
    st.plotly_chart(fig,width='stretch')

with tabs[0]:
    st.subheader('Which days deserve an operating review?')
    sec=perf.calculated_sec_kwh_m3
    a,b,c,d=st.columns(4)
    a.metric('Performance days',len(perf));b.metric('Median energy intensity',f'{sec.median():.3f} kWh/m³')
    c.metric('Median apparent recovery',f'{perf.calculated_apparent_recovery.median():.1%}')
    d.metric('Equipment / performance overlap',len(set(equipment.date)&set(perf.date)))
    view=perf[['date','effluent_volume_m3','daily_electricity_kwh','calculated_sec_kwh_m3','calculated_apparent_recovery']].copy()
    st.caption('The date selector ranks high energy intensity, not confirmed inefficient operation.')
    candidates=perf.loc[sec.notna()].sort_values('calculated_sec_kwh_m3',ascending=False)
    if candidates.empty:st.warning('No valid energy intensity values.')
    else:
        date=st.selectbox('Investigate a day (highest energy intensity first)',candidates.date.tolist(),format_func=lambda x:x.strftime('%d %b %Y'),key=f'day_{plant}')
        row=perf.loc[perf.date.eq(date)].iloc[0]
        comparisons=[]
        for col,label in [('effluent_volume_m3','Product volume'),('daily_electricity_kwh','Electricity'),('calculated_sec_kwh_m3','Energy per m³')]:
            med=perf[col].median();change=(row[col]/med-1)*100 if med>0 else np.nan
            comparisons.append({'Measure':label,'Selected day':row[col],'Period median':med,'Change vs median (%)':change})
        st.dataframe(pd.DataFrame(comparisons),hide_index=True,width='stretch')
        st.info('Conclusion: compare the change in production with the change in total electricity. A higher ratio identifies energy intensity; operating hours, cleaning and feed conditions are needed to explain it.')
        if row.effluent_volume_m3<perf.effluent_volume_m3.median():
            st.success('Next check: investigate reduced output and whether auxiliary loads or running hours stayed high. Review demand, shutdown and cleaning logs for this date.')
        else:st.success('Next check: compare operating hours, feed conditions, cleaning and electricity meter coverage against similar-production days.')
    left,right=st.columns(2)
    with left:chart(px.scatter(perf,x='effluent_volume_m3',y='daily_electricity_kwh',color='period',hover_data=['date','calculated_sec_kwh_m3'],labels={'effluent_volume_m3':'Product volume (m³/day)','daily_electricity_kwh':'Electricity (kWh/day)'},title='Electricity versus production'))
    with right:chart(px.line(perf,x='date',y='calculated_sec_kwh_m3',labels={'calculated_sec_kwh_m3':'kWh/m³'},title='Energy intensity over time'))
    quality_cols=[c for c in ['effluent_conductivity_us_cm','influent_conductivity_us_cm'] if c in perf]
    if quality_cols:chart(px.line(perf,x='date',y=quality_cols,title='Conductivity context (µS/cm) — not a compliance determination'))
    with st.expander('Exploratory electricity baseline and its limitations'):
        st.json(energy_model)
        st.write('First 31 records define the reference. Later records outside its production range are unscored. A weak fit and missing operating variables limit interpretation. No high-residual flags does not establish efficiency.')

original_cols=audit.loc[(audit.source=='Equipment'),'column'].tolist()
varying=[c for c in original_cols if equipment[c].notna().sum()>0 and equipment[c].nunique()>2]
counter=[c for c in varying if any(x in c.lower() for x in ['cumulative','_ep','advancedtags'])]
eligible=[c for c in varying if c not in counter]
result=None
with tabs[1]:
    st.subheader('Investigate changes in a selected equipment signal')
    st.caption('Units and tag meanings remain provisional. Statistical fences describe recent observations, not safe operating limits.')
    if not eligible:st.warning('No varying continuous signals available for screening.')
    else:
        default=next((i for i,c in enumerate(eligible) if c=='zz_1_rohigh_pressure_pump_power'),0)
        target=st.selectbox('Signal',eligible,index=default,key=f'target_{plant}')
        a,b,c=st.columns(3)
        mode=a.selectbox('Reference method',['Prior rolling window','Fixed first 31 days'])
        days=b.slider('Previous calendar days',7,30,14,disabled=mode!='Prior rolling window')
        mult=c.slider('IQR multiplier',1.,3.,1.5,.25)
        result=screen(equipment.set_index('date')[target],days,mult,mode)
        scored=result.status.isin(['Within statistical band','Statistical deviation'])
        a,b,c=st.columns(3);a.metric('Scored days',int(scored.sum()));b.metric('Flagged days',int(result.flag.sum()));c.metric('Unscored days',int((~scored).sum()))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=result.date,y=result.lower_fence,name='Lower statistical fence',line_color='#b6d8df',connectgaps=False))
        fig.add_trace(go.Scatter(x=result.date,y=result.upper_fence,name='Upper statistical fence',fill='tonexty',fillcolor='rgba(23,127,131,.12)',line_color='#b6d8df',connectgaps=False))
        fig.add_trace(go.Scatter(x=result.date,y=result.value,name=target,line_color='#177f83',connectgaps=False))
        f=result.loc[result.flag];fig.add_trace(go.Scatter(x=f.date,y=f.value,mode='markers',name='Statistical deviation',marker=dict(color='#d65c40',size=9)))
        fig.update_layout(title='Historical deviations from the chosen reference');chart(fig)
        if not f.empty:
            st.dataframe(f[['date','value','lower_fence','upper_fence','consecutive_flagged_days']],hide_index=True,width='stretch')
            st.write('Next check: compare equipment load/state, related signals and operating logs. A change of operating mode can produce the same flag as deterioration.')
        else:st.info('No deviations among scored readings. Unscored periods and gradual drift still require review.')
        st.caption('At least 80% reference coverage is required. Flat references are explicitly unscored. Rolling baselines can adapt to gradual deterioration; compare with the fixed reference and raw trend. Zeroes remain observations.')
        with st.expander('Distribution of observed readings'):
            chart(px.histogram(equipment,x=target,nbins=30,title='Distribution — mixed operating modes may form separate groups'))
    st.subheader('Compare related signals')
    search=st.text_input('Filter tag names',key=f'search_{plant}')
    choices=[c for c in original_cols if equipment[c].notna().any() and search.lower() in c.lower()]
    cols=st.columns(4);selected=[]
    for i,name in enumerate(choices):
        if cols[i%4].checkbox(name,key=f'overlay_{plant}_{name}'):selected.append(name)
    if selected:
        overlay=equipment[['date']+selected].copy();normalize=st.checkbox('Scale each selected signal to 0–100',value=True)
        if normalize:
            for name in selected:
                s=overlay[name];span=s.max()-s.min();overlay[name]=(s-s.min())/span*100 if span>0 else s.where(s.isna(),0)
        fig=px.line(overlay,x='date',y=selected,title='Shape comparison');fig.update_layout(yaxis_title='Min–max index' if normalize else 'Recorded values · units vary');chart(fig)
        st.caption('Matching shapes suggest a relationship to investigate. This chart does not calculate correlation or establish causation.')

with tabs[2]:
    st.subheader('Can these readings support the interpretation?')
    st.dataframe(audit,hide_index=True,width='stretch')
    st.write('NULL/blank means unavailable. Zero is an observed numerical value whose meaning depends on the tag. Missing readings remain missing; no interpolation or temperature clipping is performed.')
    st.dataframe(pd.DataFrame(logs),hide_index=True,width='stretch')
    missing_dates=pd.date_range(equipment.date.min(),equipment.date.max()).difference(equipment.date)
    st.write(f'Missing equipment calendar dates: {len(missing_dates)}')
    if len(missing_dates):st.write(missing_dates.strftime('%Y-%m-%d').tolist())
    if plant=='Plant A':
        st.write('Plant A checks: counter decreases and zero-frequency/nonzero-power pairs require tag/aggregation review.')
    else:st.caption('Plant B: raw counter-like tags remain available in the audit/overlay but are excluded from the default IQR selector pending definitions. No Plant A tag mapping is assumed.')
    st.dataframe(pd.DataFrame(events,columns=EVENT_COLS),hide_index=True,width='stretch')

with tabs[3]:
    st.subheader('Reproducible evidence')
    st.write('Workflow: parse and validate → audit → calculate indicators → compare against a defined reference → inspect flagged dates → recommend the next evidence check.')
    st.write('Exact duplicate rows are removed and counted. Conflicting daily keys and ambiguous dates stop processing. Invalid numeric tokens are logged before becoming missing. Entirely missing and constant columns remain in the audit. Binary and counter-like fields are excluded from default continuous-signal screening.')
    st.write('SEC = daily plant electricity / daily product volume. Apparent recovery = product volume / feed volume. Conductivity reduction = 1 − product conductivity / feed conductivity. Denominators must be positive; no epsilon is added. Units and measurement boundaries must match.')
    st.write('IQR fences = Q1 − k×IQR and Q3 + k×IQR. The rolling reference uses only previous calendar days, excluding the point being assessed. The fixed reference uses the first 31 calendar days and scores later observations only. Neither method defines safe, healthy or optimal operation.')
    st.write('No plant controls, verified failure forecasts or automatic repair/dispatch decisions are implemented. The dashboard runs locally and has no authentication layer; use appropriate hosting access controls if deployed.')
    outputs={'column_audit.csv':audit,'cleaning_log.csv':pd.DataFrame(logs),'prepared_equipment.csv':equipment,'performance_analysis.csv':perf,'review_events.csv':pd.DataFrame(events,columns=EVENT_COLS)}
    if result is not None:outputs['selected_signal_screening.csv']=result
    for name,table in outputs.items():st.download_button('Download '+name,table.to_csv(index=False,date_format='%Y-%m-%d').encode(),file_name=f'{plant.replace(" ","_")}_{name}',mime='text/csv')
    cfg={'plant':plant,'energy_model':energy_model,'energy_settings':settings}
    if result is not None:cfg.update({'signal':target,'mode':mode,'previous_days':days,'iqr_multiplier':mult,'minimum_coverage':.8})
    st.download_button('Download analysis settings',json.dumps(cfg,indent=2),file_name='analysis_settings.json',mime='application/json')
