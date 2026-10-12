#!/usr/bin/env python3
"""Offline Chicago Berlin-compatible aggregate dashboard exporter.

Reads canonical SQLite only. Does not change the database or original captures.
"""
import argparse
import csv
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BQ_ORDER = ['18-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+']
BQ = {
 'M': dict(zip(BQ_ORDER,[10500,10800,11100,11700,12000,12600,13800,14700,15600,16500,17400])),
 'F': dict(zip(BQ_ORDER,[12300,12600,12900,13500,13800,14400,15600,16500,17400,18300,19200])),
 'X': dict(zip(BQ_ORDER,[12300,12600,12900,13500,13800,14400,15600,16500,17400,18300,19200]))}
POINTS = [('5K',5.0),('10K',10.0),('15K',15.0),('20K',20.0),('Half',21.0975),('25K',25.0),('30K',30.0),('35K',35.0),('40K',40.0),('Finish',42.195)]
SEGMENTS = ['Start-5K','5K-10K','10K-15K','15K-20K','20K-Half','Half-25K','25K-30K','30K-35K','35K-40K','40K-Finish']
BANDS = [('<3h',0,10800),('3–3:30',10800,12600),('3:30–4h',12600,14400),('4–4:30',14400,16200),('4:30–5h',16200,18000),('5h+',18000,360000)]
SPLIT_LABELS = ['Negative >2%','Even ±2%','Positive >2%']

def pct(n,d): return round(100*n/d,2) if d else 0

def med(xs): return statistics.median(xs) if xs else None

def fmt(x):
    if x is None: return None
    h, rem = divmod(round(x),3600); m,s = divmod(rem,60)
    return f'{h}:{m:02d}:{s:02d}'

def clock(v):
    if not v: return None
    m=re.search(r'\b(\d{1,2}):(\d\d)(?::(\d\d))?\b',str(v))
    if not m: return None
    h,mi,s=int(m[1]),int(m[2]),int(m[3] or 0)
    return h*3600+mi*60+s if h<24 and mi<60 and s<60 else None

def bins(vals,width):
    if not vals: return []
    lo=min(vals)//width*width
    hi=math.ceil(max(vals)/width)*width
    counts=Counter((v-lo)//width for v in vals)
    return [{'start':x,'end':x+width,'count':counts.get((x-lo)//width,0)} for x in range(lo,hi+width,width)]

def marker_label(v):
    s=(v or '').strip().lower().replace(' ','')
    if s in ('half','halb','halfmarathon'): return 'Half'
    if s=='finish': return 'Finish'
    m=re.fullmatch(r'0?(5|10|15|20|25|30|35|40)k(?:m)?',s)
    return m[1]+'K' if m else None

def dist(v):
    label=marker_label(v)
    if label: return dict(POINTS)[label]
    m=re.fullmatch(r'mile\s*(\d+(?:\.\d+)?)',(v or '').strip().lower())
    return float(m[1])*1.609344 if m else None

def age_group(division):
    """Only exact five-year division bands map to BQ; ambiguous/open divisions excluded."""
    s=(division or '').upper().strip()
    if re.search(r'\b80\s*\+',s): return '80+'
    m=re.search(r'(?<!\d)(\d{2})\s*[-–]\s*(\d{2})(?!\d)',s)
    if m:
        a,b=int(m[1]),int(m[2])
        if a>=18 and b>=a and b-a<=5:
            for label in BQ_ORDER[:-1]:
                lo,hi=map(int,label.split('-'))
                if lo<=a and b<=hi: return label
        return None
    # Single five-year division lower bound, e.g. M35 / F40 / 35
    m=re.search(r'(?<!\d)(\d{2})(?!\d)',s)
    if m:
        a=int(m[1]); return next((br for br in BQ_ORDER[:-1] if int(br.split('-')[0])<=a<=int(br.split('-')[1])), '80+' if a>=80 else None)
    return None

def gender(v):
    s=(v or '').strip().upper()
    return {'MALE':'M','MAN':'M','MEN':'M','FEMALE':'F','WOMAN':'F','WOMEN':'F','W':'F','NON-BINARY':'X','NONBINARY':'X'}.get(s,s if s in ('M','F','X') else 'Unknown')

def safe_json(data,path):
    forbidden={'name','idpcode','runnerid','fullname','citystate','club','detailurl'}
    def check(v):
        if isinstance(v,dict):
            for k,item in v.items():
                if re.sub('[^a-z0-9]','',k.lower()) in forbidden: raise ValueError('Forbidden public key: '+k)
                check(item)
        elif isinstance(v,list):
            for item in v: check(item)
    check(data)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(f'Wrote {path} ({path.stat().st_size:,} bytes)')


COUNTRY_SUFFIX = re.compile(r"\(([A-Z]{3})\)\s*$")

def country_aggregates(rows, mapping_path):
    """Aggregate CTZ suffixes from private canonical names; never export names."""
    if not mapping_path.exists():
        raise FileNotFoundError(f'Country mapping CSV not found: {mapping_path}')
    with mapping_path.open(encoding='utf-8-sig', newline='') as f:
        mapping = {r['Code'].strip().upper(): (r['Country'].strip(), r['Continent'].strip())
                   for r in csv.DictReader(f) if r.get('Code')}
    counts = Counter()
    for row in rows:
        match = COUNTRY_SUFFIX.search(row.get('name') or '')
        if match:
            counts[match.group(1)] += 1
    unmatched = {code: count for code, count in counts.items() if code not in mapping}
    countries = [{'code': code, 'country': mapping.get(code, (code, 'Unmapped'))[0],
                  'label': mapping.get(code, (code, 'Unmapped'))[0], 'count': count}
                 for code, count in counts.most_common()]
    continents = Counter()
    for code, count in counts.items():
        continents[mapping.get(code, (code, 'Unmapped'))[1]] += count
    regions = [{'label': region, 'count': count} for region, count in continents.most_common()]
    return countries, regions, len(rows)-sum(counts.values()), unmatched

def build(db,run_id,include_runners,country_map):
    db.row_factory=sqlite3.Row
    rows=[dict(r) for r in db.execute('SELECT * FROM canonical_runner_results WHERE cohort_run_id=? ORDER BY idp_code',(run_id,))]
    if not rows: raise ValueError('No canonical runners for run '+run_id)
    countries, regions, missing_country_codes, unmatched_codes = country_aggregates(rows, country_map)
    split_map=defaultdict(list)
    for r in db.execute('SELECT idp_code,sequence,marker,time_day,time,diff,min_km,kmh,min_mile,miles_h,elapsed_seconds,diff_seconds FROM canonical_runner_splits WHERE cohort_run_id=? ORDER BY idp_code,sequence',(run_id,)):
        split_map[r['idp_code']].append(dict(r))
    statuses=Counter(r['normalized_status'] for r in rows)
    if any(x not in {'finished','dnf','dns','dsq'} for x in statuses): raise ValueError('Unexpected statuses: '+repr(statuses))
    finishers=[r for r in rows if r['normalized_status']=='finished']
    finishes=[r['finish_seconds'] for r in finishers if isinstance(r['finish_seconds'],int) and r['finish_seconds']>0]
    if len(finishes)!=len(finishers): raise ValueError('Finishers without valid finish_seconds')
    # Finishers and DNF have race participation evidence by classification.
    # A DSQ status alone does not prove that the athlete crossed the start line.
    # Require a positive timed mat or finish time for DSQ starters; a start clock
    # may be a scheduled time and is not proof of participation.
    dsq_started = sum(
        r['normalized_status'] == 'dsq' and (
            (isinstance(r.get('valid_mat_count'), int) and r['valid_mat_count'] > 0)
            or (isinstance(r.get('finish_seconds'), int) and r['finish_seconds'] > 0)
            or any(isinstance(s.get('elapsed_seconds'), int) and s['elapsed_seconds'] > 0
                   for s in split_map[r['idp_code']])
        ) for r in rows
    )
    starters = len(finishers) + statuses['dnf'] + dsq_started
    dsq_unverified = statuses['dsq'] - dsq_started
    if starters > len(rows) - statuses['dns']:
        raise ValueError('Verified starters exceed all non-DNS runners')
    # Start-time clocks are plotted only for classified starters. Their presence
    # does not itself establish that a runner started.
    starts=[x for r in rows if (
        r['normalized_status'] in ('finished','dnf') or
        (r['normalized_status'] == 'dsq' and (
            (isinstance(r.get('valid_mat_count'),int) and r['valid_mat_count'] > 0) or
            (isinstance(r.get('finish_seconds'),int) and r['finish_seconds'] > 0) or
            any(isinstance(s.get('elapsed_seconds'),int) and s['elapsed_seconds'] > 0
                for s in split_map[r['idp_code']])
        ))
    ) if (x:=clock(r['start_time'])) is not None]
    genders=Counter(gender(r['gender']) for r in rows)
    ages=Counter((r['division'] or 'Unknown').strip() for r in finishers)
    age_dnf=Counter((r['division'] or 'Unknown').strip() for r in rows if r['normalized_status']=='dnf')
    gender_ages=defaultdict(Counter); gender_age_outcomes=defaultdict(lambda:defaultdict(Counter))
    for r in rows:
        a=(r['division'] or 'Unknown').strip(); g=gender(r['gender']); status=r['normalized_status']
        if status=='finished': gender_ages[a][g]+=1
        if status in ('finished','dnf'): gender_age_outcomes[a][status][g]+=1
    gender_splits=defaultdict(Counter); gender_walls=defaultdict(Counter); gender_band_slowdowns=defaultdict(lambda:defaultdict(list))
    bq_counts={a:{g:0 for g in ('M','F','X')} for a in BQ_ORDER}
    bq_eligible=bq_total=0
    for r in finishers:
        g=gender(r['gender']); a=age_group(r['division'])
        if g in BQ and a:
            bq_eligible+=1
            if r['finish_seconds']<=BQ[g][a]: bq_total+=1; bq_counts[a][g]+=1
    bq={'count':bq_total,'ratePct':pct(bq_total,bq_eligible),'eligibleSample':bq_eligible,
        'byAgeGender':[{'age':a,**bq_counts[a]} for a in BQ_ORDER],
        'note':'Meets the Boston qualifying time standard, not guaranteed entry. Chicago 2026 results are assessed for Boston 2028. Age is approximated from Chicago division bands; unknown and 19-and-under divisions are excluded because their Boston race-day age cannot be established.'}
    # Reference values are the exact BQ standards used by this analysis.
    # They are not a guarantee of acceptance or a verification of future standards.
    bq['standards'] = [
        {'age': age, 'men': fmt(BQ['M'][age]), 'women': fmt(BQ['F'][age]),
         'nonBinary': fmt(BQ['X'][age])}
        for age in BQ_ORDER
    ]
    bq['standardsNote'] = ('Reference times used by this exporter. Boston race-day age is '
        'approximated from Chicago divisions; qualifying standards and acceptance '
        'rules for the relevant Boston edition should be verified independently.')
    segment_values=defaultdict(list); runner_metrics=[]; dnf_checkpoints=Counter(); runner_exports=[]; split_exports=[]
    for export_index, r in enumerate(rows, start=1):
        export_key=f'CHI26-{export_index:06d}'  # Unique within this cohort; never exposes the private IDP.
        idp=r['idp_code']; raw=split_map[idp]
        # Keep one valid positive elapsed time for each standardized kilometer checkpoint.
        cumulative={}
        for s in raw:
            label=marker_label(s['marker']); t=s['elapsed_seconds']
            if label and isinstance(t,int) and t>0: cumulative[label]=t
        if r['normalized_status']=='finished': cumulative['Finish']=r['finish_seconds']
        # DNF distribution uses last valid mat distance; retain mile checkpoints in individual exports.
        if r['normalized_status']=='dnf':
            last_km=r.get('last_timed_mat_km')
            if last_km is None:
                distances=[dist(s['marker']) for s in raw if isinstance(s['elapsed_seconds'],int) and s['elapsed_seconds']>0 and marker_label(s['marker'])!='Finish']
                last_km=max((x for x in distances if x is not None),default=None)
            if last_km is not None:
                candidates=[lab for lab,d in POINTS[:-1] if d<=last_km+0.001]
                if candidates: dnf_checkpoints[candidates[-1]]+=1
        split_class=None; slowdown=None; split_pct=None
        if r['normalized_status']=='finished':
            prev_label,prev_dist,prev_t='Start',0.0,0
            for label,d in POINTS:
                t=cumulative.get(label)
                if t is not None and prev_t is not None and t>prev_t and d>prev_dist:
                    pace=(t-prev_t)/(d-prev_dist)
                    if 120<=pace<=1200: segment_values[f'{prev_label}-{label}'].append(pace)
                prev_label,prev_dist,prev_t=label,d,t
            half=cumulative.get('Half'); finish=r['finish_seconds']
            if half and finish>half:
                split_pct=((finish-half)/half-1)*100
                split_class='Negative >2%' if split_pct < -2 else 'Even ±2%' if split_pct<=2 else 'Positive >2%'
            if all(cumulative.get(k) is not None for k in ('10K','20K','30K','40K')):
                early=(cumulative['20K']-cumulative['10K'])/10
                late=(cumulative['40K']-cumulative['30K'])/10
                if early>0 and late>0: slowdown=(late/early-1)*100
            runner_metrics.append({'finish':finish,'splitClass':split_class,'splitPct':split_pct,'slowdown':slowdown})
            g=gender(r['gender'])
            if split_class: gender_splits[split_class][g]+=1
            if slowdown is not None and -80<slowdown<300:
                for wall_label,wall_test in [('Faster late',lambda x:x<=0),('0–5%',lambda x:0<x<=5),('5–10%',lambda x:5<x<=10),('10–20%',lambda x:10<x<=20),('20%+',lambda x:x>20)]:
                    if wall_test(slowdown): gender_walls[wall_label][g]+=1; break
                for band_label,band_lo,band_hi in BANDS:
                    if band_lo<=finish<band_hi: gender_band_slowdowns[band_label][g].append(slowdown); break
        if include_runners:
            runner_exports.append({'runnerKey':export_key,'bib':r['bib'],'status':r['normalized_status'],'gender':gender(r['gender']),
                'division':r['division'],'finishSec':r['finish_seconds'],'startTime':r['start_time'],
                'lastTimedMat':r.get('last_timed_mat'),'lastTimedMatKm':r.get('last_timed_mat_km'),
                'lastTimedMatSec':r.get('last_timed_mat_seconds'),'splitClass':split_class,
                'splitPct':round(split_pct,2) if split_pct is not None else None,
                'lateSlowdownPct':round(slowdown,2) if slowdown is not None else None})
            for s in raw:
                split_exports.append({'runnerKey':export_key,'bib':r['bib'],'sequence':s['sequence'],'marker':s['marker'],
                    'distanceKm':dist(s['marker']),'elapsedSec':s['elapsed_seconds'] if s['elapsed_seconds'] and s['elapsed_seconds']>0 else None,
                    'timeOfDay':s['time_day'],'diffSec':s['diff_seconds'],'paceMinKm':s['min_km'],
                    'speedKmh':s['kmh'],'paceMinMile':s['min_mile'],'speedMph':s['miles_h']})
    pacing=[{'segment':k,'medianPaceSecPerKm':round(med(segment_values[k]),1) if segment_values[k] else None,'sample':len(segment_values[k])} for k in SEGMENTS]
    sc=Counter(m['splitClass'] for m in runner_metrics if m['splitClass']); sc_total=sum(sc.values())
    split_summary=[{'label':x,'count':sc[x],'pct':pct(sc[x],sc_total)} for x in SPLIT_LABELS]
    slow=[m['slowdown'] for m in runner_metrics if m['slowdown'] is not None and -80<m['slowdown']<300]
    walls=[('Faster late',lambda x:x<=0),('0–5%',lambda x:0<x<=5),('5–10%',lambda x:5<x<=10),('10–20%',lambda x:10<x<=20),('20%+',lambda x:x>20)]
    wall_distribution=[{'label':label,'count':sum(test(x) for x in slow),'pct':pct(sum(test(x) for x in slow),len(slow))} for label,test in walls]
    by_band=[]
    for label,lo,hi in BANDS:
        vals=[m['slowdown'] for m in runner_metrics if lo<=m['finish']<hi and m['slowdown'] is not None and -80<m['slowdown']<300]
        by_band.append({'label':label,'medianSlowdownPct':round(med(vals),1) if vals else None,'sample':len(vals)})
    pace_clean=[p for p in pacing if p['medianPaceSecPerKm'] is not None and p['segment']!='20K-Half']
    wall_segment=max(zip(pace_clean,pace_clean[1:]),key=lambda pair:pair[1]['medianPaceSecPerKm']-pair[0]['medianPaceSecPerKm'])[1]['segment'] if len(pace_clean)>1 else None
    overview={'totalRunners':len(rows),'registeredRunners':len(rows),'starters':starters,'finishers':len(finishers),'finishRatePct':pct(len(finishers),starters),
        'startRatePct':pct(starters,len(rows)),'registeredFinishRatePct':pct(len(finishers),len(rows)),'dnfCount':statuses['dnf'],'verifiedDsqStarters':dsq_started,'unverifiedDsq':dsq_unverified,
        'medianFinishSec':round(med(finishes)) if finishes else None,'medianFinish':fmt(med(finishes)),
        'under4Count':sum(t<14400 for t in finishes),'under4Pct':pct(sum(t<14400 for t in finishes),len(finishes)),
        'fastestFinish':fmt(min(finishes)) if finishes else None,'lastFinish':fmt(max(finishes)) if finishes else None,
        'countriesRepresented':len(countries),'dsqCount':statuses['dsq'],'bqCount':bq_total,'bqRatePct':bq['ratePct'],
        'medianLateSlowdownPct':round(med(slow),1) if slow else None,
        'negativeSplitPct':split_summary[0]['pct'],'evenSplitPct':split_summary[1]['pct'],
        'positiveSplitPct':split_summary[2]['pct'],'largestSlowdownSegment':wall_segment}
    data={'overview':overview,'boston':bq,
        'status':[{'label':k,'count':(dsq_started if k=='dsq' else statuses[k])} for k in ('finished','dnf','dsq')],
        'gender':[{'label':k,'count':n} for k,n in genders.most_common()],
        'ageGroups':[{'label':k,'count':n} for k,n in sorted(ages.items())],
        'ageOutcomes':[{'label':k,'finishers':ages[k],'dnf':age_dnf[k]} for k in sorted(ages.keys() | age_dnf.keys())],
        'ageGender':[{'label':k,**dict(gender_ages[k])} for k in sorted(gender_ages)],
        'ageOutcomeGender':[{'label':k,'finished':dict(gender_age_outcomes[k]['finished']),'dnf':dict(gender_age_outcomes[k]['dnf'])} for k in sorted(gender_age_outcomes)],
        'countries':countries,'regions':regions, 'finishDistribution':bins(finishes,900),'startDistribution':bins(starts,1800),
        'dnfByLastCheckpoint':[{'label':k,'count':dnf_checkpoints[k]} for k,_ in POINTS[:-1] if dnf_checkpoints[k]],
        'pacing':pacing,'splitSummary':split_summary,'wallDistribution':wall_distribution,
        'splitByGender':[{'label':label,**dict(gender_splits[label])} for label in SPLIT_LABELS],
        'wallByGender':[{'label':label,**dict(gender_walls[label])} for label,_ in walls],
        'slowdownByFinishBandGender':[{'label':label,**{g:round(med(gender_band_slowdowns[label][g]),1) if gender_band_slowdowns[label][g] else None for g in ('M','F','X','Unknown')}} for label,_,_ in BANDS],
        'slowdownByFinishBand':by_band,'weather':[],
        'meta':{'year':2026,'raceDate':'2026-10-11','sourceRunId':run_id,
            'privacy':'Aggregate-only dashboard; optional runner-level exports are private and outside public.',
            'splitDefinition':'Negative/positive split categories use a ±2% even-split band.',
            'outcomeDefinition':'Finishers have valid finish times; DNF have positive intermediate timing evidence; DSQ count as starters only with positive timed mat or finish evidence. DSQ without timing evidence are excluded from starters and starter-outcome chart. DNS are excluded from starters. Finish rate = finishers / confirmed starters.',
            'slowdownDefinition':'Late slowdown compares average pace from 30–40K with 10–20K.',
            'note':'Country codes are extracted from the CTZ suffix of private canonical runner names; country and continent labels come from the supplied lookup CSV. Start-time histogram uses available start times, including nonfinishers. DNF mile markers are binned to the last preceding standard kilometer checkpoint.'}}
    audit={'runId':run_id,'runners':len(rows),'statuses':dict(statuses),'finishersWithValidTime':len(finishes),
        'verifiedStarters':starters,'verifiedDsqStarters':dsq_started,'unverifiedDsq':dsq_unverified,'validStartClocks':len(starts),'splitRows':sum(map(len,split_map.values())),
        'dnfWithBinnedCheckpoint':sum(dnf_checkpoints.values()),'dnfWithoutBinnedCheckpoint':statuses['dnf']-sum(dnf_checkpoints.values()),
        'halfSplitSample':sc_total,'slowdownSample':len(slow),'bqEligibleSample':bq_eligible,'bqMeetingStandard':bq_total,
        'missingDivision':sum(not r['division'] for r in rows),
        'missingBib':sum(not r['bib'] for r in rows),
        'duplicateBibs':sum(n-1 for n in Counter(r['bib'] for r in rows if r['bib']).values() if n>1),
        'countryCodesFound':sum(x['count'] for x in countries),
        'countryCodesMissing':missing_country_codes,'distinctCountryCodes':len(countries),
        'unmappedCountryCodes':unmatched_codes,
        'warnings':['Country codes are CTZ suffixes from canonical names, not inferred residence.',
                    'BQ ages approximated from Chicago divisions; 19-and-under and missing divisions excluded.',
                    'Optional private runner exports are pseudonymous, not anonymous.']}
    if sum(x['count'] for x in countries)+missing_country_codes!=len(rows): raise ValueError('Country code coverage mismatch')
    if sum(x['count'] for x in regions)!=sum(x['count'] for x in countries): raise ValueError('Continent count mismatch')
    if len(rows)!=sum(statuses.values()): raise ValueError('Status total mismatch')
    if starters != len(finishers)+statuses['dnf']+dsq_started: raise ValueError('Starter reconciliation mismatch')
    if sum(x['count'] for x in data['status']) != starters: raise ValueError('Starter outcome chart mismatch')
    if starters+statuses['dns']+dsq_unverified != len(rows): raise ValueError('Registered runner reconciliation mismatch')
    if sum(b['count'] for b in data['finishDistribution'])!=len(finishes): raise ValueError('Finish histogram total mismatch')
    return data,audit,runner_exports,split_exports

def load_weather(path):
    """Optional user-supplied weather observations; not independently verified."""
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    return [{'time':r['time'],'temperatureC':float(r['temperatureC']),
             'temperatureF':float(r['temperatureF']),
             'conditions':r['conditions'],'wind':r['wind']} for r in rows]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db',default='data/chicago/2026/marathon.sqlite')
    ap.add_argument('--run-id',default=None,help='Canonical run ID; defaults to most recent canonical run')
    ap.add_argument('--out-dir',default='public/data/chicago/2026')
    ap.add_argument('--include-runner-details',action='store_true',help='Optional private pseudonymous runner exports, never written under public/')
    ap.add_argument('--private-out-dir',default='private_exports/chicago/2026',help='Private runner-level output directory')
    ap.add_argument('--country-map',default='data/support/MapCountryData.csv',help='Offline country code to country/continent lookup')
    ap.add_argument('--weather-csv',default='data/chicago/2026/weather_race_day.csv',help='Optional offline weather CSV')
    args=ap.parse_args()
    path=Path(args.db)
    if not path.exists(): ap.error('SQLite database not found: '+str(path))
    db=sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro',uri=True)
    try:
        if args.run_id is None:
            result=db.execute('SELECT cohort_run_id,COUNT(*) AS n FROM canonical_runner_results GROUP BY cohort_run_id ORDER BY n DESC, cohort_run_id DESC LIMIT 1').fetchone()
            if not result: ap.error('No canonical results found')
            args.run_id=result[0]
        data,audit,runners,splits=build(db,args.run_id,args.include_runner_details,Path(args.country_map))
    finally: db.close()
    data['weather']=load_weather(Path(args.weather_csv))
    data['meta']['weatherSource']='User-supplied half-hourly observations; not independently verified' if data['weather'] else 'Unavailable'
    audit['weatherObservations']=len(data['weather'])
    out=Path(args.out_dir)
    for name in ('runners.json','runner_splits.json'):
        if (out/name).exists():
            print(f'WARNING: existing {out/name} is runner-level data and should be removed before publishing.')
    safe_json(data,out/'dashboard.json')
    safe_json(audit,out/'validation_audit.json')
    if args.include_runner_details:
        if audit['missingBib'] or audit['duplicateBibs']:
            print('Warning: missing or duplicate bibs; runnerKey uniquely links runners and splits.')
        private=Path(args.private_out_dir)
        if private.resolve()==out.resolve() or out.resolve() in private.resolve().parents or any(p.name.lower()=='public' for p in (private.resolve(),*private.resolve().parents)):
            ap.error('Private runner-level exports must be outside any public directory')
        safe_json(runners,private/'runners.json')
        safe_json(splits,private/'runner_splits.json')
    print(json.dumps(audit,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
