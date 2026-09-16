#!/usr/bin/env python3
"""Bounded original-source transcript intake. Does not request evaluation rows or audio."""
from __future__ import annotations
import concurrent.futures, hashlib, json, pathlib, random, re, urllib.parse, urllib.request, urllib.error, time, collections, unicodedata
import sys
CACHED_ONLY=False
ROOT=None
CACHE=None
SPRING_METADATA=None
CACHE_EXPECTED={}
SEED=20260907
AUDIT={'seed':SEED,'policy':'Only official train rows or explicitly nonbenchmark corpus directories. No benchmark/test/dev transcripts requested. No normalizer change. Unknown metadata remains null.','sources':[],'requests':[]}

def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url,max_bytes=15_000_000):
    key=sha(url.encode()); f=CACHE/(key+'.bin')
    if f.exists():
        b=f.read_bytes()
        sidecar=f.with_suffix('.sha256')
        expected=CACHE_EXPECTED.get(url) or (sidecar.read_text().strip() if sidecar.exists() else None)
        if not expected or sha(b)!=expected:raise ValueError('Cached response lacks a matching integrity receipt: '+url)
    elif CACHED_ONLY:raise FileNotFoundError('No cached response '+url)
    else:
        for attempt in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'sttok-corpus-intake/1.0'}),timeout=60) as r:b=r.read(max_bytes+1)
                break
            except urllib.error.HTTPError as exc:
                if exc.code==429:
                    print('Rate limited; backing off',url.split('?')[0],flush=True)
                    if attempt==3:raise
                    delay=min(45,max(10,int(exc.headers.get('Retry-After','15'))))
                    time.sleep(delay)
                elif attempt==3:raise
                else:time.sleep(1+attempt)
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)
        if len(b)>max_bytes:raise ValueError('Response exceeded cap '+url)
        f.write_bytes(b)
        f.with_suffix('.sha256').write_text(sha(b)+'\n')
    if len(b)>max_bytes:raise ValueError('Cached response exceeds acquisition cap')
    AUDIT['requests'].append({'url':url,'bytes':len(b),'sha256':sha(b),'cache_path':str(f.relative_to(ROOT))})
    return b

def api(url):return json.loads(fetch(url))
def assert_head(repo,rev):
    if CACHED_ONLY:return
    # Dataset viewer serves current default revision. Verify the advertised revision on both sides.
    d=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=30))
    if d['sha']!=rev:raise RuntimeError(f'Revision moved: {repo}: {d["sha"]} != {rev}')

def viewer(repo,config,offset,length):
    q=urllib.parse.urlencode({'dataset':repo,'config':config,'split':'train','offset':offset,'length':length})
    return api('https://datasets-server.huggingface.co/rows?'+q)

def write(name,rows):
    rows=sorted(rows,key=lambda x:x['record_id'])
    b=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows).encode()
    f=ROOT/name;f.write_bytes(b)
    return {'path':name,'rows':len(rows),'sha256':sha(b),'characters':sum(len(x['text']) for x in rows),'languages':dict(collections.Counter(x['language'] for x in rows))}

def clean_meta(text):
    # These are dataset annotation labels, not lexical model output. Keep false starts/repetitions.
    return re.sub(r'<(?:laugh|hesitation|unintelligible|noise)>',' ',text)

def ingest_meta():
    repo='facebook/omnilingual-asr-corpus';rev='8648ba8946377697b427ae952076e49fc0e5e44d'
    assert_head(repo,rev); allrows=[]; quarantine=[]
    for config,lang,total in [('brx_Deva','brx',380),('gom_Deva','kok',568),('kas_Arab','ks',796)]:
        tasks=[(o,min(100,total-o)) for o in range(0,total,100)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            batches=list(ex.map(lambda t:viewer(repo,config,*t),tasks))
        seen=set();n=0
        for b in batches:
            assert b.get('num_rows_total',total)==total
            for item in b['rows']:
                row=item['row']; idx=item['row_idx'];assert idx not in seen;seen.add(idx)
                raw=row['raw_text'];record=f'{repo}:{config}:train:{idx}'
                if '<' in clean_meta(raw) or '>' in clean_meta(raw):
                    quarantine.append({'record_id':record,'raw_text':raw,'reason':'malformed_or_undocumented_angle_annotation'}); n+=1; continue
                allrows.append({'text':clean_meta(raw),'raw_text':raw,'language':lang,'script':row['iso_15924'],'source':repo,'source_revision':rev,'source_split':'train','source_config':config,'record_id':record,'speaker_id':config+':'+str(row['speaker_id']),'recording_id':config+':'+str(row['speaker_id'])+':'+str(row['prompt_id']),'district':None,'style':'spontaneous_prompted','prompt_id':row['prompt_id'],'segment_id':row['segment_id'],'glottocode':row['glottocode'],'duration_seconds':row.get('duration'),'annotation_policy':'Remove documented four nonlexical tags; preserve all other text and raw_text.','label_status':'publisher_transcript_not_independently_reviewed','license':'CC-BY-4.0'})
                n+=1
        assert n==total,(config,n,total)
        AUDIT['sources'].append({'source':repo,'revision':rev,'config':config,'acquired_train_rows':n,'split':'train','license':'CC-BY-4.0','script_verified':True,'quality_limitation':'Language and transcript accuracy not independently validated. Recording key groups same speaker/prompt; original session unknown.'})
    assert_head(repo,rev)
    AUDIT['meta_output']=write('meta-train.jsonl',allrows)
    AUDIT['meta_quarantined_rows']=len(quarantine)
    (ROOT/'meta-quarantine.json').write_text(json.dumps(quarantine,ensure_ascii=False,indent=2))

def eligible_script(raw,lang):
    alphabet=[c for c in raw if unicodedata.category(c).startswith('L')]
    if not alphabet:return False
    # English code switching is expected, but require at least one target Arabic letter for ur.
    return any('ARABIC' in unicodedata.name(c,'') for c in alphabet)

def ingest_urdu():
    repo='ASLP-lab/UrduSpeech';rev='16dd380cfd9049a3db7f06a98e878086916bf833';assert_head(repo,rev)
    domains=['interview','podcast','roadside_interview','vlogs']
    paths=[]
    for variety in ['US-Std','US-CS']:
        for domain in domains:
            base=f'corpus/{variety}/short/{domain}'
            listing=api(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{base}?limit=1000')
            matches=[x['path'] for x in listing if x['path'].endswith('_final_transcription.jsonl')]
            if not matches:
                base=f'corpus/{variety}/long/{domain}'
                listing=api(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{base}?limit=1000')
                matches=[x['path'] for x in listing if x['path'].endswith('_final_transcription.jsonl')]
            assert len(matches)==1,(base,matches)
            assert matches[0].startswith('corpus/') and 'benchmark' not in matches[0].lower()
            paths.append((variety,domain,matches[0]))
    allrows=[];quarantine=[]
    for variety,domain,path in paths:
        url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}'
        data=fetch(url);rawrows=[json.loads(x) for x in data.splitlines() if x.strip()]
        groups=collections.defaultdict(list)
        for i,row in enumerate(rawrows):
            clip=row.get('Audio_Clip')
            if not clip:quarantine.append({'source_path':path,'source_line':i+1,'reason':'missing_clip_id'});continue
            groups[clip].append((i,row))
        eligible=[];conflicts=0;dups=0;wrong=0
        for clip,entries in groups.items():
            texts={x[1].get('Transcription','') for x in entries}
            if len(texts)!=1:
                conflicts+=1;quarantine.append({'source_path':path,'clip_id':clip,'source_lines':[i+1 for i,r in entries],'reason':'conflicting_transcriptions_for_clip','versions':len(texts)});continue
            i,row=entries[0];raw=row.get('Transcription','')
            if not isinstance(raw,str) or not eligible_script(raw,'ur'):wrong+=1;continue
            dups+=len(entries)-1
            record=f'{repo}:{path}:{clip}'
            eligible.append({'text':raw,'raw_text':raw,'language':'ur','script':'Arab','source':repo,'source_revision':rev,'source_split':'corpus_nonbenchmark','source_path':path,'source_line':i+1,'source_config':variety+'/'+domain,'record_id':record,'speaker_id':None if not row.get('Speaker_id') else variety+'/'+domain+':'+row['Speaker_id'],'speaker_id_status':'source_label_scoped_to_domain_not_verified_identity','recording_id':None,'clip_id':clip,'district':None,'style':domain,'label_status':'Gemini_assisted_publisher_manual_checks_not_gold','duration_seconds':row.get('Duration_seconds'),'confidence_source_self_reported':row.get('Confidence_score'),'accuracy_level_source_self_reported':row.get('Accuracy_level'),'source_audio_id':row.get('audio_id'),'license':'CC-BY-4.0_declared'})
        # Source-stratified stable hash sample over eligible records, not the file prefix.
        selected=sorted(eligible,key=lambda x:sha((str(SEED)+x['record_id']).encode()))[:250]
        allrows.extend(selected)
        AUDIT['sources'].append({'source':repo,'revision':rev,'path':path,'raw_rows':len(rawrows),'unique_clip_ids':len(groups),'conflicting_clip_ids_excluded':conflicts,'identical_duplicate_rows_collapsed':dups,'no_arabic_letter_or_empty_excluded':wrong,'eligible_unique_rows':len(eligible),'selected_rows':len(selected),'strategy':'Lowest stable record hash, at most250pervariety/domain. One audio-duration form per variety/domain, short preferred, long fallback only where short has no transcript. Never count both forms.','license':'CC-BY-4.0 declared by publisher','source_groups_unknown':True})
        print('Urdu',variety,domain,'raw',len(rawrows),'conflicts',conflicts,'selected',len(selected),flush=True)
    assert_head(repo,rev)
    AUDIT['urdu_output']=write('urdu-nonbenchmark.jsonl',allrows)
    (ROOT/'urdu-quarantine.json').write_text(json.dumps(quarantine,ensure_ascii=False,indent=2))

def ingest_spring():
    metadata=json.loads(SPRING_METADATA.read_text())
    code={'Assamese':'as','Bengali':'bn','Gujarati':'gu','Kannada':'kn','Malayalam':'ml','Marathi':'mr','Odia':'or','Punjabi':'pa','Tamil':'ta'}
    rows=[]
    for d in metadata:
        repo=d['repo']
        match=next((x for x in code if ('_'+x+'_R1') in repo),None)
        if match is None:continue
        lang=code[match];rev=d['revision'];assert_head(repo,rev)
        inf=d['card']['dataset_info'];total=next(x['num_examples'] for x in inf['splits'] if x['name']=='train')
        # 20 evenly spaced strata, random bounded block per stratum: at most500rows/source.
        rng=random.Random(str(SEED)+repo);tasks=[]
        for s in range(20):
            lo=s*total//20;hi=(s+1)*total//20
            start=rng.randint(lo,max(lo,hi-25));tasks.append((start,min(25,hi-start)))
        errors=[]
        def one_block(t):
            try:return viewer(repo,'default',*t)
            except (FileNotFoundError,urllib.error.HTTPError) as exc:
                errors.append({'offset':t[0],'length':t[1],'error':str(exc)});return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            batches=list(ex.map(one_block,tasks))
        count=0
        for batch in batches:
            if batch is None:continue
            for x in batch['rows']:
                row=x['row'];raw=row.get('text');uid=row.get('utterance_id')
                if not isinstance(raw,str) or not raw.strip():continue
                style='monologue' if uid and 'monologue' in str(uid).lower() else 'mixed_conversation_read_extempore_unclassified'
                rows.append({'text':raw,'raw_text':raw,'language':lang,'source':repo,'source_revision':rev,'source_split':'train','record_id':f'{repo}:train:{x["row_idx"]}:{uid}','utterance_id':uid,'speaker_id':None,'recording_id':None,'district':None,'style':style,'label_status':'publisher_manually_transcribed','license':'author_declared_public_domain_first_release','license_evidence':'https://github.com/Speech-Lab-IITM/SPRING_INX_ESPnet_Recipe#data-statistics','intake_status':'staged_permission_review','intake_note':'Original R1 not later Hindi1482 release. Data card lacks standardized license field; source author explicitly declares first release public domain. No automatic MIT inheritance.'});count+=1
        assert_head(repo,rev)
        AUDIT['sources'].append({'source':repo,'revision':rev,'selected_rows':count,'train_population_rows':total,'strategy':'20randomblocks of25 distributed evenly overtrain offsets; unavailable blocks are reported, not silently substituted','failed_blocks':errors,'complete':not errors,'permission':'Author primary README says both audio and corresponding manually transcribed data in first2000h release are in publicdomain. Recipe MIT not treated as data license.','speaker_session_metadata':'Not exposed in HF schema; raw utterance ID kept without invented parsing.','status':'staged_permission_review'})
        AUDIT['spring_output']=write('spring-r1-staged.jsonl',rows)
        (ROOT/'source-audit-spring.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2))
        print('SPRING',lang,count,flush=True)
    AUDIT['spring_output']=write('spring-r1-staged.jsonl',rows)

def main():
    import argparse
    global ROOT, CACHE, CACHED_ONLY, SPRING_METADATA, CACHE_EXPECTED
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sources',nargs='+',choices=['meta','urdu','spring'])
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--spring-metadata',type=pathlib.Path,required=True)
    p.add_argument('--cached-only',action='store_true')
    p.add_argument('--cache-receipt',type=pathlib.Path,nargs='*',default=[],help='Prior source-audit JSON receipts for historical cached responses')
    args=p.parse_args()
    ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    CACHE=ROOT/'cache';CACHE.mkdir(exist_ok=True)
    CACHED_ONLY=args.cached_only;SPRING_METADATA=args.spring_metadata
    for receipt in args.cache_receipt:
        for request in json.loads(receipt.read_text()).get('requests',[]):
            old=CACHE_EXPECTED.setdefault(request['url'],request['sha256'])
            if old!=request['sha256']:raise ValueError('Conflicting cached-response receipts')
    for arg in args.sources:
        try:globals()['ingest_'+arg]()
        finally:(ROOT/('source-audit-'+arg+'.json')).write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in AUDIT.items() if k.endswith('_output')},indent=2))

if __name__=='__main__':main()
