#!/usr/bin/env python3
"""Pinned, resumable IndicVoices transcript intake. Outputs only, no model fitting.

Remote access uses only the existing Hugging Face token. No token arguments,
interactive login, gate acceptance, mirrors, audio decoding or whole-file fallback.
"""
import argparse, collections, concurrent.futures, hashlib, io, json, math, os, re, struct, sys, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

OUT=None
LANGUAGES=dict(zip('as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa sa sat sd ta te ur'.split(),
    'assamese bengali bodo dogri gujarati hindi kannada konkani kashmiri maithili malayalam manipuri marathi nepali odia punjabi sanskrit santali sindhi tamil telugu urdu'.split()))
COLUMNS=['text','verbatim','normalized','unsanitized_verbatim','unsanitized_normalized','speaker_id',
    'lang','duration','scenario','task_name','gender','age_group','job_type','qualification','area',
    'district','state','occupation','verification_report','audio_filepath.path']
SEED='sttok-native-primary-intake-2026-09-07-v1'
SOURCE_METADATA={}


class IntakeError(RuntimeError):pass

def digest(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):return digest(Path(path).read_bytes())
def rank(value):return digest((SEED+'|'+str(value)).encode())
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def token():
    from huggingface_hub import get_token
    return get_token()
def dependencies():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:raise IntakeError('pyarrow is unavailable. Use the native study dependencies (pyarrow, huggingface_hub).') from exc
    return pa,pq

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        nxt=super().redirect_request(req,fp,code,msg,headers,newurl)
        if nxt is not None and urllib.parse.urlsplit(req.full_url).netloc!=urllib.parse.urlsplit(newurl).netloc:
            nxt.remove_header('Authorization')
        if urllib.parse.urlsplit(newurl).scheme!='https':raise IntakeError('Refusing non-HTTPS redirect')
        return nxt

class RangeSource:
    """Reads only explicit byte ranges; never serializes credentials or signed URLs."""
    def __init__(self,url,auth=None,byte_limit=100*1024*1024,local=False):
        self.url=url;self.auth=auth;self.byte_limit=byte_limit;self.local=local
        self.requests=[];self.bytes=0;self.size=Path(url).stat().st_size if local else None
        self.opener=urllib.request.build_opener(SafeRedirect())
    def record(self,entry):
        self.requests.append(entry)
        if not self.local:
            journal=OUT/'primary-intake'/'network-requests.jsonl';journal.parent.mkdir(parents=True,exist_ok=True)
            payload=(json.dumps({'source_url':self.url,'recorded_unix':time.time(),**entry})+'\n').encode()
            fd=os.open(journal,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
            try:
                written=os.write(fd,payload)
                if written!=len(payload):raise IntakeError('Request journal write was incomplete')
                os.fsync(fd)
            finally:os.close(fd)
    def open_response(self,request):
        # Each worker issues at most one request at a time. Retry only transient
        # transport/server failures; account gates and ignored ranges fail closed.
        for attempt in range(4):
            try:return self.opener.open(request,timeout=30)
            except urllib.error.HTTPError as exc:
                if exc.code not in [408,429,500,502,503,504] or attempt==3:raise
                delay=min(30,2**(attempt+1))
                exc.close();time.sleep(delay)
            except (urllib.error.URLError,TimeoutError):
                if attempt==3:raise
                time.sleep(2**(attempt+1))
    def fetch(self,start,end,reason):
        if start<0 or end<=start:raise IntakeError('Invalid range')
        wanted=end-start
        if self.bytes+wanted>self.byte_limit:raise IntakeError('Projected I/O byte budget exhausted')
        if self.local:
            with open(self.url,'rb') as f:f.seek(start);data=f.read(wanted)
            status=206
        else:
            headers={'Range':f'bytes={start}-{end-1}','Accept-Encoding':'identity'}
            if self.auth:headers['Authorization']='Bearer '+self.auth
            try:
                with self.open_response(urllib.request.Request(self.url,headers=headers)) as response:
                    status=response.status
                    if status!=206:raise IntakeError(f'Refusing HTTP {status}: exact range response required')
                    cr=response.headers.get('Content-Range','');m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                    if not m or (int(m[1]),int(m[2])+1)!=(start,end):raise IntakeError('Content-Range does not match requested bytes')
                    if self.size is not None and self.size!=int(m[3]):raise IntakeError('Pinned file size changed during read')
                    self.size=int(m[3]);data=response.read(wanted+1)
            except urllib.error.HTTPError as exc:raise IntakeError(f'Hugging Face access/range request failed with HTTP {exc.code}; no response body or credentials logged') from None
            except urllib.error.URLError:raise IntakeError('Network request failed; safe to resume later') from None
        if len(data)!=wanted:raise IntakeError('Short or oversized range body')
        self.bytes+=len(data);self.record({'start':start,'end_exclusive':end,'bytes':len(data),'reason':reason,'status':status,'sha256':digest(data)})
        return data
    def footer_tail(self):
        if self.bytes+8>self.byte_limit:raise IntakeError('Projected I/O byte budget exhausted')
        if self.local:return self.fetch(self.size-8,self.size,'footer_tail')
        headers={'Range':'bytes=-8','Accept-Encoding':'identity'}
        if self.auth:headers['Authorization']='Bearer '+self.auth
        try:
            with self.open_response(urllib.request.Request(self.url,headers=headers)) as response:
                if response.status!=206:raise IntakeError(f'Refusing HTTP {response.status}: suffix range required')
                m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if not m or int(m[2])+1!=int(m[3]) or int(m[2])-int(m[1])!=7:raise IntakeError('Invalid footer suffix Content-Range')
                self.size=int(m[3]);data=response.read(9)
        except urllib.error.HTTPError as exc:raise IntakeError(f'Hugging Face access failed with HTTP {exc.code}; account access is required') from None
        except urllib.error.URLError:raise IntakeError('Network access failed; safe to resume later') from None
        if len(data)!=8:raise IntakeError('Invalid footer tail length')
        self.bytes+=8;self.record({'start':self.size-8,'end_exclusive':self.size,'bytes':8,'reason':'footer_tail','status':206,'sha256':digest(data)})
        return data

def load_footer(source):
    pa,pq=dependencies();tail=source.footer_tail()
    if tail[4:]!=b'PAR1':raise IntakeError('Unsupported/encrypted Parquet footer')
    length=struct.unpack('<I',tail[:4])[0]
    if not 0<length<=8*1024*1024 or length+12>source.size:raise IntakeError('Parquet footer length exceeds safe bound')
    start=source.size-8-length;footer=source.fetch(start,source.size-8,'footer_metadata')
    # Metadata contains original absolute column offsets. This tiny envelope lets Arrow
    # parse it without the normal trailing64KiB prefetch that could include audio bytes.
    metadata=pq.read_metadata(pa.BufferReader(b'PAR1'+footer+tail))
    return metadata,{'start':start,'sha256':digest(footer),'bytes':length,'row_groups':metadata.num_row_groups,'rows':metadata.num_rows}

def column_ranges(metadata,rowgroups,columns):
    allowed=[];selected=[];forbidden=[]
    for rg in range(metadata.num_row_groups):
        group=metadata.row_group(rg)
        for i in range(group.num_columns):
            col=group.column(i);name=col.path_in_schema
            offsets=[x for x in [col.dictionary_page_offset,col.data_page_offset] if x is not None and x>=0]
            if not offsets:raise IntakeError('Missing column page offsets')
            start=min(offsets);end=start+col.total_compressed_size
            audio_payload=(name.split('.')[-1]=='bytes' and name.split('.')[0] in ['audio','audio_filepath'])
            if audio_payload:forbidden.append((start,end,name,rg))
            if rg in rowgroups and name in columns:
                if audio_payload:raise IntakeError('Audio payload column requested')
                allowed.append((start,end));selected.append({'row_group':rg,'column':name,'start':start,'end_exclusive':end})
    return allowed,selected,forbidden

class ProjectedFile(io.RawIOBase):
    """Arrow cannot read any byte outside explicitly selected column chunks."""
    def __init__(self,source,allowed):
        super().__init__();self.source=source;self.allowed=allowed;self.pos=0;self.cache={}
        # Coalesce only contiguous selected chunks, never gaps or audio columns.
        merged=[]
        for a,b in sorted(allowed):
            if merged and a==merged[-1][1]:merged[-1]=(merged[-1][0],b)
            else:merged.append((a,b))
        for a,b in merged:self.cache[(a,b)]=source.fetch(a,b,'projected_column')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        value=offset if whence==0 else self.pos+offset if whence==1 else self.source.size+offset
        if value<0:raise IntakeError('Negative seek')
        self.pos=value;return value
    def read(self,size=-1):
        if size<0:raise IntakeError('Whole-file reads are forbidden')
        if size==0:return b''
        start=self.pos;end=start+size
        if not any(a<=start and end<=b for a,b in self.allowed):raise IntakeError('Arrow requested bytes outside projected columns')
        for (a,b),payload in self.cache.items():
            if a<=start and end<=b:
                self.pos=end;return payload[start-a:end-a]
        raise IntakeError('Selected column data was not prefetched')
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)

def projected_group(source,metadata,rg,columns):
    pa,pq=dependencies();names={metadata.row_group(rg).column(i).path_in_schema for i in range(metadata.row_group(rg).num_columns)}
    selected_columns=[c for c in columns if c in names]
    if not any(c in selected_columns for c in ['text','verbatim','transcript']):raise IntakeError('No expected transcript column in shard')
    allowed,selected,forbidden=column_ranges(metadata,[rg],selected_columns)
    reader=ProjectedFile(source,allowed)
    pf=pq.ParquetFile(reader,metadata=metadata,pre_buffer=False,buffer_size=0)
    table=pf.read_row_group(rg,columns=selected_columns,use_threads=False,use_pandas_metadata=False)
    for request in source.requests:
        if request['reason']!='projected_column':continue
        if any(request['start']<end and request['end_exclusive']>start for start,end,_,_ in forbidden):raise IntakeError('Audio column bytes unexpectedly touched')
    return table.to_pylist(),{'selected_columns':selected,'audio_column_bytes_read':0,'missing_requested_columns':[c for c in columns if c not in names]}

def catalog():
    return {k:json.loads(p.read_text()) for k,p in SOURCE_METADATA.items()}

def make_plan(args):
    source=catalog()['indicvoices'];revision=source['sha'];files=[x['rfilename'] for x in source['siblings']]
    split='valid' if args.command=='reserve' else 'train';target=args.reserve_rows if split=='valid' else args.train_rows
    groups=[]
    for lang in (args.languages or list(LANGUAGES)):
        config=LANGUAGES[lang];paths=sorted(p for p in files if p.startswith(config+'/'+split+'-') and p.endswith('.parquet'))
        if not paths:raise IntakeError(f'No pinned {split} shards for {lang}')
        # Deterministic evenly spread shard positions avoid a first-shard convenience prefix.
        n=min(args.shards,len(paths));indices=sorted({round(i*(len(paths)-1)/max(1,n-1)) for i in range(n)})
        chosen=[paths[i] for i in indices]
        groups.append({'language':lang,'configuration':config,'official_split':split,'target_rows':target,'available_shards':len(paths),'selected_shards':chosen})
    plan={'dataset':source['id'],'revision':revision,'metadata_sha256':file_sha(SOURCE_METADATA['indicvoices']),'role':'reserved_official_valid' if split=='valid' else 'training_intake_not_final_split','seed':SEED,'columns':COLUMNS,'selected_text_field':args.text_field,'normalization':'none; exact source values retained','speaker_cap':args.speaker_cap,'candidate_multiplier':args.candidate_multiplier,'max_rowgroups_per_shard':args.max_rowgroups,'byte_limit_per_language':args.max_language_mb*1024*1024,'groups':groups,'selection':'Evenly spread shard indices; deterministic hashed rowgroup order; then round-robin across observed scenario/state/district strata and speakers. This is an intake convenience design, not representative sampling or a verified dialect split.','reserve_policy':'Official valid only; text never emitted in logs. Root must check global speaker/source/text overlaps before using it as a final test.'}
    return plan

def balanced_select(rows,target,speaker_cap):
    strata=collections.defaultdict(lambda:collections.defaultdict(list));missing=0
    for r in rows:
        raw=r['raw'];speaker=str(raw.get('speaker_id') or '')
        if not speaker:missing+=1;continue  # Fail closed for speaker-aware sampling, with explicit count.
        stratum=tuple(str(raw.get(k) or '<missing>') for k in ['scenario','state','district'])
        strata[stratum][speaker].append(r)
    for speakers in strata.values():
        for speaker in speakers:speakers[speaker].sort(key=lambda r:rank(r['id']),reverse=True)
    counts=collections.Counter();selected=[];order=sorted(strata,key=lambda x:rank('|'.join(x)))
    queues={s:collections.deque(sorted(strata[s],key=rank)) for s in order}
    while len(selected)<target:
        progress=False
        for stratum in order:
            speakers=strata[stratum]
            queue=queues[stratum]
            while queue:
                speaker=queue.popleft()
                if not speakers[speaker] or counts[speaker]>=speaker_cap:continue
                selected.append(speakers[speaker].pop());counts[speaker]+=1;progress=True
                if speakers[speaker] and counts[speaker]<speaker_cap:queue.append(speaker)
                break
            if len(selected)==target:break
        if not progress:break
    return selected,{'missing_speaker_rows_excluded':missing,'unique_speakers':len(counts),'maximum_rows_per_speaker':max(counts.values(),default=0),'strata_in_candidate_pool':len(strata)}

def ingest(args):
    auth=token()
    if not auth:raise IntakeError('No existing Hugging Face token. Establish approved account access; no ingestion attempted.')
    dependencies();plan=make_plan(args)
    role='reserve' if args.command=='reserve' else 'train';receipt_dir=OUT/'primary-intake';atomic_json(receipt_dir/f'{role}-plan.json',plan)
    if args.workers==1:
        for group in plan['groups']:_ingest_group(args,plan,role,receipt_dir,auth,group)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(_ingest_group,args,plan,role,receipt_dir,auth,group) for group in plan['groups']]
            for future in concurrent.futures.as_completed(futures):future.result()

def _ingest_group(args,plan,role,receipt_dir,auth,group):
    language_plan={**{k:v for k,v in plan.items() if k!='groups'},'group':group}
    plan_hash=digest(json.dumps(language_plan,sort_keys=True).encode())
    lang=group['language'];final=receipt_dir/role/f'{lang}.jsonl';summary_path=final.with_suffix('.manifest.json')
    if final.exists() and summary_path.exists():
        previous=json.loads(summary_path.read_text())
        if previous['plan_sha256']!=plan_hash:raise IntakeError('Existing output belongs to a different plan; use a new run directory')
        if previous['sha256']!=file_sha(final):raise IntakeError('Existing output hash mismatch')
        print(json.dumps({'language':lang,'status':'verified_resume_complete','rows':previous['rows']}),flush=True);return
    candidates=[];seen=set();receipts=[];language_bytes=0;empty=0
    per_shard=math.ceil(group['target_rows']*plan['candidate_multiplier']/len(group['selected_shards']))
    for shard in group['selected_shards']:
        url=f'https://huggingface.co/datasets/{plan["dataset"]}/resolve/{plan["revision"]}/{shard}'
        cache_dir=receipt_dir/'cache'/role/lang/rank(shard)[:16];cache_dir.mkdir(parents=True,exist_ok=True)
        footer_cache=cache_dir/'footer.json';source=RangeSource(url,auth,plan['byte_limit_per_language']-language_bytes)
        metadata,footer=load_footer(source);rowgroup_order=sorted(range(metadata.num_row_groups),key=lambda rg:rank(shard+'|'+str(rg)))
        shard_rows=0
        for rg in rowgroup_order[:plan['max_rowgroups_per_shard']]:
            cache=cache_dir/f'rowgroup-{rg}.json';request_start=len(source.requests);byte_start=source.bytes
            if cache.exists():
                bundle=json.loads(cache.read_text())
                if bundle['revision']!=plan['revision'] or bundle['footer_sha256']!=footer['sha256'] or bundle['columns']!=plan['columns']:raise IntakeError('Cached rowgroup identity mismatch')
                rows=bundle['rows'];evidence=bundle['evidence'];cache_hit=True
                if bundle.get('rows_sha256')!=digest(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()):raise IntakeError('Cached transcript payload integrity mismatch')
            else:
                rows,evidence=projected_group(source,metadata,rg,plan['columns']);cache_hit=False
                bundle={'revision':plan['revision'],'shard':shard,'row_group':rg,'footer_sha256':footer['sha256'],'columns':plan['columns'],'rows':rows,'rows_sha256':digest(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()),'evidence':evidence}
                atomic_json(cache,bundle)
            row_offset=sum(metadata.row_group(i).num_rows for i in range(rg))
            for i,raw in enumerate(rows):
                value=raw.get(args.text_field)
                if not isinstance(value,str) or not value.strip():empty+=1;continue
                # No Unicode/lexical cleanup, trimming, tag deletion or transliteration.
                identity=f'{plan["dataset"]}@{plan["revision"]}:{shard}:{row_offset+i}'
                if identity in seen:continue
                seen.add(identity);speaker=str(raw.get('speaker_id') or '')
                path_value=raw.get('audio_filepath',{});recording_path=path_value.get('path') if isinstance(path_value,dict) else None
                candidates.append({'id':identity,'language':lang,'text':value,'text_field':args.text_field,'raw':raw,'source':'IndicVoices','domain':'indicvoices','dataset':plan['dataset'],'revision':plan['revision'],'source_shard':shard,'source_row_group':rg,'source_row_index':row_offset+i,'official_split':group['official_split'],'intake_role':plan['role'],'speaker_group':f'IndicVoices:{speaker}' if speaker else None,'recording_path':recording_path,'recording_group_status':'original path retained where available; session/dialogue identity must be verified, never inferred solely from speaker'})
                shard_rows+=1
            receipts.append({'shard':shard,'row_group':rg,'rows':len(rows),'cache_hit':cache_hit,'new_request_bytes':source.bytes-byte_start,'cache_file_sha256':file_sha(cache),'projection':evidence})
            if shard_rows>=per_shard:break
        language_bytes+=source.bytes
        atomic_json(footer_cache,{'dataset':plan['dataset'],'revision':plan['revision'],'shard':shard,'footer':footer,'file_bytes':source.size,'requests':source.requests,'body_bytes_read':source.bytes,'audio_column_bytes_read':0})
        print(json.dumps({'language':lang,'official_split':group['official_split'],'shards_done':len({x['shard'] for x in receipts}),'candidate_rows':len(candidates),'new_body_bytes':language_bytes}),flush=True)
    selected,selection=balanced_select(candidates,group['target_rows'],args.speaker_cap)
    final.parent.mkdir(parents=True,exist_ok=True);partial=final.with_suffix('.jsonl.partial')
    with partial.open('w') as f:
        for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    os.replace(partial,final)
    manifest={'plan_sha256':plan_hash,'dataset':plan['dataset'],'revision':plan['revision'],'language':lang,'official_split':group['official_split'],'rows':len(selected),'target_rows':group['target_rows'],'shortfall_rows':max(0,group['target_rows']-len(selected)),'candidate_rows':len(candidates),'empty_selected_text_rows':empty,'selection':selection,'sha256':file_sha(final),'new_http_body_bytes':language_bytes,'audio_column_bytes_read':0,'rowgroup_receipts':receipts,'raw_text_inspection':'No text samples emitted; source values preserved unchanged.','status':'intake_complete' if len(selected)>=group['target_rows'] else 'intake_shortfall_reported'}
    atomic_json(summary_path,manifest);print(json.dumps({'language':lang,'status':manifest['status'],'rows':len(selected),'sha256':manifest['sha256']}),flush=True)

def probe(args):
    auth=token();reports=[]
    for key,d in catalog().items():
        path=next(x['rfilename'] for x in d['siblings'] if x['rfilename'].endswith('.parquet'))
        source=RangeSource(f'https://huggingface.co/datasets/{d["id"]}/resolve/{d["sha"]}/{path}',auth,byte_limit=8*1024*1024)
        try:meta,footer=load_footer(source);status='footer_access_verified';error=None
        except IntakeError as exc:status='blocked';error=str(exc);footer=None
        reports.append({'dataset':d['id'],'revision':d['sha'],'token_present':bool(auth),'status':status,'error':error,'footer':footer,'body_bytes_read':source.bytes})
    atomic_json(OUT/'primary-intake'/'access-probe.json',reports);print(json.dumps(reports,indent=2))

def fixture_test(args):
    pa,pq=dependencies();folder=OUT/'primary-intake-fixture';folder.mkdir(exist_ok=True)
    n=64;payload=os.urandom(512*1024);rows=[]
    for i in range(n):rows.append({'text':f'fixture {i}','verbatim':f'raw {i} \u200d','normalized':f'std {i}','speaker_id':f'speaker-{i%16}','scenario':['read','extempore','conversation'][i%3],'state':f'state-{i%2}','district':f'district-{i%4}','audio_filepath':{'path':f'recording-{i}.wav','bytes':payload}})
    table=pa.Table.from_pylist(rows);path=folder/'large-audio.parquet';pq.write_table(table,path,compression=None,use_dictionary=False,row_group_size=8,write_statistics=False)
    source=RangeSource(str(path),local=True,byte_limit=1000000);metadata,footer=load_footer(source)
    projected,evidence=projected_group(source,metadata,3,COLUMNS)
    assert len(projected)==8 and projected[0]['verbatim']==rows[24]['verbatim']
    assert 'bytes' not in projected[0]['audio_filepath'] and projected[0]['audio_filepath']['path']=='recording-24.wav'
    assert source.bytes<path.stat().st_size*.01
    allowed,selected,forbidden=column_ranges(metadata,[3],[c for c in COLUMNS if c in {metadata.row_group(3).column(i).path_in_schema for i in range(metadata.row_group(3).num_columns)}])
    assert not any(r['start']<b and r['end_exclusive']>a for r in source.requests for a,b,_,_ in forbidden)
    blocked=False
    try:
        pf=ProjectedFile(source,[]);pf.seek(forbidden[0][0]);pf.read(min(128,forbidden[0][1]-forbidden[0][0]))
    except IntakeError:blocked=True
    assert blocked
    # A source preserves whitespace/joiners and raw variants; sampler keeps speaker caps.
    sample=[{'id':str(i),'raw':row,'text':row['verbatim']} for i,row in enumerate(rows)]
    chosen,stats=balanced_select(sample,40,2);assert len(chosen)==32 and stats['maximum_rows_per_speaker']==2
    assert balanced_select(list(reversed(sample)),40,2)[0]==chosen
    # Same stratum must distribute first picks across speakers before returning to one.
    one=[{'id':str(i),'raw':{'speaker_id':f's{i%8}','scenario':'same','state':'same','district':'same'}} for i in range(80)]
    fair,_=balanced_select(one,8,100);assert len({r['raw']['speaker_id'] for r in fair})==8
    # A server ignoring Range must be rejected without consuming a whole-file body.
    class WholeResponse:
        status=200
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,*a):raise AssertionError('Whole response body must not be read')
    class WholeOpener:
        def open(self,*a,**k):return WholeResponse()
    unsafe=RangeSource('https://huggingface.co/datasets/example/test/resolve/'+'0'*40+'/a.parquet');unsafe.opener=WholeOpener()
    refused=False
    try:unsafe.footer_tail()
    except IntakeError:refused=True
    assert refused and unsafe.bytes==0
    # A redirected credential is not sent to a different host. Uses a dummy marker.
    request=urllib.request.Request('https://huggingface.co/test',headers={'Authorization':'Bearer fixture-dummy','Range':'bytes=1-2'})
    redirected=SafeRedirect().redirect_request(request,None,302,'',{},'https://cdn.example.test/fixture')
    assert not redirected.has_header('Authorization') and redirected.get_header('Range')=='bytes=1-2'
    original_token=globals()['token'];globals()['token']=lambda:None;no_auth_stopped=False
    try:ingest(args)
    except IntakeError as exc:no_auth_stopped='No existing Hugging Face token' in str(exc)
    finally:globals()['token']=original_token
    assert no_auth_stopped
    result={'pyarrow_version':pa.__version__,'fixture_file_bytes':path.stat().st_size,'requested_body_bytes':source.bytes,'fraction_read':source.bytes/path.stat().st_size,'selected_rows':len(projected),'audio_column_bytes_read':0,'raw_joiner_and_text_exact':True,'audio_path_retained_payload_excluded':True,'forbidden_audio_read_rejected':blocked,'speaker_cap_and_order_determinism_passed':True,'round_robin_speaker_fairness_passed':True,'whole_http_response_rejected_without_body_read':True,'cross_host_auth_stripped':True,'no_token_intake_stops_before_network_or_rows':True,'requests':source.requests,'footer':footer,'evidence':evidence,'limitations':'Local instrumented byte-range fixture only. HTTP server behavior and authenticated remote schema remain to be verified with probe/first shard.'}
    result['fixture_sha256']=file_sha(path);path.unlink();result['temporary_fixture_removed']=True
    atomic_json(folder/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['requests','evidence']},indent=2))

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('command',choices=['plan','probe','ingest','reserve','fixture-test']);p.add_argument('--workers',type=int,choices=range(1,5),default=1);p.add_argument('--languages',nargs='+',choices=list(LANGUAGES));p.add_argument('--train-rows',type=int,default=15000);p.add_argument('--reserve-rows',type=int,default=1000);p.add_argument('--shards',type=int,default=8);p.add_argument('--candidate-multiplier',type=int,default=2);p.add_argument('--max-rowgroups',type=int,default=8);p.add_argument('--speaker-cap',type=int,default=100);p.add_argument('--max-language-mb',type=int,default=100);p.add_argument('--text-field',choices=['verbatim','normalized','unsanitized_verbatim','unsanitized_normalized','text'],default='verbatim');return p

def main():
    global OUT, SOURCE_METADATA
    args=parser().parse_args()
    OUT=args.output_root.resolve(); OUT.mkdir(parents=True,exist_ok=True)
    SOURCE_METADATA={'indicvoices':args.metadata.resolve()}
    if any(getattr(args,k)<=0 for k in ['train_rows','reserve_rows','shards','candidate_multiplier','max_rowgroups','speaker_cap','max_language_mb']):raise IntakeError('All count/budget settings must be positive')
    if args.command=='plan':
        plan=make_plan(args);atomic_json(OUT/'primary-intake'/'proposed-train-plan.json',plan);print(json.dumps(plan,indent=2))
    elif args.command=='probe':probe(args)
    elif args.command=='fixture-test':fixture_test(args)
    else:ingest(args)
if __name__=='__main__':
    try:main()
    except IntakeError as exc:print(json.dumps({'status':'stopped','reason':str(exc)}),file=sys.stderr);sys.exit(2)
