"""Train fresh reservoirs and fit native-constrained banks from a frozen corpus.

This script never imports the previous fitted research model. Native weights are
immutable; donor models contribute strings, not BPE scores. Outputs use the
explicit destination directory. An accepted data-manifest hash is required.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,io,json,math,re,time,unicodedata,shutil
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb
from untok.unigram_fit import NativeFitter,load_native,build_model

OUT=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[2]
ALPHABETS=REPO/'configs/native-study-alphabets.json'
BASE=None
DONORS=None
SOURCE_LOCK=REPO/'configs/native-study-sources.lock.json'
GROUPS={'deva':(['hi','mr','brx','doi','kok','mai','ne','sa','sd'],1400),
        'bengali':(['as','bn'],500)}
GROUPS.update({l:([l],500) for l in ['gu','kn','ml','or','pa','ta','te','ks','ur','mni','sat']})
JOBS={g:[g] for g in GROUPS if g not in ['ks','ur']}
JOBS['arabic']=['ks','ur']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def keyhash(seed,text):return hashlib.sha256(f'{seed}\0{text}'.encode()).hexdigest()
def readrows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]

def snapshot(data,langs):
    files=[data/'manifest.json']+[data/s/f'{l}.jsonl' for s in ['train','dev','reserve'] for l in langs]
    return {str(p.resolve()):sha(p) for p in files}

def sample_training(records,fraction,seed):
    """Nested deterministic source/speaker-group subsets, rescaled to source mass.

    This never uses reserve or development text for fitting. Known speaker IDs
    are grouped; absent IDs use record/document IDs. The data freeze remains the
    authority for cross-source speaker/session separation.
    """
    if fraction==1:return list(records)
    by_source=defaultdict(list)
    for r in records:by_source[(r['language'],r['source'])].append(r)
    result=[]
    for (lang,source),rows in sorted(by_source.items()):
        units=defaultdict(list)
        for r in rows:
            unit=next((r[k] for k in ['speaker_id','document_id','record_id']
                       if r.get(k) is not None and str(r[k]).strip().lower() not in {'','unknown','none','null','-1'}),
                      hashlib.sha256(r['text'].encode()).hexdigest())
            units[str(unit)].append(r)
        ordered=sorted(units,key=lambda u:keyhash(seed,lang+'\0'+source+'\0'+u))
        take=max(1,math.ceil(fraction*len(ordered)))
        chosen=[r for u in ordered[:take] for r in units[u]]
        # Coefficient rescaling is based on normalized-character exposure,
        # using the lengths already attached during intake below.
        before=sum(r['weight']*r['_normalized_chars'] for r in rows)
        after=sum(r['weight']*r['_normalized_chars'] for r in chosen)
        factor=before/after if after else 1.
        result.extend(dict(r,weight=r['weight']*factor) for r in chosen)
    return result


def group_alphabets(cfg,native):
    report=json.loads((ALPHABETS).read_text())
    alph={r['language']:set(r['standard_plus_current_required']['character_list']) for r in report['profiles']}
    targets={r['language']:r for r in cfg['targets']}
    hindi={s.split('\t')[0] for s in (REPO/'configs/approved-hindi-103.tsv').read_text().splitlines() if s}
    protected={};allowed={};patterns={}
    for g,(langs,budget) in GROUPS.items():
        mandatory=set().union(*(alph[l] for l in langs))
        if any(unicodedata.category(c) in {'Cn','Co','Cs'} for c in mandatory):
            raise ValueError(f'{g}: unassigned/private/surrogate codepoint in pinned mandatory alphabet')
        protected[g]=set(mandatory)
        if g=='deva':protected[g].update(hindi)
        ranges=set(tuple(r) for l in langs for r in targets[l]['ranges'])
        # Observed garbage is not made mandatory. Assigned letters/marks/numbers
        # and script punctuation can seed candidates; native/full-text unknowns
        # remain visible when the corpus contains other codepoints.
        charset={chr(c) for a,b in ranges for c in range(a,b+1)
                 if unicodedata.category(chr(c)) not in {'Cn','Co','Cs'} and not chr(c).isspace()}
        charset|=mandatory
        allowed[g]=charset
        rawchars=charset|{'\u200c','\u200d'}
        patterns[g]=re.compile('['+re.escape(''.join(sorted(rawchars)))+']+')
    return protected,allowed,patterns


def span_weights(records,langs,pattern,native_proc):
    weights=Counter();normalization_count=0;overlong=0
    for r in records:
        if r['language'] not in langs:continue
        for raw in pattern.findall(r['text']):
            if len(raw)>80:overlong+=1;continue
            text=native_proc.normalize(raw);normalization_count+=1
            if text.startswith('▁'):text=text[1:]
            # Trainer identity will escape these real normalized boundaries.
            # Removing exactly the synthetic prefix prevents double prefixes.
            text=text.replace('▁',' ')
            if text.strip():weights[text]+=float(r['weight'])
    return weights,{'raw_spans_normalized_once':normalization_count,'over80_raw_spans_excluded_from_seed_only':overlong,
                    'weighted_span_frequency':sum(weights.values()),'unique_normalized_spans':len(weights)}


def integer_frequency(value):return max(1,round(value*1_000_000))

def train_reservoir(weights,mandatory,size,seed,seed_file=None):
    spm.set_random_generator_seed(seed)
    ordered=sorted(weights,key=lambda w:(keyhash(seed,w),w))
    buf=io.BytesIO()
    kwargs=dict(sentence_iterator=iter(f'{w}\t{integer_frequency(weights[w])}' for w in ordered),
                input_format='tsv',model_writer=buf,model_type='unigram',vocab_size=size+4,
                hard_vocab_limit=False,required_chars=''.join(sorted(c for c in mandatory if len(c)==1)),
                character_coverage=1.,normalization_rule_name='identity',num_threads=1,
                shuffle_input_sentence=False,input_sentence_size=0,minloglevel=2)
    if seed_file is not None:kwargs['seed_sentencepieces_file']=str(seed_file)
    spm.SentencePieceTrainer.train(**kwargs)
    result=pb.ModelProto();result.ParseFromString(buf.getvalue())
    return result


def normal_scores(model):return {p.piece:float(p.score) for p in model.pieces if p.type==1 and p.piece!='▁'}

def donor_strings(langs):
    paths=[DONORS/f'{l}.model' for l in langs]
    bart=DONORS/'IndicBARTSS.model'
    paths.append(bart)
    if set(langs)&set(GROUPS['deva'][0]):
        bart=DONORS/'IndicBART.model'
        paths.append(bart)
    expected={Path(item['path']).name:item['sha256'] for item in json.loads(SOURCE_LOCK.read_text())['files'] if item['path'].startswith('donors/')}
    entries=defaultdict(set);pins={}
    for path in paths:
        if not path.is_file():raise FileNotFoundError('Required donor is missing: '+str(path))
        if sha(path)!=expected.get(path.name):raise ValueError('Donor differs from native source lock: '+str(path))
        p=pb.ModelProto();p.ParseFromString(path.read_bytes());pins[str(path)]=sha(path)
        for x in p.pieces:
            if x.type==1:entries[x.piece].add(str(path))
    return entries,pins


def permitted(piece,allowed):
    body=piece.removeprefix('▁')
    return bool(body) and len(body)<=40 and '▁' not in body and set(body)<=allowed


def make_seed(path,pieces,weights):
    trie={}
    for p in pieces:
        node=trie
        for c in p:node=node.setdefault(c,{})
        node[None]=p
    counts=Counter()
    for text,weight in weights.items():
        text='▁'+text.replace(' ','▁')
        for i in range(len(text)):
            node=trie
            for c in text[i:]:
                node=node.get(c)
                if node is None:break
                if None in node:counts[node[None]]+=weight
    content=''.join(f'{p}\t{integer_frequency(counts[p])}\n' for p in sorted(pieces))
    if path.exists() and path.read_text()!=content:
        raise ValueError(f'Existing seed file differs from recomputed frozen training frequencies: {path}')
    if not path.exists():path.write_text(content)
    return counts


def measure(model,records):
    proc=spm.SentencePieceProcessor(model_proto=model.SerializeToString())
    groups=defaultdict(list)
    for start in range(0,len(records),256):
        batch=records[start:start+256]
        seqs=proc.encode([r['text'] for r in batch],out_type=int,num_threads=2)
        for r,ids in zip(batch,seqs):
            chars=max(1,len(proc.normalize(r['text'])))
            groups[r['language']].append({'tokens':len(ids),'unknown':ids.count(proc.unk_id()),'chars':chars})
    result={}
    for lang,rows in groups.items():
        ratios=sorted(r['tokens']/r['chars'] for r in rows);lengths=sorted(r['tokens'] for r in rows)
        result[lang]={'records':len(rows),'tokens':sum(r['tokens'] for r in rows),'unknown_tokens':sum(r['unknown'] for r in rows),
                      'normalized_characters':sum(r['chars'] for r in rows),
                      'tokens_per_normalized_character':sum(r['tokens'] for r in rows)/sum(r['chars'] for r in rows),
                      'mean_record_tokens_per_character':sum(ratios)/len(rows),
                      'p95_record_tokens_per_character':ratios[int(.95*(len(rows)-1))],
                      'p95_record_tokens':lengths[int(.95*(len(rows)-1))]}
    return result


def choose(results,unknown_reference,tail_ratio=1.02):
    eligible=[]
    for r in results:
        if not all(r['gates'].values()):continue
        okay=True
        for lang,m in r['development'].items():
            baseline=unknown_reference[lang]
            if m['unknown_tokens']>baseline['unknown_tokens']:okay=False
            if m['p95_record_tokens_per_character']>baseline['p95_record_tokens_per_character']*tail_ratio:okay=False
        if okay:eligible.append(r)
    if not eligible:return None
    return min(eligible,key=lambda r:(sum(m['tokens_per_normalized_character'] for m in r['development'].values())/len(r['development']),r['name']))


def run_job(args):
    data=args.data_dir.resolve();manifest=data/'manifest.json'
    if sha(manifest)!=args.accept_data_sha256:raise ValueError('Data manifest hash differs from explicitly accepted freeze')
    declared=json.loads(manifest.read_text())
    declared_files=declared.get('files')
    if not isinstance(declared_files,dict) or not declared_files:raise ValueError('Frozen manifest must declare relative file SHA256 digests')
    for name,digest in declared_files.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Frozen file paths must be relative and contained')
        if not isinstance(digest,str) or sha(data/relative)!=digest:raise ValueError(f'Frozen declared digest mismatch: {name}')
    if args.job not in JOBS:raise ValueError(f'job must be one of {sorted(JOBS)}')
    groups=JOBS[args.job];langs=[l for g in groups for l in GROUPS[g][0]]
    for split in ['train','dev','reserve']:
        for lang in langs:
            if f'{split}/{lang}.jsonl' not in declared_files:raise ValueError(f'Unpinned frozen input: {split}/{lang}.jsonl')
    before=snapshot(data,langs)
    native_path=BASE
    expected_base=declared.get('native_base_sha256') or args.expected_native_base_sha256
    if not expected_base or sha(native_path)!=expected_base:raise ValueError('Native base hash differs from frozen/explicit expected base')
    native_bytes=native_path.read_bytes();native=load_native(native_bytes)
    norm=spm.SentencePieceProcessor(model_proto=native_bytes)
    cfg=json.loads((REPO/'configs/build.json').read_text())
    protected,allowed,patterns=group_alphabets(cfg,native)
    external_reference=None
    if args.reference_result:
        if not args.reference_result_sha256 or sha(args.reference_result)!=args.reference_result_sha256:
            raise ValueError('External common reference digest is not explicitly pinned')
        external_reference=json.loads(args.reference_result.read_text())
        if external_reference.get('fraction')!=1 or external_reference.get('seed')!=0 or external_reference.get('family')!='fresh' or not external_reference.get('development'):
            raise ValueError('External common reference must be full-data seed0 fresh with development metrics')
        refprotocol=json.loads((args.reference_result.parent.parent/'protocol.json').read_text())
        if refprotocol.get('accepted_data_manifest_sha256')!=args.accept_data_sha256 or refprotocol.get('native_sha256')!=expected_base:
            raise ValueError('External reference used a different data or native-base freeze')
        artifacts=external_reference.get('artifact_sha256',{})
        if set(artifacts)!={'tokenizer.model','selected-pieces.json','fit-report.json'}:
            raise ValueError('External reference artifact integrity is incomplete')
        for filename,digest in artifacts.items():
            if sha(args.reference_result.parent/filename)!=digest:raise ValueError('External common reference artifact changed')
        if set(external_reference['development'])!=set(langs):raise ValueError('External reference profile set differs from job')
    rawtrain=[r for l in langs for r in readrows(data/'train'/f'{l}.jsonl')]
    dev=[r for l in langs for r in readrows(data/'dev'/f'{l}.jsonl')]
    if any('flores' in str(r.get('source','')).lower() for r in rawtrain):raise ValueError('FLORES training source is forbidden')
    for r in rawtrain:
        if not isinstance(r.get('weight'),(float,int)) or r['weight']<=0:raise ValueError('Positive frozen explicit row weight required')
        r['_normalized_chars']=len(norm.normalize(r['text']))
    jobdir=args.output_root.resolve()/args.job;jobdir.mkdir(parents=True,exist_ok=True)
    protocol={'accepted_data_manifest_sha256':args.accept_data_sha256,'input_snapshot':before,
              'native_sha256':sha(native_path),'candidate_code_sha256':sha(__file__),
              'mandatory_policy_file_sha256':{str(path):sha(path) for path in [REPO/'configs/build.json', REPO/'configs/approved-hindi-103.tsv', ALPHABETS]},
              'native_fit_code_sha256':sha(Path(__import__(NativeFitter.__module__,fromlist=["__file__"]).__file__)),'groups':groups,'languages':langs,
              'budgets':{g:GROUPS[g][1] for g in groups},'fractions':args.fractions,'seeds':args.seeds,
              'mass_multipliers':args.mass_multipliers,'families':args.families,
              'external_common_reference_sha256':sha(args.reference_result) if args.reference_result else None,
              'score_mass_policy':'max(initial_candidate_added_score_mass * multiplier, native_only_mandatory_minimum_mass * 1.01); exact value recorded before fit',
              'selection':'Only passed structural/usage/reachability candidates; no per-language unknown increase and p95 token-per-character ratio <= 1.02 against one common f1-s0 fresh smallest-mass baseline. Minimize macro language tokens per native-normalized character. Final choice compares full-data trials across seeds. Full record development only; reserve never read for selection.',
              'seed_semantics':'SentencePiece random generator seed plus reproducible hashed input order; frequencies and ranking stay evidence-based. Identical full-corpus outputs across seeds are reported, not claimed as independent variation.',
              'training_raw_rows':len(rawtrain),'training_normalized_characters':sum(r['_normalized_chars'] for r in rawtrain),
              'fit_balancing':'Use frozen row weights unchanged; do not rebalance inside NativeFitter.',
              'comparison_model_sha256':sha(args.comparison_model) if args.comparison_model else None}
    existing=jobdir/'protocol.json'
    if existing.exists() and json.loads(existing.read_text())!=protocol:
        raise ValueError('Existing job protocol differs; use a separate output experiment or exact resume')
    write(existing,protocol)
    if args.comparison_model and not (jobdir/'old-union-development.json').exists():
        old=load_native(args.comparison_model.read_bytes())
        write(jobdir/'old-union-development.json',{'model_sha256':sha(args.comparison_model),
              'development':measure(old,dev),'scope':'Comparison only, never initialization. Old full combined bank has other Indic scripts absent from a single-job candidate; compare final combined model separately.'})
    results=[]
    for fraction in args.fractions:
      for seed in args.seeds:
        train=sample_training(rawtrain,fraction,seed)
        sample_stats={'requested_fraction':fraction,'seed':seed,'selected_raw_rows':len(train),
                      'normalized_characters':sum(r['_normalized_chars'] for r in train),
                      'per_language_rows':dict(Counter(r['language'] for r in train)),
                      'per_source_character_exposure':{source:sum(r['weight']*r['_normalized_chars'] for r in train if r['source']==source) for source in sorted({r['source'] for r in train})}}
        pooled={family:{} for family in args.families};pool_manifest={}
        old_pool_path=jobdir/f'f{fraction:g}-s{seed}'/'pools.json'
        old_pools=json.loads(old_pool_path.read_text()) if old_pool_path.exists() else {}
        for g in groups:
            langset,budget=GROUPS[g]
            directory=jobdir/f'f{fraction:g}-s{seed}'/g;directory.mkdir(parents=True,exist_ok=True)
            weights,stats=span_weights(train,langset,patterns[g],norm)
            donors,pins=donor_strings(langset)
            previous_pool=old_pools.get(g)
            if previous_pool and previous_pool['donor_file_sha256']!=pins:raise ValueError(f'Donor files changed on resume: {g}')
            fresh_path=directory/'fresh-reservoir.model'
            if fresh_path.exists():
                if not previous_pool or sha(fresh_path)!=previous_pool['fresh_reservoir_sha256']:raise ValueError(f'Unverified/corrupt fresh reservoir on resume: {fresh_path}')
                fresh=load_native(fresh_path.read_bytes())
            else:
                fresh=train_reservoir(weights,protected[g],budget*3,seed)
                fresh_path.write_bytes(fresh.SerializeToString())
            fresh_scores=normal_scores(fresh)
            donor_set={p for p in donors if permitted(p,allowed[g])}
            native_set={p.piece for p in native.pieces if p.type==1 and permitted(p.piece,allowed[g])}
            seedpieces=set(fresh_scores)|donor_set|native_set|protected[g]|{'▁'}
            seedpath=directory/'donor-union.seed.tsv'
            if seedpath.exists() and previous_pool and sha(seedpath)!=previous_pool['seed_file_sha256']:raise ValueError(f'Corrupt seed file on resume: {seedpath}')
            occurrence=make_seed(seedpath,seedpieces,weights)
            union_path=directory/'donor-union-reservoir.model'
            if union_path.exists():
                if not previous_pool or sha(union_path)!=previous_pool['union_reservoir_sha256']:raise ValueError(f'Unverified/corrupt union reservoir on resume: {union_path}')
                union=load_native(union_path.read_bytes())
            else:
                union=train_reservoir(weights,protected[g],budget*3,seed,seedpath)
                union_path.write_bytes(union.SerializeToString())
            for family,model in [('fresh',fresh),('donor_union',union)]:
                if family not in pooled:continue
                scores=normal_scores(model)
                for p in protected[g]:scores.setdefault(p,min(scores.values())-2)
                for p,score in scores.items():
                    if not permitted(p,allowed[g]) and p not in protected[g]:continue
                    entry=pooled[family].setdefault(p,{'piece':p,'score':score,'groups':[],
                                                     'provenance':[]})
                    entry['score']=max(entry['score'],score);entry['groups'].append(g)
                    entry['provenance'].append({'group':g,'family':family,'fresh_training_split':'frozen train',
                                                'donor_string_sources':sorted(donors.get(p,[])) if family=='donor_union' else []})
            pool_manifest[g]={'span_data':stats,'donor_file_sha256':pins,'fresh_reservoir_sha256':sha(fresh_path),
                              'union_reservoir_sha256':sha(union_path),'seed_file_sha256':sha(seedpath),
                              'candidate_union_strings':len(seedpieces),'corpus_observed_seed_strings':sum(v>0 for v in occurrence.values())}
        write(jobdir/f'f{fraction:g}-s{seed}'/'pools.json',pool_manifest)
        write(jobdir/f'f{fraction:g}-s{seed}'/'training-subset.json',sample_stats)
        local=[]
        for family,candidates in pooled.items():
          fitter=NativeFitter(native,train,candidates.values(),{g:GROUPS[g][1] for g in groups},
                              {g:protected[g] for g in groups},balance_language_source=False)
          policy=fitter.policy();write(jobdir/f'f{fraction:g}-s{seed}'/f'{family}-policy.json',policy)
          mass_cache={}
          for multiplier in args.mass_multipliers:
            name=f'f{fraction:g}-s{seed}-{family}-m{multiplier:g}';folder=jobdir/name;folder.mkdir(exist_ok=True)
            resultfile=folder/'result.json'
            if resultfile.exists():
                r=json.loads(resultfile.read_text())
                expected_artifacts=r.get('artifact_sha256')
                if not isinstance(expected_artifacts,dict):raise ValueError('Missing resume artifact hashes')
                if r.get('model_sha256') and set(expected_artifacts)!={'tokenizer.model','selected-pieces.json','fit-report.json'}:raise ValueError('Incomplete successful resume artifact hashes')
                for file,digest in expected_artifacts.items():
                    if file not in {'tokenizer.model','selected-pieces.json','fit-report.json'} or sha(folder/file)!=digest:raise ValueError(f'Resume artifact hash mismatch: {folder/file}')
                if r.get('model_sha256') and sha(folder/'tokenizer.model')!=r['model_sha256']:raise ValueError('Resume model hash mismatch')
                mass_cache[r['added_score_mass']]=r
                local.append(r);results.append(r);continue
            mass=max(policy['initial_candidate_added_score_mass']*multiplier,
                     policy['native_only_mandatory_minimum_mass']*1.01)
            if mass in mass_cache:
                previous=mass_cache[mass]
                r=dict(previous,name=name,mass_multiplier=multiplier,duplicate_of=previous['name'],elapsed_seconds=0.)
                oldfolder=jobdir/previous['name']
                for filename in ['tokenizer.model','selected-pieces.json','fit-report.json']:
                    if (oldfolder/filename).exists():shutil.copyfile(oldfolder/filename,folder/filename)
                write(resultfile,r);local.append(r);results.append(r)
                print('DUPLICATE MASS',args.job,name,previous['name'],flush=True);continue
            started=time.monotonic();print('FIT',args.job,name,'rows',len(train),'mass',mass,flush=True)
            try:
                result=fitter.fit(added_score_mass=mass,em_passes=args.em_passes,
                                  prune_fraction=.15,refill_rounds=6)
                (folder/'tokenizer.model').write_bytes(result['model_proto'].SerializeToString())
                write(folder/'selected-pieces.json',result['selected_pieces']);write(folder/'fit-report.json',result['report'])
                metrics=measure(result['model_proto'],dev)
                r={'name':name,'family':family,'fraction':fraction,'seed':seed,'mass_multiplier':multiplier,
                   'added_score_mass':mass,'model_sha256':sha(folder/'tokenizer.model'),
                   'gates':result['report']['gates'],'development':metrics,
                   'fit_status':result['report']['status'],'elapsed_seconds':time.monotonic()-started}
            except (ValueError,RuntimeError) as e:
                r={'name':name,'family':family,'fraction':fraction,'seed':seed,'mass_multiplier':multiplier,
                   'added_score_mass':mass,'gates':{'fit_completed':False},'development':{},
                   'fit_status':'failed','error':str(e),'elapsed_seconds':time.monotonic()-started}
            r['artifact_sha256']={filename:sha(folder/filename) for filename in ['tokenizer.model','selected-pieces.json','fit-report.json'] if (folder/filename).exists()}
            write(resultfile,r);local.append(r);results.append(r);mass_cache[mass]=r
            print('DONE',args.job,name,r['fit_status'],round(r['elapsed_seconds'],2),flush=True)
        write(jobdir/f'f{fraction:g}-s{seed}'/'local-results.json',{'candidates':[r['name'] for r in local],'selection_pending_common_full_data_seed0_reference':True})
    if snapshot(data,langs)!=before:raise RuntimeError('Frozen data changed during fitting')
    references=[r for r in results if r['fraction']==1 and r['seed']==0 and r['family']=='fresh' and r['development']]
    common_reference=external_reference or (min(references,key=lambda r:(r['added_score_mass'],r['name'])) if references else None)
    for fraction in args.fractions:
        for seed in args.seeds:
            local=[r for r in results if r['fraction']==fraction and r['seed']==seed]
            winner=choose(local,common_reference['development']) if common_reference else None
            write(jobdir/f'f{fraction:g}-s{seed}'/'choice.json',{'common_reference':common_reference['name'] if common_reference else None,
                  'winner':winner['name'] if winner else None,'no_passing_candidate':winner is None,
                  'reference_missing':common_reference is None,'candidates':[r['name'] for r in local]})
    final=choose([r for r in results if r['fraction']==1],common_reference['development']) if common_reference else None
    write(jobdir/'final-choice.json',{'common_reference':common_reference['name'] if common_reference else None,
          'winner':final['name'] if final else None,
          'purpose':'final_selection' if any(r['fraction']==1 for r in results) else 'learning_curve_only',
          'no_passing_candidate':final is None if any(r['fraction']==1 for r in results) else None,
          'reference_missing':common_reference is None})
    write(jobdir/'results.json',results)
    hashgroups=defaultdict(list)
    for r in results:
        if r.get('model_sha256'):hashgroups[r['model_sha256']].append(r['name'])
    write(jobdir/'duplicate-models.json',{'same_hash_groups':{h:names for h,names in hashgroups.items() if len(names)>1},
          'unique_successful_model_hashes':len(hashgroups),'successful_trial_rows':sum(len(v) for v in hashgroups.values()),
          'note':'Identical masses/models are counted as repeated configurations, not independent candidate evidence.'})
    print('JOB COMPLETE',args.job,len(results),'candidates',flush=True)


def main():
    global REPO, ALPHABETS, BASE, DONORS, SOURCE_LOCK
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--donors',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--alphabets',type=Path,default=ALPHABETS)
    p.add_argument('--source-lock',type=Path,default=SOURCE_LOCK)
    p.add_argument('--accept-data-sha256',required=True)
    p.add_argument('--expected-native-base-sha256')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--job',choices=sorted(JOBS),required=True)
    p.add_argument('--fractions',nargs='+',type=float,default=[1.])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    p.add_argument('--mass-multipliers',nargs='+',type=float,default=[.8,1.2])
    p.add_argument('--em-passes',type=int,default=3)
    p.add_argument('--families',nargs='+',choices=['fresh','donor_union'],default=['fresh','donor_union'])
    p.add_argument('--reference-result',type=Path)
    p.add_argument('--reference-result-sha256')
    p.add_argument('--comparison-model',type=Path)
    args=p.parse_args()
    REPO=args.repo.resolve(); ALPHABETS=args.alphabets.resolve(); BASE=args.base.resolve(); DONORS=args.donors.resolve()
    SOURCE_LOCK=args.source_lock.resolve()
    if any(not 0<f<=1 for f in args.fractions):p.error('fractions must be in (0,1]')
    if any(m<=0 or not math.isfinite(m) for m in args.mass_multipliers):p.error('mass multipliers must be positive')
    run_job(args)
if __name__=='__main__':main()
