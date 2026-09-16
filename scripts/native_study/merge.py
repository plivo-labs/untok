"""Merge final native bank winners and audit their joint behavior before packaging.

This script never fits scores, chooses a new bank, changes quotas, reads reserve
contents, or creates a production bundle. Cross-job maximum scores are provisional
until all joint gates pass. Failed audits produce pending artifacts for explicit
bank reoptimization, not silently dropped or filler tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.unigram_fit import Lattice, _trie, build_model
import untok.unigram_fit as fitter_module
FITTER_PATH = Path(fitter_module.__file__)

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
ALPHABETS = REPO/'configs/native-study-alphabets.json'
BASE = None
BASE_SHA = 'ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291'
DATA_SHA = 'b31dfd0d45542614ab51bb9c49621d2bbb100747bfe8f93f83a28df599973188'
GROUPS = {'deva': (['hi','mr','brx','doi','kok','mai','ne','sa','sd'], 1400), 'bengali': (['as','bn'], 500)}
GROUPS.update({l: ([l], 500) for l in ['gu','kn','ml','or','pa','ta','te','ks','ur','mni','sat']})
JOBS = {g: [g] for g in GROUPS if g not in ['ks','ur']}
JOBS['arabic'] = ['ks','ur']
LANGS = [l for languages, _ in GROUPS.values() for l in languages]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
def read(path): return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as f:
        for i, line in enumerate(f, 1):
            if line.strip(): yield i, json.loads(line)


def prefix_check(base, model):
    if len(model.pieces) < len(base.pieces): raise ValueError('Native pieces removed')
    for i, p in enumerate(base.pieces):
        if p.SerializeToString() != model.pieces[i].SerializeToString(): raise ValueError(f'Native message changed at ID {i}')
    a, b = pb.ModelProto(), pb.ModelProto(); a.CopyFrom(base); b.CopyFrom(model)
    for obj in [a,b]: obj.ClearField('pieces'); obj.trainer_spec.ClearField('vocab_size')
    if a.SerializeToString() != b.SerializeToString(): raise ValueError('Native model metadata changed')
    if model.trainer_spec.vocab_size != len(model.pieces): raise ValueError('Trainer vocabulary size mismatch')
    low, high = min(p.score for p in base.pieces if p.type==1), max(p.score for p in base.pieces if p.type==1)
    seen = {p.piece for p in base.pieces}
    for p in model.pieces[len(base.pieces):]:
        if p.piece in seen or p.type != 1 or not low <= p.score <= high: raise ValueError('Invalid appended physical piece')
        seen.add(p.piece)
    return {'native_entries':len(base.pieces), 'native_piece_messages_unchanged':True,
            'native_metadata_unchanged':True, 'normal_score_extrema':[low,high]}


def policy_inputs(base):
    cfg_path = REPO/'configs/build.json'
    cfg = read(cfg_path)
    latin_path = REPO/'configs'/cfg['latin_characters']
    hindi_path = REPO/'configs'/cfg['hindi_pieces']
    alphabet_path = ALPHABETS
    if sha(latin_path) != cfg['latin_sha256'] or sha(hindi_path) != cfg['hindi_sha256']:
        raise ValueError('Approved Latin/Hindi lists changed from pinned configuration')
    latin = {line.split('\t')[1] for line in latin_path.read_text().splitlines() if line}
    hindi = {line.split('\t')[0] for line in hindi_path.read_text().splitlines() if line}
    if len(latin)!=190 or len(hindi)!=103 or any(len(p)!=1 for p in latin): raise ValueError('Approved inventory count/type mismatch')
    norm = spm.SentencePieceNormalizer(model_proto=base.SerializeToString())
    shared = {c for row in cfg['required_characters'] if row['language']=='und'
              for c in norm.normalize(row['character']) if c!='▁' and not c.isspace()}
    alphabets = {r['language']:set(r['standard_plus_current_required']['character_list']) for r in read(alphabet_path)['profiles']}
    if set(alphabets)!=set(LANGS): raise ValueError('Pinned alphabet profiles differ from intended 22')
    reasons = defaultdict(list)
    for p in latin: reasons[p].append({'kind':'approved_rare_latin_character', 'source':str(latin_path)})
    for p in shared: reasons[p].append({'kind':'normalized_global_required_character', 'source':str(cfg_path)})
    for lang, chars in alphabets.items():
        for p in chars: reasons[p].append({'kind':'pinned_standard_alphabet', 'language':lang, 'source':str(alphabet_path)})
    for p in hindi: reasons[p].append({'kind':'approved_hindi_piece', 'source':str(hindi_path)})
    return {'latin':latin, 'hindi':hindi, 'shared':shared, 'alphabets':alphabets,
            'reasons':reasons, 'hashes':{str(p):sha(p) for p in [cfg_path,latin_path,hindi_path,alphabet_path]}}


def common_reference_choice(results, reference, tail=1.02):
    eligible = []
    for r in results:
        if r.get('fraction') != 1 or not r.get('gates') or not all(r['gates'].values()): continue
        if not r.get('development') or set(r['development']) != set(reference): continue
        if all(m['unknown_tokens'] <= reference[l]['unknown_tokens'] and
               m['p95_record_tokens_per_character'] <= reference[l]['p95_record_tokens_per_character']*tail
               for l,m in r['development'].items()): eligible.append(r)
    return min(eligible, key=lambda r:(sum(m['tokens_per_normalized_character'] for m in r['development'].values())/len(r['development']),r['name'])) if eligible else None


def load_winners(fits, base, data_sha, pins):
    providers = defaultdict(list); all_selected = []; winners = {}; job_models = {}; pending = []
    native = {p.piece:p for p in base.pieces}
    for job in sorted(JOBS):
        path = fits/job/'final-choice.json'
        if not path.exists(): pending.append(job)
    if pending: return None, {'missing_final_choices':pending}
    for job, groups in sorted(JOBS.items()):
        folder=fits/job; choice=read(folder/'final-choice.json'); protocol=read(folder/'protocol.json'); results=read(folder/'results.json')
        if not choice.get('winner') or choice.get('purpose')!='final_selection' or choice.get('no_passing_candidate') or choice.get('reference_missing'):
            raise ValueError('Job has no accepted full-data winner: '+job)
        name=choice['winner']
        if Path(name).name != name: raise ValueError('Unexpected winner path')
        if protocol['accepted_data_manifest_sha256']!=data_sha or protocol['native_sha256']!=BASE_SHA:
            raise ValueError('Job was fitted on another data/native freeze: '+job)
        for path,digest in protocol.get('mandatory_policy_file_sha256',{}).items():
            if pins.get(str(Path(path).resolve()))!=digest:raise ValueError('Job required inventory differs from current pinned policy: '+path)
        if protocol.get('native_fit_code_sha256')!=sha(FITTER_PATH):
            raise ValueError('Joint lattice audit implementation differs from fitted implementation')
        if protocol['groups']!=groups or protocol['budgets']!={g:GROUPS[g][1] for g in groups}:
            raise ValueError('Job quotas changed: '+job)
        reference=next((r for r in results if r['name']==choice['common_reference']),None)
        if reference is None: raise ValueError('Main common development reference missing: '+job)
        references=[r for r in results if r.get('fraction')==1 and r.get('seed')==0 and r.get('family')=='fresh' and r.get('development')]
        expected_reference=min(references,key=lambda r:(r['added_score_mass'],r['name'])) if references else None
        if reference!=expected_reference: raise ValueError('Main reference does not match pinned smallest-mass seed0 fresh policy')
        chosen=common_reference_choice(results,reference['development'])
        if chosen is None or chosen['name']!=name: raise ValueError('Stored choice disagrees with common-reference selection: '+job)
        trial=folder/name; result=read(trial/'result.json'); report=read(trial/'fit-report.json'); selected=read(trial/'selected-pieces.json')
        if result!=chosen: raise ValueError('Winner result differs from job result list')
        if result['fraction']!=1 or not all(result['gates'].values()) or report['status']!='passed': raise ValueError('Winner failed required gates')
        if set(result['artifact_sha256'])!={'tokenizer.model','selected-pieces.json','fit-report.json'}:
            raise ValueError('Incomplete winner artifact integrity manifest')
        for filename,digest in result['artifact_sha256'].items():
            if sha(trial/filename)!=digest: raise ValueError('Winner artifact changed: '+str(trial/filename))
        model=pb.ModelProto(); model.ParseFromString((trial/'tokenizer.model').read_bytes()); prefix_check(base,model)
        if sha(trial/'tokenizer.model')!=result['model_sha256']: raise ValueError('Winner model digest mismatch')
        bypiece={p.piece:p for p in model.pieces}; counts=Counter(g for item in selected for g in item['groups'])
        if dict(counts)!={g:GROUPS[g][1] for g in groups}: raise ValueError('Selected membership quotas differ from protocol')
        if len({r['piece'] for r in selected})!=len(selected): raise ValueError('Repeated selected physical piece in a job')
        appended=set(bypiece)-set(native)
        if appended!={r['piece'] for r in selected if r['piece'] not in native}: raise ValueError('Model additions do not equal selected nonnative strings')
        for item in selected:
            p=item['piece']
            if p not in bypiece or abs(bypiece[p].score-float(item['score']))>1e-6: raise ValueError('Selected score differs from persisted model')
            entry=dict(item,job=job,trial=name,source_model_sha256=result['model_sha256'])
            all_selected.append(entry)
            if p not in native: providers[p].append(entry)
        for path in [folder/'final-choice.json',folder/'protocol.json',folder/'results.json',trial/'result.json',trial/'fit-report.json',trial/'selected-pieces.json',trial/'tokenizer.model']:
            pins[str(path.resolve())]=sha(path)
        winners[job]={'winner':name,'common_reference':reference['name'],'common_reference_development':reference['development'],
                      'result':result,'groups':groups,'selected_memberships':dict(counts)}
        job_models[job]=model
    return {'providers':providers,'selected':all_selected,'winners':winners,'job_models':job_models},None


def merge_inventory(base, banks, approved):
    native={p.piece for p in base.pieces}; additions={}; provenance={}; conflicts=[]
    for piece, providers in sorted(banks['providers'].items()):
        scores=[float(p['score']) for p in providers]
        additions[piece]=max(scores)
        required=list(approved['reasons'].get(piece,[]))
        required += [{'kind':'bank_protected_piece','job':r['job'],'groups':r['mandatory_in']} for r in providers if r.get('mandatory_in')]
        provenance[piece]={'owners':sorted({p['job'] for p in providers}),
                           'groups':sorted({g for p in providers for g in p['groups']}),
                           'fit_providers':providers,'required_reasons':required,'coverage_extra':False}
        if len(providers)>1:
            conflicts.append({'piece':piece,'providers':[{'job':p['job'],'groups':p['groups'],'persisted_score':p['score']} for p in providers],
                              'scores_differ':len(set(scores))>1,'provisional_merged_score':max(scores)})
    coverage=set(approved['shared'])|set(approved['latin'])|set().union(*approved['alphabets'].values())
    extras=coverage-native-set(additions)
    for piece in sorted(extras):
        if len(piece)!=1: raise ValueError('Coverage extras must be individual characters')
        additions[piece]=-32.
        provenance[piece]={'owners':['global_coverage'],'groups':[], 'fit_providers':[],
                           'required_reasons':list(approved['reasons'][piece]),'coverage_extra':True}
    if not approved['hindi'] <= native|set(additions): raise ValueError('Approved Hindi pieces missing from selected banks')
    if not coverage <= native|set(additions): raise ValueError('Required character coverage missing')
    model=build_model(base,additions); structure=prefix_check(base,model)
    # Record persisted float32 values, not higher precision incidental JSON values.
    persisted={p.piece:float(p.score) for p in model.pieces[len(base.pieces):]}
    return model,persisted,provenance,{'cross_job_shared_strings':conflicts,'coverage_extra_characters':sorted(extras),
                                    'coverage_extra_count':len(extras),'structure':structure}


def strict_dominance(model, new_pieces):
    scores={p.piece:float(p.score) for p in model.pieces if p.type==1}; trie=_trie(scores)
    unknown=min(scores.values())-10.; failures={}; margins={}
    for piece in sorted(new_pieces):
        alternative,path=Lattice(piece,trie,unknown).best(scores,excluded=piece)
        margin=alternative-scores[piece]
        margins[piece]=margin
        if margin>1e-5 and path and all(p is not None for p in path):
            failures[piece]={'piece_score':scores[piece],'alternative_score':alternative,
                             'alternative':path,'strict_margin':margin}
    return failures,margins


def stream_usage(proc, data, declared, new_ids, pins, batch_size=256):
    counts=Counter(); weighted=Counter(); first={}; source_counts=defaultdict(Counter); source_weighted=defaultdict(Counter)
    source_rows=defaultdict(Counter); totals=defaultdict(Counter)
    for lang in LANGS:
        path=data/'train'/f'{lang}.jsonl'; digest=declared['files'][f'train/{lang}.jsonl']
        if sha(path)!=digest: raise ValueError('Training input changed: '+lang)
        pins[str(path.resolve())]=digest; batch=[]
        def encode_batch():
            if not batch:return
            ids_batch=proc.encode([r['text'] for _,r in batch],out_type=int,num_threads=2)
            for (line,row),ids in zip(batch,ids_batch):
                key=(lang,row['source']); weight=float(row['weight']); local=Counter(ids)
                totals[key]['rows']+=1; totals[key]['tokens']+=len(ids); totals[key]['unknown_tokens']+=local[proc.unk_id()]
                for idx,n in local.items():
                    if idx not in new_ids:continue
                    counts[idx]+=n; weighted[idx]+=n*weight; source_counts[key][idx]+=n
                    source_weighted[key][idx]+=n*weight; source_rows[key][idx]+=1
                    first.setdefault(idx,{'kind':'frozen_training_record','path':str(path),'line':line,'record_id':row.get('record_id'),
                                          'native_id':idx,'occurrences':n})
            batch.clear()
        for line,row in rows(path):
            if row['language']!=lang:raise ValueError('Training language metadata mismatch')
            batch.append((line,row))
            if len(batch)>=batch_size:encode_batch()
        encode_batch()
    return counts,weighted,first,source_counts,source_weighted,source_rows,totals


def witness(proc,piece,first):
    idx=proc.piece_to_id(piece)
    if proc.id_to_piece(idx)!=piece:return None
    if idx in first:return first[idx]
    body=piece.removeprefix('▁')
    contexts=[body,' '+body,body+' ','a '+body+' b','a'+body+'a','अ'+body+'अ','अ '+body+' अ']
    for raw,ids in zip(contexts,proc.encode(contexts,out_type=int,num_threads=2)):
        if idx in ids:return {'kind':'approved_literal_context','input':raw,'native_ids':ids,'pieces':[proc.id_to_piece(i) for i in ids]}
    return None


def development_metrics(proc,data,declared,pins):
    result={}
    for lang in LANGS:
        path=data/'dev'/f'{lang}.jsonl';digest=declared['files'][f'dev/{lang}.jsonl']
        if sha(path)!=digest:raise ValueError('Development input changed: '+lang)
        pins[str(path.resolve())]=digest;stats=[];batch=[]
        def encode_batch():
            if not batch:return
            encoded=proc.encode([r['text'] for r in batch],out_type=int,num_threads=2)
            for row,ids in zip(batch,encoded):
                # The frozen manifest's old wording says raw_text, but the
                # actual contract is lexical text after source tag cleanup.
                chars=max(1,len(proc.normalize(row['text'])))
                stats.append((len(ids),ids.count(proc.unk_id()),chars))
            batch.clear()
        for _,row in rows(path):
            batch.append(row)
            if len(batch)==256:encode_batch()
        encode_batch()
        ratios=sorted(t/c for t,u,c in stats); lengths=sorted(t for t,u,c in stats)
        result[lang]={'records':len(stats),'tokens':sum(t for t,u,c in stats),'unknown_tokens':sum(u for t,u,c in stats),
                      'normalized_characters':sum(c for t,u,c in stats),
                      'tokens_per_normalized_character':sum(t for t,u,c in stats)/sum(c for t,u,c in stats),
                      'p95_record_tokens_per_character':ratios[int(.95*(len(ratios)-1))],
                      'p95_record_tokens':lengths[int(.95*(len(lengths)-1))]}
    return result


def run(args):
    data=args.data.resolve(); fits=args.fits.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Use a new audit output directory; existing artifacts are immutable')
    if sha(data/'manifest.json')!=args.accept_data_sha256:raise ValueError('Data manifest not explicitly accepted')
    declared=read(data/'manifest.json'); base=pb.ModelProto();base.ParseFromString(BASE.read_bytes())
    if sha(BASE)!=BASE_SHA or declared['native_base_sha256']!=BASE_SHA:raise ValueError('Wrong native base')
    pins={str(BASE.resolve()):BASE_SHA,str((data/'manifest.json').resolve()):args.accept_data_sha256}
    pins[str(FITTER_PATH.resolve())]=sha(FITTER_PATH)
    approved=policy_inputs(base);pins.update(approved['hashes'])
    banks,pending=load_winners(fits,base,args.accept_data_sha256,pins)
    if pending:
        out.mkdir(parents=True);write(out/'pending.json',{'status':'waiting_for_final_bank_choices',**pending,'reserve_content_read':False})
        return {'status':'waiting_for_final_bank_choices',**pending}
    model,additions,provenance,merge_report=merge_inventory(base,banks,approved)
    proc=spm.SentencePieceProcessor(model_proto=model.SerializeToString()); new_ids=set(range(len(base.pieces),len(model.pieces)))
    counts,weighted,first,by_source,weighted_source,source_rows,totals=stream_usage(proc,data,declared,new_ids,pins)
    witnesses={};missing_required=[];unused_optional=[]
    for piece in additions:
        idx=proc.piece_to_id(piece);required=provenance[piece]['required_reasons']
        if required:
            w=witness(proc,piece,first)
            if w:witnesses[piece]=w
            else:missing_required.append(piece)
        elif not counts[idx]:unused_optional.append(piece)
    approved_witnesses={};approved_failures={}
    for name,pieces in [('hindi103',approved['hindi']),('latin190',approved['latin']),('standard_alphabets',set().union(*approved['alphabets'].values())),('global_required_characters',approved['shared'])]:
        okay={};failed=[]
        for p in sorted(pieces):
            w=witness(proc,p,first)
            if w:okay[p]=w
            else:failed.append(p)
        approved_witnesses[name]=okay;approved_failures[name]=failed
    dominated,margins=strict_dominance(model,additions)
    dev=development_metrics(proc,data,declared,pins); comparisons=[]
    for job,info in banks['winners'].items():
        for lang,baseline in info['result']['development'].items():
            current=dev[lang]; unknown_ok=current['unknown_tokens']<=baseline['unknown_tokens']
            tail_ok=current['p95_record_tokens_per_character']<=baseline['p95_record_tokens_per_character']*1.02
            reference=info['common_reference_development'][lang]
            reference_tail_ok=current['p95_record_tokens_per_character']<=reference['p95_record_tokens_per_character']*1.02
            comparisons.append({'job':job,'language':lang,'winner':info['winner'],'baseline':baseline,'merged':current,
                                'common_reference':reference,'unknown_nonincrease':unknown_ok,
                                'p95_within_2_percent_of_winner':tail_ok,'p95_within_2_percent_of_common_reference':reference_tail_ok})
    gates={'native_messages_metadata_scores_preserved':True,'group_membership_quotas_preserved':True,
           'all_required_new_pieces_have_actual_encoding_witness':not missing_required,
           'approved_hindi103_latin190_alphabet_global_witnesses':not any(approved_failures.values()),
           'all_optional_new_pieces_used_on_full_training_records':not unused_optional,
           'no_strictly_dominated_new_pieces':not dominated,
           'joint_development_nonincrease_unknown_and_bounded_tail':all(r['unknown_nonincrease'] and r['p95_within_2_percent_of_winner'] and r['p95_within_2_percent_of_common_reference'] for r in comparisons)}
    passed=all(gates.values())
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Pinned input changed during joint audit: '+path)
    out.mkdir(parents=True)
    model_path=out/('merged-audited.model' if passed else 'merged-pending.model');model_path.write_bytes(model.SerializeToString())
    usage_path=out/'piece-usage-matrix.jsonl'
    with usage_path.open('w') as f:
        for p in sorted(additions):
            idx=proc.piece_to_id(p)
            entries=[{'language':lang,'source':source,'occurrences':counter[idx],'weighted_occurrences':weighted_source[lang,source][idx],
                      'records_using_piece':source_rows[lang,source][idx]} for (lang,source),counter in sorted(by_source.items()) if counter[idx]]
            f.write(json.dumps({'piece':p,'native_id':idx,'public_id':idx+2,'score':additions[p],
                                'training_occurrences':counts[idx],'weighted_training_occurrences':weighted[idx],
                                'required_reasons':provenance[p]['required_reasons'],'usage':entries,
                                'literal_alternative_margin':margins[p]},ensure_ascii=False,sort_keys=True)+'\n')
    counts_by_group={g:sum(g in r['groups'] for r in banks['selected']) for g in GROUPS}
    native_strings={x.piece for x in base.pieces}
    selection={'schema_version':1,'status':'joint_audits_passed_candidate' if passed else 'pending_reoptimization',
               'base_tokenizer_sha256':BASE_SHA,'data_manifest_sha256':args.accept_data_sha256,
               'native_model_entries':len(base.pieces),'native_normalizer_sha256':hashlib.sha256(base.normalizer_spec.SerializeToString()).hexdigest(),
               'ordering':'Unicode lexical order of physical addition strings; native prefix retained exactly',
               'ordered_additions':sorted(additions),
               'additions':[{'piece':p,'score':additions[p],**provenance[p],
                             'training_occurrences':counts[proc.piece_to_id(p)],'weighted_training_occurrences':weighted[proc.piece_to_id(p)],
                             'required_encoding_witness':witnesses.get(p)} for p in sorted(additions)],
               'groups':{g:{'languages':GROUPS[g][0],'budget':GROUPS[g][1],'selected_memberships':counts_by_group[g],
                            'selected_native_overlap':sum(g in r['groups'] and r['piece'] in native_strings for r in banks['selected'])} for g in GROUPS},
               'physical_new_pieces':len(additions),'physical_native_pieces':len(model.pieces),'public_ids_including_pad_blank':len(model.pieces)+2,
               'score_policy':'Persisted bank scores unchanged except one shared physical cross-job string provisionally uses max provider score; accepted only if joint gates pass. Coverage-only extras fixed at -32. No old research fitted scores used.',
               'source_fit_file_hashes':{p:d for p,d in pins.items() if str(fits) in p},'all_input_hashes':pins,
               'joint_gates':gates,'joint_model_sha256':sha(model_path),'piece_usage_matrix_sha256':sha(usage_path),
               'scope_limit':'Tokenizer candidate only. No checkpoint migration, audio validation or production packaging is performed here.'}
    selection_path=out/'selection.pending.json';write(selection_path,selection)
    report={'status':'passed' if passed else 'failed_joint_gates','gates':gates,**merge_report,
            'group_budgets':{g:GROUPS[g][1] for g in GROUPS},'group_selected_counts':counts_by_group,
            'training_rows':sum(c['rows'] for c in totals.values()),
            'source_language_totals':[{'language':l,'source':s,**dict(c)} for (l,s),c in sorted(totals.items())],
            'unused_optional_pieces':unused_optional,'missing_required_encoding_witnesses':missing_required,
            'approved_inventory_witness_failures':approved_failures,'approved_inventory_witnesses':approved_witnesses,
            'strict_dominance_failures':dominated,'dominance_tolerance':1e-5,
            'dominance_definition':'A NORMAL-piece-only literal replacement excluding the candidate itself has score greater by more than 1e-5. All new pieces checked. This is a literal dominance proof, not a claim about all USER_DEFINED/control contexts.',
            'development_comparisons':comparisons,'reserve_content_read':False,'all_input_hashes':pins,
            'repair_required_if_failed':'Reoptimize or replace affected optional pieces within their existing quotas, then rerun joint audits. Do not silently drop, pad, change required lists or adjust native scores.',
            'weighting_clarification':'Weights use native.normalize(row[text]) after source annotation cleanup, not preserved raw_text. No rebalancing here.'}
    write(out/'joint-audit.json',report)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Pinned input changed during joint audit: '+path)
    if passed:
        selection_path.rename(out/'selection.json');selection_path=out/'selection.json'
    write(out/'receipt.json',{'status':report['status'],'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                               'script_sha256':sha(__file__),'reserve_content_read':False})
    return {'status':report['status'],'output':str(out),'gates':gates,'new_pieces':len(additions),
            'global_coverage_extras':merge_report['coverage_extra_count'],'selection_sha256':sha(selection_path)}


def main():
    global REPO, ALPHABETS, BASE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='Run only once all final bank choices exist')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--alphabets',type=Path,default=ALPHABETS)
    p.add_argument('--fits',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accept-data-sha256',required=True)
    args=p.parse_args()
    REPO=args.repo.resolve(); ALPHABETS=args.alphabets.resolve(); BASE=args.base.resolve()
    if not args.run:
        plan={'status':'planned_pending_final_bank_choices','jobs':JOBS,'data_manifest_sha256':args.accept_data_sha256,
              'required_final_choice_paths':[str(args.fits/job/'final-choice.json') for job in sorted(JOBS)],
              'policy':'Merge fixed native winners; add required coverage characters at -32; max shared cross-job scores provisional until all gates pass',
              'audits':['native full message/metadata prefix','per-group membership budgets','all-training actual SentencePiece usage by source/language','all-new literal strict dominance excluding itself','required literal/training encoding witnesses','103 Hindi and 190 Latin witnesses','per-language merged dev unknown/tail against owning job winner'],
              'failure_action':'Pending artifact and reoptimization within quotas, never silent drop or filler',
              'reserve_content_read':False,'production_packaging_performed':False}
        print(json.dumps(plan,sort_keys=True));return
    started=time.monotonic();result=run(args);result['elapsed_seconds']=round(time.monotonic()-started,2);print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
