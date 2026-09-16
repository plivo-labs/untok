"""Freeze native-normalized audit boundaries while preserving selected source text.

No network requests, model fitting, or corpus downloads occur here. The build
command requires hash-verified intake files and all 22 IndicVoices profiles.
Official reserve text is never printed. Input text is never normalized in place.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time
import unicodedata
import zlib

import numpy as np
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

HERE = Path(__file__).resolve().parent
LANGS = ('hi mr brx doi kok mai ne sa sd as bn gu kn ml or pa ta te ks ur mni sat').split()
BASE = None
EXPECTED_BASE = 'ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291'
SEED = '20260907-frozen-v1'
IV = 'ai4bharat/IndicVoices'
META = 'facebook/omnilingual-asr-corpus'
URDU = 'ASLP-lab/UrduSpeech'
SPRING = 'SPRING-INX-R1'
VAANI = 'ARTPARK-IISc/Vaani'
SCRIPT = {l: 'DEVANAGARI' for l in 'hi mr brx doi kok mai ne sa sd'.split()}
SCRIPT.update(dict(as_='BENGALI', bn='BENGALI', gu='GUJARATI', kn='KANNADA',
                   ml='MALAYALAM', or_='ORIYA', pa='GURMUKHI', ta='TAMIL',
                   te='TELUGU', ks='ARABIC', ur='ARABIC', mni='MEETEI MAYEK', sat='OL CHIKI'))
SCRIPT['as'] = SCRIPT.pop('as_'); SCRIPT['or'] = SCRIPT.pop('or_')


def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def stable(value): return sha_bytes((SEED + '\0' + value).encode())


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def json_rows(path):
    with Path(path).open() as f:
        for n, line in enumerate(f, 1):
            if line.strip(): yield n, json.loads(line)


def canonical_origin(row, kind):
    source = str(row.get('dataset') or row.get('source') or 'unknown')
    domain = str(row.get('domain') or '')
    if source in ('IndicVoices', IV): return IV
    if source == META: return META
    if source == URDU: return URDU
    if source.startswith('SPRINGLab/SPRING_INX_'): return SPRING
    if 'ARTPARK-IISc/Vaani' in source: return VAANI
    if kind == 'written':
        if source == 'Dakshina': return 'google/Dakshina'
        if 'IndicCorp' in source or domain == 'indiccorp': return 'ai4bharat/IndicCorpV2'
        if source == 'Manually-Collected': return 'Bhasha-Abhijnaanam/Manually-Collected'
        if domain == 'wikimedia': return 'Wikimedia'
        if domain == 'udhr': return 'Unicode/UDHR'
        if domain == 'sindhi_blog': return 'rakeshlakhani.wordpress.com'
    return source


@lru_cache(maxsize=8192)
def letter_script(ch):
    if not unicodedata.category(ch).startswith('L'): return None
    name = unicodedata.name(ch, 'UNASSIGNED')
    for script in set(SCRIPT.values()) | {'LATIN'}:
        if script in name: return script
    return 'OTHER'


def script_audit(text, language):
    letters = Counter(letter_script(c) for c in text if letter_script(c))
    target = letters[SCRIPT[language]]
    latin = letters['LATIN']
    other = sum(letters.values()) - target - latin
    if target:
        status = 'target_script_present'
    elif latin and not other:
        status = 'english_only_retained_as_code_switch_context'
    elif not letters:
        status = 'nonletter_context'
    else:
        status = 'wrong_script_no_target_letters'
    return {'status': status, 'target_letters': target, 'latin_letters': latin,
            'other_script_letters': other, 'letter_scripts': dict(sorted(letters.items())),
            'zwnj': text.count('\u200c'), 'zwj': text.count('\u200d'),
            'replacement_characters': text.count('\ufffd')}


def grouping_keys(row, origin, kind, record_id, language):
    raw = row.get('raw') or {}
    keys = []
    status = 'verified_source_identifiers_not_independently_verified_people'
    speaker = row.get('speaker_id')
    recording = row.get('recording_id') or row.get('recording_path')
    if origin == IV:
        speaker = row.get('speaker_group')
        if speaker: keys.append('speaker:' + str(speaker))
        if recording: keys.append('recording:' + origin + ':' + str(recording))
        if not speaker: status = 'limited_missing_speaker_identifier'
    elif origin == META:
        if speaker: keys.append('speaker:' + origin + ':' + str(speaker))
        if row.get('prompt_id'): keys.append('prompt:' + origin + ':' + str(row['prompt_id']))
        if recording: keys.append('recording:' + origin + ':' + str(recording))
        status = 'source_speaker_and_shared_prompt_components'
    elif origin == URDU:
        domain = row.get('style') or str(row.get('source_config', '')).split('/')[-1]
        keys.append('domain:' + origin + ':' + str(domain))
        status = 'limited_generic_speaker_labels_not_verified_grouped_by_domain'
    elif origin == SPRING:
        # No unsupported parsing of utterance identifiers into speakers/sessions.
        keys.append('source_corpus:' + origin + ':' + language)
        status = 'limited_no_speaker_or_session_entire_language_source_grouped'
    elif origin == VAANI:
        # The acquired published transcription subset omits speaker identity.
        # Parent policy admits it only as a grouped training supplement. Its
        # prompts/regions are preserved as provenance, never invented speakers.
        speaker = None
        keys.append('source_corpus:' + origin + ':' + language)
        status = 'limited_no_speaker_identity_train_only_entire_language_source_grouped'
    elif kind == 'written':
        document = row.get('document_id') or row.get('source_group')
        if document:
            keys.append('document:' + origin + ':' + str(document))
            status = 'legacy_source_group_document_identity_not_always_verifiable'
        else:
            keys.append('source_corpus:' + origin + ':' + language)
            status = 'limited_missing_document_identifier_entire_source_grouped'
    else:
        if speaker: keys.append('speaker:' + origin + ':' + str(speaker))
        if recording: keys.append('recording:' + origin + ':' + str(recording))
    if not keys:
        keys = ['source_corpus:' + origin + ':' + language]
        status = 'limited_missing_group_metadata_entire_source_grouped'
    return sorted(set(keys)), status, speaker, recording


def prepare_row(row, kind, role, input_file, line, proc, annotation_policy=None):
    lang = row['language']
    if lang not in SCRIPT: raise ValueError('Unsupported language metadata: ' + str(lang))
    text = row.get('text')
    if not isinstance(text, str): raise ValueError('Missing string text at input row ' + str(line))
    origin = canonical_origin(row, kind)
    source_policy = (annotation_policy or {}).get('sources', {}).get(origin, {})
    source_text_path = source_policy.get('select_raw_text_path')
    intake_text = text
    if source_text_path:
        selected = row
        for key in source_text_path.split('.'):
            selected = selected.get(key) if isinstance(selected, dict) else None
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError(f'Required selected source field {source_text_path} is empty/missing at {input_file}:{line}; no fallback performed')
        text = selected
    selected_raw = text
    annotation_counts = {}
    # Source annotations are separate from, and do not modify, normalizer 1A.
    for marker in source_policy.get('replace_with_space', []):
        if not isinstance(marker, str) or not marker or not (marker.startswith('<') and marker.endswith('>') or marker.startswith('[') and marker.endswith(']')):
            raise ValueError('Invalid explicit annotation marker policy')
        count = text.count(marker)
        if count: annotation_counts[marker] = count; text = text.replace(marker, ' ')
    previous_counts = Counter(row.get('source_annotation_replacements') or {})
    previous_total = max(int(row.get('marker_removed_count') or 0), sum(previous_counts.values()))
    # Reselecting the original raw field replays cleanup, so do not count the
    # same markers twice. Otherwise these are successive cleanup stages.
    if source_text_path:
        cumulative_counts = previous_counts | Counter(annotation_counts)
        cumulative_total = max(previous_total, sum(cumulative_counts.values()))
    else:
        cumulative_counts = previous_counts + Counter(annotation_counts)
        cumulative_total = previous_total + sum(annotation_counts.values())
    partial_transcript = bool(row.get('tokenizer_fit_only_where_annotations_removed')) or cumulative_total > 0
    record_id = str(row.get('record_id') or row.get('id') or f'{input_file}:{line}')
    keys, group_status, speaker, recording = grouping_keys(row, origin, kind, record_id, lang)
    raw = row.get('raw') or {}
    norm = proc.normalize(text)
    fingerprint = re.sub('▁+', '▁', norm).strip('▁')
    result = dict(row)
    result.update(language=lang, source=origin, source_original=row.get('source'),
                  source_revision=row.get('source_revision') or row.get('revision'),
                  source_split=row.get('source_split') or row.get('official_split') or row.get('split'),
                  record_id=record_id, speaker_id=speaker, recording_id=recording,
                  district=row.get('district') or raw.get('district'),
                  state=row.get('state') or raw.get('state'),
                  style=row.get('style') or raw.get('scenario') or row.get('domain'),
                  text=text, raw_text=selected_raw if source_text_path else row.get('raw_text', selected_raw), corpus_kind=kind,
                  selected_raw_text_path=source_text_path or row.get('text_field') or 'text',
                  text_field=source_text_path.removeprefix('raw.') if source_text_path else row.get('text_field', 'text'),
                  source_field_changed_from_intake=bool(source_text_path) and selected_raw != intake_text,
                  intake_selected_text_sha256=sha_bytes(intake_text.encode()),
                  intake_text_field=row.get('text_field'),
                  source_selected_raw_text=selected_raw if source_text_path else row.get('source_selected_raw_text', selected_raw if annotation_counts else None),
                  source_annotation_replacements=dict(cumulative_counts),
                  marker_removed_count=cumulative_total,
                  tokenizer_fit_only_where_annotations_removed=partial_transcript,
                  marker_only_after_source_cleanup=partial_transcript and not text.strip(),
                  input_file=str(input_file), input_line=line,
                  normalized_sha256=sha_bytes(norm.encode()), normalized_character_count=len(norm),
                  dedup_sha256=sha_bytes(fingerprint.encode()),
                  group_metadata_status=group_status, script_audit=script_audit(text, lang),
                  raw_script_audit=script_audit(selected_raw, lang),
                  unrecognized_angle_marker_hashes={sha_bytes(tag.encode()): n for tag, n in Counter(re.findall(r'<[^<>]*>', text)).items()},
                  unmatched_angle_delimiters=any(c in re.sub(r'<[^<>]*>', '', text) for c in '<>'),
                  legacy_research_input=kind == 'written', _role=role, _norm=norm, _fingerprint=fingerprint, _keys=keys)
    # Preserve old dev as dev, including groups joined to it. Other old written
    # records have been observed in prior research and are never called unseen.
    result['_forced_dev'] = role == 'dev'
    result['_quarantine_unmatched_angles'] = bool(source_policy.get('quarantine_unmatched_angles'))
    result['_uid'] = sha_bytes((str(input_file) + '\0' + str(line) + '\0' + record_id).encode())
    return result


class UnionFind:
    def __init__(self, n): self.parent = list(range(n)); self.size = [1] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.size[a] < self.size[b]: a, b = b, a
        self.parent[b] = a; self.size[a] += self.size[b]


def removal(row, reason, match=None):
    # Deliberately no raw or normalized strings in removal/audit logs.
    r = {'uid': row['_uid'], 'record_id': row['record_id'], 'language': row['language'],
         'source': row['source'], 'role': row['_role'], 'normalized_sha256': row['normalized_sha256'],
         'dedup_sha256': row['dedup_sha256'],
         'reason': reason}
    if match is not None: r['matched_uid'] = match['_uid']
    return r


def assign_groups(rows):
    """Metadata forms components. Common transcript text never joins speakers."""
    uf = UnionFind(len(rows)); seen = {}
    for i, row in enumerate(rows):
        for key in row['_keys']:
            if key in seen: uf.union(i, seen[key])
            else: seen[key] = i
    comps = defaultdict(list)
    for i in range(len(rows)): comps[uf.find(i)].append(i)
    removed = []; eligible = []; component_report = []
    strata = defaultdict(list)
    for members in comps.values():
        keys = sorted({k for i in members for k in rows[i]['_keys']})
        gid = sha_bytes('\n'.join(keys).encode())
        reserve = any(rows[i]['_role'] == 'reserve' for i in members)
        for i in members:
            rows[i]['split_group_id'] = gid
            rows[i]['group_keys_sha256'] = [sha_bytes(k.encode()) for k in rows[i]['_keys']]
        if reserve:
            for i in members:
                if rows[i]['_role'] != 'reserve': removed.append(removal(rows[i], 'reserve_metadata_component_collision'))
                else: eligible.append(rows[i])
            continue
        signature = tuple(sorted({(rows[i]['language'], rows[i]['source']) for i in members}))
        forced = any(rows[i]['_forced_dev'] for i in members)
        strata[signature].append((gid, members, forced))
        component_report.append({'id': gid, 'rows': len(members), 'languages_sources': signature,
                                 'forced_legacy_dev': forced,
                                 'keys': len(keys), 'metadata_status': sorted({rows[i]['group_metadata_status'] for i in members})})
    for signature, groups in sorted(strata.items()):
        total = sum(len(m) for _, m, _ in groups)
        target = total * .10
        chosen = {gid for gid, _, forced in groups if forced}
        dev_rows = sum(len(m) for gid, m, _ in groups if gid in chosen)
        choices = sorted((g for g in groups if not g[2]), key=lambda g: stable(g[0]))
        # Closest attainable group-level allocation. A single giant component
        # remains whole; it is never split to manufacture a dev sample.
        for gid, members, _ in choices:
            if len(chosen) + 1 >= len(groups): continue
            if abs(dev_rows + len(members) - target) < abs(dev_rows - target):
                chosen.add(gid); dev_rows += len(members)
        if not chosen and len(groups) >= 2:
            gid, members, _ = min(choices, key=lambda g: (len(g[1]), stable(g[0])))
            chosen.add(gid); dev_rows += len(members)
        for gid, members, _ in groups:
            role = 'dev' if gid in chosen else 'train'
            for i in members: rows[i]['_role'] = role; eligible.append(rows[i])
    component_report.sort(key=lambda r: (-r['rows'], r['id']))
    return eligible, removed, {'component_count': len(comps), 'largest_nonreserve_components': component_report[:100],
                              'single_component_strata': [str(s) for s, g in sorted(strata.items()) if len(g) == 1],
                              'split_policy': 'Approximately 90/10 by deterministic metadata components, preserve legacy dev, never split a group'}


def priority(row):
    return ({'reserve': 0, 'dev': 1, 'train': 2}[row['_role']],
            0 if row['source'] == IV else 1, 0 if row['corpus_kind'] == 'speech' else 1,
            row['_uid'])


def exact_dedup(rows):
    seen = {}; kept = []; removed = []; reserve_duplicates = 0
    for row in sorted(rows, key=priority):
        key = row['dedup_sha256']
        if key in seen:
            if row['_role'] == 'reserve':
                seen[key].setdefault('reserve_equivalent_records', []).append(
                    {'uid': row['_uid'], 'record_id': row['record_id'], 'language': row['language'],
                     'input_file': row['input_file'], 'input_line': row['input_line'], 'relation': 'exact_boundary_collapsed_audit_fingerprint'})
                reserve_duplicates += 1
            removed.append(removal(row, 'exact_audit_fingerprint_duplicate', seen[key])); continue
        else: seen[key] = row
        kept.append(row)
    return kept, removed, reserve_duplicates


def grams(text): return set(text[i:i+5] for i in range(len(text)-4))


class NearIndex:
    """32 deterministic MinHash values, eight bands of four; exact verification.

    CRC32 maps grams only for approximate candidate discovery. Final Jaccard
    uses full Unicode character grams. No recall guarantee is claimed.
    """
    def __init__(self, max_candidates=5000):
        rng = np.random.default_rng(20260907)
        self.a = rng.integers(1, 2**32-1, size=32, dtype=np.uint64)
        self.b = rng.integers(0, 2**32-1, size=32, dtype=np.uint64)
        self.prime = np.uint64(4294967311)
        self.index = {}; self.rows = []; self.max_candidates = max_candidates
        self.stats = Counter()

    def signature(self, values):
        # uint64 wrapping is deterministic. It is an approximation, not a
        # cryptographic/minwise-independent permutation family.
        h = np.array([zlib.crc32(x.encode()) for x in values], dtype=np.uint64)
        sig = ((self.a[:, None] * h[None, :] + self.b[:, None]) % self.prime).min(axis=1).astype('<u4')
        return [(i, sig[i*4:i*4+4].tobytes()) for i in range(8)]

    def lookup(self, row):
        text = row['_fingerprint']
        if len(text) < 40: self.stats['below_minimum_length'] += 1; return None
        values = grams(text); keys = self.signature(values); candidates = set()
        for k in keys:
            bucket = self.index.get(k)
            if isinstance(bucket, int): candidates.add(bucket)
            elif bucket: candidates.update(bucket)
        if len(candidates) > self.max_candidates:
            candidates = set(sorted(candidates)[:self.max_candidates]); self.stats['candidate_capped_rows'] += 1
        self.stats['eligible_rows'] += 1
        match = None
        for i in sorted(candidates):
            prev = self.rows[i]
            # The ratio of set sizes is a strict upper bound on Jaccard.
            other = grams(prev['_fingerprint'])
            if min(len(values), len(other)) / max(len(values), len(other)) < .9: continue
            self.stats['exact_jaccard_comparisons'] += 1
            similarity = len(values & other) / len(values | other)
            if similarity >= .9:
                self.stats['verified_matches'] += 1
                match = prev, similarity
                break
        i = len(self.rows); self.rows.append(row)
        for k in keys:
            bucket = self.index.get(k)
            if bucket is None: self.index[k] = i
            elif isinstance(bucket, int): self.index[k] = [bucket, i]
            else: bucket.append(i)
        return match


def near_dedup(rows, max_candidates=5000):
    index = NearIndex(max_candidates); kept = []; removed = []; reserve_duplicates = 0
    representatives = {}
    for row in sorted(rows, key=priority):
        match = index.lookup(row)
        if match:
            other, score = match
            representative = representatives.get(other['_uid'], other)
            if row['_role'] == 'reserve':
                representative.setdefault('reserve_equivalent_records', []).append(
                    {'uid': row['_uid'], 'record_id': row['record_id'], 'language': row['language'],
                     'input_file': row['input_file'], 'input_line': row['input_line'],
                     'relation': 'near_boundary_collapsed_audit_fingerprint', 'matched_uid': other['_uid'], 'jaccard': score})
                representative['reserve_equivalent_records'].extend(row.get('reserve_equivalent_records', []))
                reserve_duplicates += 1
            representatives[row['_uid']] = representative
            record = removal(row, 'near_audit_fingerprint_duplicate', other)
            record['char5gram_jaccard'] = score; removed.append(record); continue
        representatives[row['_uid']] = row
        kept.append(row)
    stats = dict(index.stats)
    stats.update(method='32 deterministic MinHash hashes; 8 bands x 4; exact Unicode char5gram Jaccard >= 0.90 on boundary-collapsed audit fingerprints',
                 minimum_audit_fingerprint_characters=40, maximum_candidates_per_row=max_candidates,
                 recall='Approximate candidate discovery, not an exhaustive or zero-near-duplicate guarantee',
                 suppressed_within_reserve_near_duplicates=reserve_duplicates,
                 suppressed_variants_remain_indexed_for_lower_priority_conflict_checks=True)
    return kept, removed, stats


def assign_weights(rows):
    by_lang = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['_role'] == 'train': by_lang[r['language']][r['source']].append(r)
    report = {}
    for lang, sources in sorted(by_lang.items()):
        written = {s: rr for s, rr in sources.items() if rr[0]['corpus_kind'] == 'written'}
        speech = {s: rr for s, rr in sources.items() if rr[0]['corpus_kind'] == 'speech'}
        shares = {}
        wm = .2 if written and speech else (1. if written else 0.)
        sm = 1. - wm
        if written: shares.update({s: wm / len(written) for s in written})
        independents = {s: rr for s, rr in speech.items() if s != IV}
        if IV in speech:
            shares[IV] = sm * (.6 if independents else 1.)
        remaining = sm - shares.get(IV, 0.)
        if independents:
            importance = {s: math.sqrt(min(len({r['normalized_sha256'] for r in rr}), 3000)) for s, rr in independents.items()}
            total = sum(importance.values())
            shares.update({s: remaining * v / total for s, v in importance.items()})
        details = {}
        for source, rr in sorted(sources.items()):
            chars = sum(r['normalized_character_count'] for r in rr)
            if chars <= 0: raise ValueError('Zero training character mass for ' + lang + ':' + source)
            coef = shares[source] * 1e6 / chars
            for row in rr: row['weight'] = coef
            details[source] = {'kind': rr[0]['corpus_kind'], 'rows': len(rr),
                               'normalized_characters': chars, 'share': shares[source],
                               'weight_per_row': coef, 'weighted_character_mass': coef * chars}
        mass = sum(v['weighted_character_mass'] for v in details.values())
        if not math.isclose(mass, 1e6, abs_tol=1e-5): raise ValueError('Character mass mismatch')
        report[lang] = details
    for row in rows:
        if row['_role'] != 'train': row['weight'] = 1.0
    return report


def summarize(rows):
    groups = defaultdict(list)
    for r in rows: groups[(r['language'], r['_role'], r['source'])].append(r)
    result = []
    for (lang, role, source), rr in sorted(groups.items()):
        speaker_labels = {str(r['speaker_id']) for r in rr if r.get('speaker_id')}
        source_ids = {r.get('source_original') for r in rr if r.get('source_original')}
        result.append({'language': lang, 'role': role, 'source': source, 'origin_count': 1,
                       'source_repositories_or_urls': sorted(source_ids), 'rows': len(rr),
                       'unique_native_normalized_texts': len({r['normalized_sha256'] for r in rr}),
                       'unique_audit_fingerprints': len({r['dedup_sha256'] for r in rr}),
                       'source_speaker_labels': len(speaker_labels),
                       'speaker_labels_are_verified_individual_people': False,
                       'speaker_label_limit': 'Generic labels, not a people count' if source == URDU else 'Source identifiers, not independently reidentified people',
                       'explicit_recording_ids': len({r['recording_id'] for r in rr if r.get('recording_id')}),
                       'split_groups': len({r['split_group_id'] for r in rr}),
                       'districts': dict(Counter(str(r['district']) for r in rr if r.get('district'))),
                       'states': dict(Counter(str(r['state']) for r in rr if r.get('state'))),
                       'styles': dict(Counter(str(r['style']) for r in rr if r.get('style'))),
                       'group_metadata_status': dict(Counter(r['group_metadata_status'] for r in rr)),
                       'script_status': dict(Counter(r['script_audit']['status'] for r in rr)),
                       'normalized_characters': sum(r['normalized_character_count'] for r in rr),
                       'source_revisions': sorted({r['source_revision'] for r in rr if r.get('source_revision')})})
    return result


def transform(rows, max_candidates=5000):
    removals = []; eligible = []; input_scripts = Counter(); annotation_counts = Counter(); annotation_rows = Counter()
    for row in rows:
        status = row['script_audit']['status']; input_scripts[(row['language'], row['_role'], status)] += 1
        for marker, count in row['source_annotation_replacements'].items():
            annotation_counts[(row['language'], row['_role'], marker)] += count
            annotation_rows[(row['language'], row['_role'], marker)] += 1
        if row.get('marker_only_after_source_cleanup'):
            removals.append(removal(row, 'marker_only_after_source_cleanup'))
        elif row['_quarantine_unmatched_angles'] and row['unmatched_angle_delimiters']:
            removals.append(removal(row, 'unmatched_source_angle_annotation_quarantine'))
        elif row['_role'] != 'reserve' and (not row['_norm'] or status == 'wrong_script_no_target_letters'):
            removals.append(removal(row, 'empty_normalized_text' if not row['_norm'] else 'wrong_script_quarantine'))
        else: eligible.append(row)
    eligible, dropped, grouping = assign_groups(eligible); removals.extend(dropped)
    eligible, dropped, reserve_exact = exact_dedup(eligible); removals.extend(dropped)
    eligible, dropped, near = near_dedup(eligible, max_candidates); removals.extend(dropped)
    weights = assign_weights(eligible)
    # Independently verify cross-role metadata and exact hash disjointness.
    roles_by_group = defaultdict(set); roles_by_hash = defaultdict(set); roles_by_fingerprint = defaultdict(set)
    for r in eligible:
        roles_by_group[r['split_group_id']].add(r['_role'])
        roles_by_hash[r['normalized_sha256']].add(r['_role'])
        roles_by_fingerprint[r['dedup_sha256']].add(r['_role'])
    if any(len(v) > 1 for v in roles_by_group.values()): raise ValueError('Metadata group crossed final roles')
    if any(len(v) > 1 for v in roles_by_hash.values()): raise ValueError('Exact normalized text crossed final roles')
    if any(len(v) > 1 for v in roles_by_fingerprint.values()): raise ValueError('Exact audit fingerprint crossed final roles')
    return eligible, removals, {'grouping': grouping, 'near_duplicates': near,
                              'suppressed_within_reserve_exact_duplicates': reserve_exact,
                              'input_script_audit': [{'language': l, 'input_role': role, 'status': s, 'rows': c} for (l, role, s), c in sorted(input_scripts.items())],
                              'source_annotation_cleanup': [{'language': l, 'input_role': role, 'marker': marker, 'occurrences': n, 'affected_rows': annotation_rows[l, role, marker]} for (l, role, marker), n in sorted(annotation_counts.items())],
                              'raw_script_traits': [{'language': lang,
                                  'rows': sum(r['language'] == lang for r in rows),
                                  'zwnj_occurrences': sum(r['raw_script_audit']['zwnj'] for r in rows if r['language'] == lang),
                                  'zwj_occurrences': sum(r['raw_script_audit']['zwj'] for r in rows if r['language'] == lang),
                                  'replacement_characters': sum(r['raw_script_audit']['replacement_characters'] for r in rows if r['language'] == lang),
                                  'remaining_angle_marker_rows': sum(bool(r['unrecognized_angle_marker_hashes']) for r in rows if r['language'] == lang)} for lang in sorted({r['language'] for r in rows})],
                              'selected_source_field': {'changed_from_published_intake_rows': sum(r['source_field_changed_from_intake'] for r in rows),
                                  'missing_required_raw_field_fallbacks': 0,
                                  'text_paths': dict(Counter(r['selected_raw_text_path'] for r in rows))},
                              'weighting': weights,
                              'final_source_matrix': summarize(eligible),
                              'removals_by_reason': dict(Counter(r['reason'] for r in removals)),
                              'cross_role_metadata_groups': 0, 'cross_role_exact_normalized_hashes': 0, 'cross_role_exact_audit_fingerprints': 0}


def load_inputs(specs, proc, annotation_policy=None, input_root=None):
    records = []; inputs = {}; exclusions = Counter()
    for path, kind, role in specs:
        path = Path(path)
        if not path.exists(): raise FileNotFoundError(path)
        digest = sha_file(path); count = 0; accepted = 0
        for line, row in json_rows(path):
            count += 1
            identifying = ' '.join(str(row.get(k, '')) for k in ('source', 'domain', 'dataset', 'source_path'))
            if 'flores' in identifying.lower(): exclusions['FLORES_all_roles_excluded'] += 1; continue
            if canonical_origin(row, kind) == VAANI:
                va_policy = (annotation_policy or {}).get('sources', {}).get(VAANI, {})
                if not (va_policy.get('admission_approved') is True and va_policy.get('source_specific_text_policy') and va_policy.get('evidence')):
                    raise ValueError('Vaani is quarantined pending an explicitly approved source-specific annotation policy')
            if any(x in str(row.get('source_split', row.get('official_split', ''))).lower() for x in ('test', 'benchmark')) and row.get('source_split') != 'corpus_nonbenchmark':
                if role != 'reserve': raise ValueError('Benchmark/test split cannot enter training or dev: ' + str(path))
            label = path.resolve().relative_to(Path(input_root).resolve()) if input_root else path
            records.append(prepare_row(row, kind, role, label, line, proc, annotation_policy)); accepted += 1
        if digest != sha_file(path): raise ValueError('Input changed during read: ' + str(path))
        inputs[str(path.resolve())] = {'sha256': digest, 'rows': count, 'admitted_before_audit': accepted, 'kind': kind, 'role': role}
        companion = path.with_suffix('.manifest.json')
        if companion.exists():
            inputs[str(path.resolve())]['intake_manifest_path'] = str(companion.resolve())
            inputs[str(path.resolve())]['intake_manifest_sha256'] = sha_file(companion)
    return records, inputs, dict(exclusions)


def freeze(specs, out, base=None, langs=LANGS, max_candidates=5000, annotation_policy=None, input_root=None):
    out = Path(out)
    if out.exists(): raise FileExistsError('Immutable freeze output already exists: ' + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    model_bytes = Path(base).read_bytes(); model = pb.ModelProto(); model.ParseFromString(model_bytes)
    base_hash = sha_bytes(model_bytes)
    if base_hash != EXPECTED_BASE: raise ValueError('Native base hash differs from verified Nemotron checkpoint tokenizer')
    proc = spm.SentencePieceProcessor(model_proto=model_bytes)
    rows, inputs, exclusions = load_inputs(specs, proc, annotation_policy, input_root)
    rows, removals, audit = transform(rows, max_candidates)
    by_role_lang = defaultdict(list)
    for row in rows: by_role_lang[(row['_role'], row['language'])].append(row)
    for lang in langs:
        if not by_role_lang['train', lang]: raise ValueError('Empty final training profile: ' + lang)
        if not by_role_lang['dev', lang]: raise ValueError('Empty final development profile: ' + lang)
        if not by_role_lang['reserve', lang]: raise ValueError('Empty final reserve profile: ' + lang)
        if not any(r['source'] == IV for r in by_role_lang['train', lang]):
            raise ValueError('No IndicVoices training records remain after leakage/script filters: ' + lang)
    temp = Path(tempfile.mkdtemp(prefix=out.name + '.partial-', dir=out.parent))
    try:
        files = {}; counts = {}
        for role in ('train', 'dev', 'reserve'):
            (temp / role).mkdir()
            for lang in langs:
                rel = f'{role}/{lang}.jsonl'; path = temp / rel
                rr = sorted(by_role_lang[role, lang], key=lambda r: r['_uid'])
                with path.open('w') as f:
                    for row in rr:
                        public = {k: v for k, v in row.items() if not k.startswith('_')}
                        public.update(frozen_role=role, frozen_record_uid=row['_uid'])
                        f.write(json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
                files[rel] = sha_file(path); counts[rel] = len(rr)
        with (temp / 'exclusions.jsonl').open('w') as f:
            for row in sorted(removals, key=lambda r: (r['uid'], r['reason'])):
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        files['exclusions.jsonl'] = sha_file(temp / 'exclusions.jsonl')
        manifest = {'schema_version': 2, 'cleanup_metadata_version': 2, 'historical_freeze_replay': False, 'seed': SEED, 'files': files, 'row_counts': counts,
                    'native_base_path': str(Path(base).resolve()), 'native_base_sha256': base_hash,
                    'normalizer_sha256': sha_bytes(model.normalizer_spec.SerializeToString(deterministic=True)),
                    'normalizer_name': model.normalizer_spec.name, 'normalizer_changed': False,
                    'text_policy': 'Selected source text is preserved in raw_text. Only explicitly documented versioned source annotation markers may be replaced with one ordinary space in text. Native normalize() is used only internally for deduplication, audits and character weighting. Never feed native-normalized text back as raw training input.',
                    'source_annotation_policy': annotation_policy,
                    'annotation_cleanup_asr_limit': 'Records with tokenizer_fit_only_where_annotations_removed=true contain partial lexical text after source noise-tag removal. They must not be reused as complete paired audio training labels without a separate ASR label policy. Upstream IndicVoices standalone ASR filtering instead excludes entire sentences containing <unintelligible>.',
                    'normalized_character_definition': 'len(native SentencePiece normalize(text)): includes metaspace boundaries and synthetic dummy prefix, preserves native whitespace behavior',
                    'dedup_fingerprint_policy': 'Audit only: native-normalized lexical text, collapse consecutive U+2581 metaspace boundaries to one, then trim U+2581 at both edges. Original text, actual native-normalized output and character weighting remain unchanged. This catches noise-marker spacing variants without altering the encoder.',
                    'reserve_policy': 'Raw primary-intake reserve retains every original record. Frozen reserve contains unique representatives with equivalent record provenance for exact and discovered near duplicates. Script issues are flagged, not silently rewritten. Lower-priority train/dev records are removed on metadata, exact text or discovered near-text conflicts. Suppressed near variants remain indexed internally. No reserve text is printed.',
                    'legacy_policy': 'Old unsealed written train/dev are previously observed research data. Old dev remains dev by metadata group. No old seal is read. All identifiable FLORES rows excluded from every role.',
                    'permissions': {'SPRING_R1': 'Pinned primary author public-domain statement; no MIT-data assumption'},
                    'optional_sources': {'Vaani': 'Included under explicit source-specific policy' if any(r['source'] == VAANI for r in rows) else 'Excluded: source-specific gloss/noise annotation semantics are not yet verified; intake is quarantined, not training data'},
                    'source_limits': ['No claim of dialect-representative sampling or verified individual speaker identities.', 'Missing speaker/session/document metadata can limit separation guarantees.', 'Supplemental source counts are measured in the source matrix; source metadata does not establish speaker identity.', 'IN22-Conv is gated for the current account and not used.'],
                    'input_files': inputs, 'early_exclusions': exclusions, 'audit': audit,
                    'freeze_script_sha256': sha_file(__file__)}
        # Verify inputs again at the end so a moving intake cannot be frozen.
        for path, meta in inputs.items():
            if sha_file(path) != meta['sha256']: raise ValueError('Input changed before freeze completed: ' + path)
        if input_root:
            root = Path(input_root).resolve()
            manifest['input_files'] = {str(Path(path).relative_to(root)): dict(meta) for path, meta in inputs.items()}
            for meta in manifest['input_files'].values():
                if 'intake_manifest_path' in meta:
                    meta['intake_manifest_path'] = str(Path(meta['intake_manifest_path']).relative_to(root))
            manifest['native_base_path'] = Path(base).name
        json_write(temp / 'manifest.json', manifest)
        temp.rename(out)
    except BaseException:
        shutil.rmtree(temp); raise
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True, help='JSON manifest of relative input files, kinds, roles and SHA256')
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--annotation-policy', type=Path, required=True)
    p.add_argument('--max-near-candidates', type=int, default=5000)
    args = p.parse_args()
    contract = json.loads(args.inputs.read_text())
    specs = []
    for item in contract['inputs']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Input paths must be relative and contained')
        path = args.input_root.resolve() / rel
        if not path.resolve().is_relative_to(args.input_root.resolve()):
            raise ValueError('Input symlink escapes root')
        if sha_file(path) != item['sha256']:
            raise ValueError('Intake hash mismatch: ' + item['path'])
        specs.append((path, item['kind'], item['role']))
    policy = json.loads(args.annotation_policy.read_text())
    policy['policy_sha256'] = sha_file(args.annotation_policy)
    result = freeze(specs, args.output, base=args.base, max_candidates=args.max_near_candidates, annotation_policy=policy, input_root=args.input_root)
    print(json.dumps({'manifest_sha256': sha_file(args.output/'manifest.json'), 'row_counts': result['row_counts']}))


if __name__ == '__main__': main()
