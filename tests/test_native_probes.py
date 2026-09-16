"""CPU controls for the read-only native audio-probe instrumentation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def probes(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    loaded = []
    for name in ("native_checkpoint_probe", "native_subset_probe"):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def test_evaluation_reads_test_and_valid_audio_without_training(probes, tmp_path):
    full, _ = probes
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"hashed fixture; audio decode is outside this test")
    manifest = tmp_path / "audio.jsonl"
    entries = [{"id": split, "source_split": split, "audio": audio.name,
                "audio_sha256": full.sha(audio)} for split in ("test", "valid", "train")]
    manifest.write_text("\n".join(map(json.dumps, entries)))
    assert [row["id"] for row in full.rows(manifest, ("test", "valid"))] == ["test", "valid"]
    assert [row["id"] for row in full.rows(manifest, "train")] == ["train"]
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Audio content"):
        full.rows(manifest, ("test", "valid"))


def test_source_and_locale_proxy_evidence_is_preserved(probes):
    full, _ = probes
    evidence = {"source_dataset": "fixture", "source_revision": "pinned",
                "source_split": "valid", "locale_audio_match": False,
                "locale_coverage_limitation": "regional locale uses another region's audio"}
    assert full.row_provenance({**evidence, "text": "excluded"}) == evidence


def stream_pair():
    source = {"chunk_count": 2, "initial_cache": [0], "streaming_cfg": {"chunk": 56},
              "effective_decoding_config": {"strategy": "greedy_batch"},
              "runtime_decoder": {"use_cuda_graph_decoder": False}, "buffer_exhausted": True,
              "steps": []}
    for index in range(2):
        source["steps"].append({"step": index, "chunk": {"sha256": str(index)}, "chunk_lengths": [12],
                                "drop_extra_pre_encoded": index, "last_chunk": index == 1,
                                "partial_hypotheses_supplied": index > 0, "cache_input": [index],
                                "cache_output": [index + 1], "cache_lengths": [index + 1],
                                "hypothesis": {"text": "ab", "model_ids": [0, 2]}})
    target = copy.deepcopy(source)
    for step in target["steps"]:
        step["hypothesis"]["model_ids"] = [0, 1]
    return source, target


@pytest.mark.parametrize("field", ["chunk_count", "initial_cache", "streaming_cfg",
                                    "effective_decoding_config", "runtime_decoder", "buffer_exhausted"])
def test_streaming_header_mismatch_fails(probes, field):
    full, _ = probes
    source, target = stream_pair()
    assert full.streaming_parity(source, target, (0, None, 1, 3))
    target[field] = "different"
    assert not full.streaming_parity(source, target, (0, None, 1, 3))


@pytest.mark.parametrize("field", ["step", "chunk", "chunk_lengths", "drop_extra_pre_encoded", "last_chunk",
                                    "partial_hypotheses_supplied", "cache_input", "cache_output", "cache_lengths"])
def test_streaming_cache_and_chunk_mismatch_fails(probes, field):
    full, _ = probes
    source, target = stream_pair()
    target["steps"][1][field] = "different"
    assert not full.streaming_parity(source, target, (0, None, 1, 3))


def test_streaming_rejects_removed_ids_and_empty_execution(probes):
    full, _ = probes
    source, target = stream_pair()
    source["steps"][0]["hypothesis"]["model_ids"] = [1]
    assert not full.streaming_parity(source, target, (0, None, 1, 3))
    source["steps"] = target["steps"] = []
    assert not full.streaming_parity(source, target, (0, None, 1, 3))


def test_subset_scope_selection_and_missing_audio_failures(probes):
    _, subset = probes
    records = [{"id": f"{lang}-{seconds}", "language": lang, "duration": seconds, "target_lang": lang + "-XX"}
               for lang in ("en", "fr", "hi", "ta", "ar", "ru") for seconds in (12, 8)]
    assert [row["id"] for row in subset.choose_rows(records, "latin")] == ["en-8", "fr-8"]
    assert [row["id"] for row in subset.choose_rows(records, "latin-indic")] == ["en-8", "fr-8", "hi-8", "ta-8"]
    assert [row["id"] for row in subset.choose_rows(records, "original")] == ["ar-8", "en-8", "fr-8", "hi-8", "ru-8"]
    with pytest.raises(ValueError, match="outside"):
        subset.choose_rows(records, "latin", ["hi"])
    with pytest.raises(ValueError, match="Missing"):
        subset.choose_rows(records, "latin-indic", ["kn"])
    with pytest.raises(ValueError, match="positive"):
        subset.choose_rows(records, "latin", max_per_language=0)


def test_subset_preserves_all_retained_source_regional_prompt_paths(probes):
    from untok.speech_metrics import ADAPTATION_LOCALES, BASE_ASR_LOCALES
    from untok.prompts import TARGET_LOCALES

    _, subset = probes
    assert len(subset.LATIN_SOURCE_LANGUAGES) == 25
    locales = sorted(set(BASE_ASR_LOCALES) | set(ADAPTATION_LOCALES) | set(TARGET_LOCALES.values()))
    records = [{"id": locale + "-" + str(seconds), "target_lang": locale,
                # Nynorsk is a Bokmal-audio proxy, with a separate prompt.
                "language": "nb" if locale == "nn-NO" else locale.split("-")[0], "duration": seconds}
               for locale in locales for seconds in (9, 6)]
    latin = subset.choose_rows(records, "latin")
    indic = subset.choose_rows(records, "latin-indic")
    assert len(latin) == 29 and len(indic) == 51
    assert {row["target_lang"] for row in latin if row["language"] == "en"} == {"en-US", "en-GB"}
    assert {row["target_lang"] for row in latin if row["language"] == "nb"} == {"nb-NO", "nn-NO"}
    assert all(row["duration"] == 6 for row in latin + indic)
    assert "ar-AR" not in {row["target_lang"] for row in indic}
    assert {row["target_lang"] for row in subset.choose_rows(records, "latin", ["en"])} == {"en-US", "en-GB"}


def test_subset_control_uses_same_original_prompt_or_explicit_auto(probes):
    _, subset = probes
    old = {"auto": 0, "en": 1, "hi": 2}
    new = {**old, "ta": 40}
    known = subset.control_prompt({"target_lang": "hi"}, old, new)
    assert known["control_target_lang"] == "hi" and known["source_has_requested_prompt"]
    added = subset.control_prompt({"target_lang": "ta"}, old, new)
    assert added["control_target_lang"] == "auto" and not added["source_has_requested_prompt"]
    assert added["requested_prompt_id"] == 40
    with pytest.raises(ValueError, match="identical"):
        subset.control_prompt({"target_lang": "ta"}, old, {**new, "auto": 9})
    with pytest.raises(ValueError, match="missing"):
        subset.control_prompt({"target_lang": "kn"}, old, new)


def test_subset_configuration_can_remove_prompt_names_but_cannot_change_slots():
    from untok.checkpoint_validation import _equivalent_inference_config

    original = SimpleNamespace(cfg={"decoding": {"strategy": "greedy_batch"},
        "model_defaults": {"num_prompts": 4, "prompt_dictionary": {"auto": 0, "en-US": 1, "ar-AR": 2}}})
    target = copy.deepcopy(original)
    del target.cfg["model_defaults"]["prompt_dictionary"]["ar-AR"]
    with pytest.raises(ValueError, match="prompt identities"):
        _equivalent_inference_config(original, target, "greedy_batch")
    assert _equivalent_inference_config(original, target, "greedy_batch", allow_removed_prompts=True)
    target.cfg["model_defaults"]["prompt_dictionary"]["en-US"] = 2
    with pytest.raises(ValueError, match="prompt identities"):
        _equivalent_inference_config(original, target, "greedy_batch", allow_removed_prompts=True)


def test_subset_hypothesis_mapping_rejects_pruned_source_rows(probes):
    _, subset = probes
    old, new = {"text": "ab", "native_ids": [0, 2]}, {"text": "ab", "native_ids": [0, 1]}
    assert subset.mapped_hypothesis_equal(old, new, (0, None, 1, 3))
    old["native_ids"] = [1]
    assert not subset.mapped_hypothesis_equal(old, new, (0, None, 1, 3))


def test_streaming_eval_boundary_restores_nested_modes_without_changing_tensors(probes):
    torch = pytest.importorskip("torch")
    _, subset = probes
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.8), torch.nn.BatchNorm1d(3)).eval()
    # Match NeMo teardown: parent mode is false, child unfreeze sets train true.
    model[1].train()
    model[2].train()
    assert not model.training and model[1].training
    tensors = {name: value.clone() for name, value in model.state_dict().items()}
    evidence = subset.prepare_streaming_evaluation(model)
    assert evidence["training_module_count_before"] == 2
    assert evidence["training_module_count_after"] == 0
    assert not any(module.training for module in model.modules())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in tensors.items())
    with torch.inference_mode():
        values = torch.ones(4, 3)
        assert torch.equal(model(values), model(values))


def test_subset_actual_head_capture_and_retained_state_replay(probes):
    torch = pytest.importorskip("torch")
    from untok.checkpoint import RNNTLayout
    from untok.checkpoint_validation import _head_trace

    _, subset = probes
    source, target = torch.nn.Linear(3, 5), torch.nn.Linear(3, 4)
    old = RNNTLayout("embed", "weight", "bias", 4, 5)
    new = RNNTLayout("embed", "weight", "bias", 3, 4)
    mapping = (0, None, 1, None, 3)
    with torch.inference_mode():
        target.weight[[0, 1, 3]] = source.weight[[0, 2, 4]]
        target.bias[[0, 1, 3]] = source.bias[[0, 2, 4]]
        states = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
        with _head_trace(source, old_to_new=[0, 2, 4]) as left:
            masked_left = source(states)
        with _head_trace(target, old_to_new=[0, 1, 3]) as right:
            masked_right = target(states)
        assert torch.isneginf(masked_left[:, [1, 3]]).all()
        assert torch.isneginf(masked_right[:, 2]).all()
        assert subset.retained_trace_checks(left, right, target, old, new, mapping)["passed"]
        changed = copy.deepcopy(right)
        changed["calls"] += 1
        assert not subset.retained_trace_checks(left, changed, target, old, new, mapping)["passed"]
        changed = copy.deepcopy(right)
        changed["probes"][0][0][0, 0] += 1
        assert not subset.retained_trace_checks(left, changed, target, old, new, mapping)["passed"]
        with pytest.raises(ValueError, match="must both execute"):
            subset.retained_trace_checks({"calls": 0, "probes": []}, right, target, old, new, mapping)
        target.bias[0] += 1
        with pytest.raises(ValueError, match="logits changed"):
            subset.retained_trace_checks(left, right, target, old, new, mapping)


def test_preserved_dormant_rows_are_excluded_from_paired_controls_and_logit_replay(probes):
    torch = pytest.importorskip("torch")
    from untok.checkpoint import RNNTLayout
    from untok.checkpoint_validation import _head_trace

    _, subset = probes
    source, target = torch.nn.Linear(3, 5), torch.nn.Linear(3, 6)
    old = RNNTLayout("embed", "weight", "bias", 4, 5)
    new = RNNTLayout("embed", "weight", "bias", 5, 6)
    mapping = (0, 1, 2, 3, 5)
    inactive = (1, 3)
    old_ids, new_ids = subset.active_retained_row_pairs(old, new, mapping, inactive)
    assert old_ids == (0, 2, 4)
    assert new_ids == (0, 2, 5)
    assert mapping == (0, 1, 2, 3, 5)  # Dormant checkpoint rows remain mapped.
    assert subset.active_retained_row_pairs(old, new, mapping) == ((0, 1, 2, 3, 4), mapping)
    for invalid in ((5,), (1, 1), (-1,), (True,)):
        with pytest.raises(ValueError, match="Inactive probe IDs"):
            subset.active_retained_row_pairs(old, new, mapping, invalid)

    def mask_dormant_outputs(module, arguments, output):
        output = output.clone()
        output[..., list(inactive)] = -torch.inf
        return output

    with torch.inference_mode():
        target.weight[list(mapping)] = source.weight
        target.bias[list(mapping)] = source.bias
        states = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
        handle = target.register_forward_hook(mask_dormant_outputs)
        try:
            with _head_trace(source, old_to_new=old_ids) as left:
                masked_left = source(states)
            with _head_trace(target, old_to_new=new_ids) as right:
                masked_right = target(states)
            assert torch.isneginf(masked_left[:, list(inactive)]).all()
            assert torch.isneginf(masked_right[:, [1, 3, 4]]).all()
            result = subset.retained_trace_checks(left, right, target, old, new, mapping,
                                                  inactive_target_ids=inactive)
            assert result["passed"]
            assert result["inactive_target_outputs_excluded"] == [1, 3]
            assert result["max_fixed_input_replay_absolute_error"] == 0
            # Forgetting inactive IDs compares masked -inf to finite source
            # logits and must fail, rather than passing a misleading control.
            with pytest.raises(ValueError, match="non-finite"):
                subset.retained_trace_checks(left, right, target, old, new, mapping)
            target.bias[0] += 1
            with pytest.raises(ValueError, match="logits changed"):
                subset.retained_trace_checks(left, right, target, old, new, mapping,
                                             inactive_target_ids=inactive)
        finally:
            handle.remove()


@pytest.fixture(scope="module")
def compact_probe_bundles(tmp_path_factory):
    from untok.bundles import PROFILES, _subset_artifacts, load_tokenizer_bundle
    from untok.clean import build_clean_bundles

    data = Path(__file__).resolve().parents[1] / "src" / "untok" / "data"
    output = tmp_path_factory.mktemp("compact-probe-bundles")
    build_clean_bundles(data / "source", output / "clean")
    # Public packaging now creates v5 profiles. Build the frozen v1 subset
    # explicitly to continue checking historical receipt compatibility.
    source = load_tokenizer_bundle(data / "source")
    legacy = output / "legacy" / "latin"
    legacy.mkdir(parents=True)
    files, manifest, _, _ = _subset_artifacts(source.base_model_bytes, source.model_bytes, "latin")
    for name, raw in files.items():
        (legacy / name).write_bytes(raw)
    (legacy / "manifest.json").write_text(json.dumps(manifest))
    return ({profile: load_tokenizer_bundle(output / "clean" / profile) for profile in PROFILES},
            load_tokenizer_bundle(output / "legacy" / "latin"))


def _native_source(raw):
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_proto=raw)
    return SimpleNamespace(tokenizer=SimpleNamespace(
        backend=processor, vocab_size=processor.get_piece_size(),
        ids_to_tokens=lambda ids: [processor.id_to_piece(index) for index in ids],
        text_to_ids=lambda text: processor.encode(text, out_type=int),
    ))


@pytest.mark.parametrize("profile", ["original", "full", "latin-indic", "latin"])
@pytest.mark.parametrize("inventory", ["original_native_base", "original_untok_full_v1"])
def test_compact_probe_selects_the_exact_migrated_source_map(probes, compact_probe_bundles, profile, inventory):
    _, subset = probes
    clean, _ = compact_probe_bundles
    target = clean[profile]
    is_base = inventory == "original_native_base"
    raw = target.base_model_bytes if is_base else target.full_model_bytes
    expected = target.source_native_to_target_native if is_base else target.full_native_to_subset_native
    report = {
        "source_inventory": inventory, "source_tokenizer_sha256": subset.sha_bytes(raw),
        "source_remapping": "source_native_to_target_native" if is_base else "full_native_to_subset_native",
        "old_model_to_new_model": list(expected), "requires_retokenized_training_labels": profile != "original" or not is_base,
    }
    actual, evidence = subset.select_probe_inventory(_native_source(raw), target, report)
    assert actual == expected and actual[-1] == target.blank_id
    if is_base:
        assert actual[:-1] == tuple(range(len(actual) - 1))
    else:
        assert any(index is None for index in actual)
    assert evidence["source_inventory"] == inventory
    assert evidence["source_tokenizer_sha256"] == subset.sha_bytes(raw)
    assert evidence["source_native_check"]["piece_ids_checked"] == len(actual) - 1
    assert evidence["requires_retokenized_training_labels"] == (profile != "original" or not is_base)
    wrong = dict(report, old_model_to_new_model=list(
        target.full_native_to_subset_native if is_base else target.source_native_to_target_native))
    with pytest.raises(ValueError, match="source mapping"):
        subset.select_probe_inventory(_native_source(raw), target, wrong)


@pytest.mark.parametrize("field", ["source_inventory", "source_tokenizer_sha256", "source_remapping",
                                   "requires_retokenized_training_labels"])
def test_compact_probe_rejects_missing_or_forged_inventory_metadata(probes, compact_probe_bundles, field):
    _, subset = probes
    clean, _ = compact_probe_bundles
    target = clean["latin-indic"]
    source = _native_source(target.full_model_bytes)
    report = {
        "source_inventory": "original_untok_full_v1",
        "source_tokenizer_sha256": subset.sha_bytes(target.full_model_bytes),
        "source_remapping": "full_native_to_subset_native",
        "old_model_to_new_model": list(target.full_native_to_subset_native),
        "requires_retokenized_training_labels": True,
    }
    for change in ("missing", "forged"):
        altered = dict(report)
        if change == "missing":
            altered.pop(field)
        else:
            altered[field] = False if field == "requires_retokenized_training_labels" else "wrong"
        with pytest.raises(ValueError, match="inventory|retokenized"):
            subset.select_probe_inventory(source, target, altered)


def test_legacy_subset_probe_accepts_its_original_report_without_new_fields(probes, compact_probe_bundles):
    _, subset = probes
    _, legacy = compact_probe_bundles
    original = _native_source(legacy.base_model_bytes)
    mapping, evidence = subset.select_probe_inventory(
        original, legacy, {"old_model_to_new_model": list(legacy.source_native_to_target_native)})
    assert mapping == legacy.source_native_to_target_native
    assert evidence["source_inventory"] == "original_native_base"
    assert not evidence["requires_retokenized_training_labels"]
    with pytest.raises(ValueError, match="not a pinned"):
        subset.select_probe_inventory(_native_source(legacy.full_model_bytes), legacy,
                                      {"old_model_to_new_model": list(legacy.full_native_to_subset_native)})


def test_clean_full_probe_includes_original_non_latin_and_indic_locales(probes):
    _, subset = probes
    languages = ["en", "ru", "ja", "ko", "zh", "el", "he", "th", "ml", "hi", "ar"]
    records = [{"id": f"{language}-{seconds}", "language": language,
                "target_lang": f"{language}-XX", "duration": seconds}
               for language in languages for seconds in (12, 5)]
    selected = subset.choose_rows(records, "full")
    assert {row["language"] for row in selected} == set(languages)
    assert all(row["duration"] == 5 for row in selected)
    assert [row["language"] for row in subset.choose_rows(records, "full", ["ru", "ml"])] == ["ml", "ru"]
    with pytest.raises(ValueError, match="outside"):
        subset.choose_rows(records, "full", ["xx"])
    with pytest.raises(ValueError, match="checkpoint"):
        subset.choose_rows(records, "unknown")
