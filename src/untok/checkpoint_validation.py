"""Read-only native checkpoint probe instrumentation."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from numbers import Integral
from typing import Any

from .checkpoint import _torch, compare_old_logits, mask_new_outputs_for_test
from .inference import _canonical_hash, _plain


def _equivalent_inference_config(original: Any, expanded: Any, decoder: str,
                                 *, allow_removed_prompts: bool = False) -> dict[str, Any]:
    old, new = _plain(original.cfg), _plain(expanded.cfg)
    for config in (old, new):
        if config.get("decoding", {}).get("strategy") != decoder:
            raise ValueError("Declared greedy strategy does not match the restored checkpoint")
    excluded = {"decoder": {"vocab_size"}, "joint": {"num_classes", "vocabulary"},
                "model_defaults": {"prompt_dictionary"}}
    hashes = {}
    for section in ("encoder", "preprocessor", "decoder", "joint", "model_defaults", "decoding"):
        old_value, new_value = old.get(section, {}), new.get(section, {})
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            old_value = {k: v for k, v in old_value.items() if k not in excluded.get(section, set())}
            new_value = {k: v for k, v in new_value.items() if k not in excluded.get(section, set())}
        if old_value != new_value:
            raise ValueError(f"Inference configuration changed outside permitted vocabulary fields: {section}")
        hashes[section] = _canonical_hash(old_value)
    old_prompts = old.get("model_defaults", {}).get("prompt_dictionary", {})
    new_prompts = new.get("model_defaults", {}).get("prompt_dictionary", {})
    retained_prompts = set(old_prompts) & set(new_prompts)
    if (not old_prompts or not retained_prompts
            or any(new_prompts[k] != old_prompts[k] for k in retained_prompts)
            or (not allow_removed_prompts and set(old_prompts) - set(new_prompts))):
        raise ValueError("Original prompt identities were not preserved")
    return {"config_section_sha256": hashes, "original_prompt_dictionary": old_prompts,
            "expanded_prompt_dictionary": new_prompts}


def _eager_decoder_state(model: Any) -> dict[str, Any]:
    """Inspect actual decoder objects; omitted config flags can default to graphs."""
    strategy = _plain(model.cfg).get("decoding", {}).get("strategy")
    decoder = getattr(getattr(model, "decoding", None), "decoding", None)
    if decoder is None or strategy not in {"greedy", "greedy_batch"}:
        raise ValueError("Cannot verify the native greedy decoder execution mode")
    if getattr(decoder, "_compiled_call_impl", None) is not None:
        raise ValueError("Compiled decoder execution cannot be instrumented")
    state = {"strategy": strategy, "decoder_class": type(decoder).__module__ + "." + type(decoder).__qualname__,
             "max_symbols": decoder.max_symbols, "loop_labels": getattr(decoder, "loop_labels", None),
             "use_cuda_graph_decoder": getattr(decoder, "use_cuda_graph_decoder", False),
             "allow_cuda_graphs": None, "cuda_graphs_mode": None}
    if strategy == "greedy_batch" and getattr(decoder, "use_cuda_graph_decoder", None) is not False:
        raise ValueError("Actual greedy decoder still permits CUDA graphs")
    computer = getattr(decoder, "decoding_computer", None)
    if strategy == "greedy_batch" and state["loop_labels"]:
        if computer is None or getattr(computer, "allow_cuda_graphs", None) is not False:
            raise ValueError("Actual label-looping decoder still permits CUDA graphs")
        if getattr(computer, "cuda_graphs_mode", "unverified") is not None:
            raise ValueError("Actual label-looping decoder has an active CUDA graph mode")
        state["allow_cuda_graphs"] = False
    return state


def _configure_eager_decoding(model: Any) -> dict[str, Any]:
    """Disable graph execution explicitly while preserving decoding and weights."""
    serialized = copy.deepcopy(_plain(model.cfg).get("decoding", {}))
    if serialized.get("strategy") not in {"greedy", "greedy_batch"}:
        raise ValueError("Eager instrumentation requires a native greedy strategy")
    for group in (serialized, serialized.get("greedy", {})):
        for key, value in group.items():
            if key != "use_cuda_graph_decoder" and ("cuda_graph" in key or "compile" in key) and value not in (False, None):
                raise ValueError("Unsupported graph/compiled decoding option for instrumentation")
    before_decoder = getattr(getattr(model, "decoding", None), "decoding", None)
    if before_decoder is None or not hasattr(before_decoder, "max_symbols"):
        raise ValueError("Cannot inspect original greedy decoder settings")
    behavior = (type(before_decoder), before_decoder.max_symbols, getattr(before_decoder, "loop_labels", None))

    def state_identity():
        return {name: (value.data_ptr(), value._version, tuple(value.shape), value.dtype)
                for name, value in model.state_dict().items()}

    original_state = state_identity()
    requested = copy.deepcopy(serialized)
    requested.setdefault("greedy", {})["use_cuda_graph_decoder"] = False
    model.change_decoding_strategy(requested, verbose=False)
    effective = _plain(model.cfg)["decoding"]

    def check_preserved(before, after, path=()):
        for key, value in before.items():
            location = path + (key,)
            if location == ("greedy", "use_cuda_graph_decoder"):
                continue
            if key not in after:
                raise ValueError(f"Decoding option disappeared during eager setup: {location}")
            if isinstance(value, dict) and isinstance(after[key], dict):
                check_preserved(value, after[key], location)
            elif after[key] != value:
                raise ValueError(f"Decoding behavior changed during eager setup: {location}")

    check_preserved(serialized, effective)
    runtime = _eager_decoder_state(model)
    after_decoder = model.decoding.decoding
    if behavior != (type(after_decoder), after_decoder.max_symbols, getattr(after_decoder, "loop_labels", None)):
        raise ValueError("Greedy decoder algorithm or max_symbols changed during eager setup")
    if original_state != state_identity():
        raise ValueError("Model tensors changed during eager decoder setup")
    return {"serialized_config": serialized, "effective_config": effective, "runtime": runtime,
            "model_tensors_unchanged": True, "changed_option": "greedy.use_cuda_graph_decoder=false",
            "purpose": "observable joint-head hooks; no checkpoint or model weights are rewritten"}


def _hypothesis(result: Any, mapping) -> dict[str, Any]:
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("Expected one RNNT Hypothesis for one audio input")
    item = result[0]
    text, ids = getattr(item, "text", None), getattr(item, "y_sequence", None)
    if not isinstance(text, str) or ids is None:
        raise ValueError("RNNT Hypothesis must expose text and y_sequence; no text-only fallback")
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    elif hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, (tuple, list)) or any(not isinstance(i, Integral) for i in ids):
        raise ValueError("RNNT y_sequence must be a one-dimensional integer sequence")
    ids = [int(i) for i in ids]
    return {"text": text, "model_ids": ids,
            "canonical_hf_ids": mapping.to_canonical(ids, drop_blank=False)}


@contextmanager
def _head_trace(head, *, old_to_new=None, max_probes: int = 8):
    """Capture actual pre-softmax head inputs; optionally mask before softmax."""
    torch = _torch()
    trace: dict[str, Any] = {"calls": 0, "probes": []}

    def hook(module, args, output):
        if len(args) != 1 or not isinstance(output, torch.Tensor) or not isinstance(args[0], torch.Tensor):
            raise ValueError("Unrecognized joint-head forward signature")
        trace["calls"] += 1
        if len(trace["probes"]) < max_probes:
            # Bound trace memory even if the runtime batches many encoder steps.
            inputs = args[0].detach().reshape(-1, args[0].shape[-1])[:8].cpu().clone()
            logits = output.detach().reshape(-1, output.shape[-1])[:8].cpu().clone()
            if inputs.numel():
                trace["probes"].append((inputs, logits))
        return mask_new_outputs_for_test(output, old_to_new) if old_to_new is not None else None

    handle = head.register_forward_hook(hook)
    try:
        yield trace
    finally:
        handle.remove()


def _probe_checks(source_trace, expanded_trace, expanded_head, row_map, *, atol, rtol):
    torch = _torch()
    if not source_trace["calls"] or not expanded_trace["calls"] or not source_trace["probes"] or not expanded_trace["probes"]:
        raise ValueError("Joint-head hooks did not execute; no migration-logit check was performed")
    errors, input_errors, replay_errors = [], [], []
    shape_match = source_trace["calls"] == expanded_trace["calls"]
    shape_match = shape_match and len(source_trace["probes"]) == len(expanded_trace["probes"])
    if shape_match:
        for (old_inputs, old_logits), (new_inputs, new_logits) in zip(source_trace["probes"], expanded_trace["probes"]):
            if old_inputs.shape != new_inputs.shape:
                shape_match = False
                break
            input_errors.append((old_inputs - new_inputs).abs().max().item())
            if not torch.allclose(old_inputs, new_inputs, atol=atol, rtol=rtol):
                shape_match = False
            try:
                result = compare_old_logits(old_logits, new_logits, row_map, atol=atol, rtol=rtol)
                errors.append(result["max_absolute_error"])
            except ValueError:
                shape_match = False
    # Independently replay captured BASELINE joint inputs into the new final
    # head. This uses real audio-derived states and does not invent decoder
    # prefixes or assume undocumented NeMo encoder.forward interfaces.
    with torch.inference_mode():
        device = expanded_head.weight.device
        for inputs, old_logits in source_trace["probes"]:
            replay = expanded_head(inputs.to(device))
            result = compare_old_logits(old_logits, replay, row_map, atol=atol, rtol=rtol)
            replay_errors.append(result["max_absolute_error"])
    return {"passed": shape_match, "source_joint_calls": source_trace["calls"],
            "masked_joint_calls": expanded_trace["calls"], "captured_probes": len(source_trace["probes"]),
            "max_joint_input_absolute_error": max(input_errors, default=None),
            "max_old_logit_absolute_error": max(errors, default=None),
            "max_fixed_input_replay_absolute_error": max(replay_errors, default=None),
            "atol": atol, "rtol": rtol,
            "scope": "bounded_actual_greedy_joint_states_and_fixed_input_head_replay"}
