"""Pinned native streaming primitives shared by checkpoint probes.

The official buffer precomputes features before executing cached encoder chunks.
This simulation does not measure a live audio frontend or end-to-end latency.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib

NEMO_REVISION = "ca4daa1470f6c01068c4e6a9a73b19b9a91dc366"


SOURCE_HASHES = {
    "nemo.collections.asr.parts.mixins.mixins": "d4531808440c9f1649064044a21419975c0f2424e2442d4f2ef68a301d576c96",
    "nemo.collections.asr.parts.utils.streaming_utils": "e90bf75cb44106906c001bb52b039d3cbf370e8c33bb5c77ef800bf415864f15",
    "nemo.collections.asr.modules.conformer_encoder": "b0b9c7997d66a654b9d54d976ef154520ce76c65f26395c0a41cdd52432fb26d",
    "nemo.collections.asr.parts.submodules.rnnt_decoding": "7d10acca5d721d2a37a303b70c3c1b70fd5aed5cde9fbe68385e67541242cd5c",
}


def tensor_evidence(tensor):
    import torch
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise ValueError("Native streaming returned an invalid cache or chunk tensor")
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "nonzero_values": int(torch.count_nonzero(value))}


@contextmanager
def observe_prompt(model, target_lang):
    import torch
    from untok.inference import _plain
    defaults = _plain(model.cfg)["model_defaults"]
    width, count = defaults["enc_hidden"], defaults["num_prompts"]
    prompt_id = defaults["prompt_dictionary"][target_lang]
    evidence = {"target_lang": target_lang, "prompt_id": prompt_id, "calls": 0, "frames": 0}
    if not model.concat or model.num_prompts != count:
        raise ValueError("Native streaming prompt conditioning is not active")

    def inspect_input(module, arguments):
        value = arguments[0]
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != width + count:
            raise ValueError("Unexpected actual prompt projection input")
        expected = torch.zeros_like(value[..., width:])
        expected[..., prompt_id] = 1
        if not torch.equal(value[..., width:], expected):
            raise ValueError("Actual streaming prompt differs from the requested language")
        evidence["calls"] += 1
        evidence["frames"] += value.shape[1]

    handle = model.prompt_kernel.register_forward_pre_hook(inspect_input)
    try:
        yield evidence
    finally:
        handle.remove()


def run_stream(model, waveform, mapping, target_lang, row_map=None):
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from untok.checkpoint_validation import _head_trace, _hypothesis, _eager_decoder_state
    from untok.inference import _plain
    buffer = CacheAwareStreamingAudioBuffer(model, online_normalization=False, pad_and_drop_preencoded=False)
    with torch.inference_mode():
        buffer.append_audio(waveform.numpy(), stream_id=-1)
    if len(buffer.streams_length) != 1:
        raise ValueError("Expected exactly one real audio stream")
    caches = model.encoder.get_initial_cache_state(batch_size=1)
    initial_cache = [tensor_evidence(value) for value in caches]
    if any(item["nonzero_values"] for item in initial_cache):
        raise ValueError("A new stream did not begin with empty caches")
    previous_hypotheses, previous_predictions = None, None
    steps = []
    streaming_cfg = _plain(vars(model.encoder.streaming_cfg))
    with observe_prompt(model, target_lang) as prompt, _head_trace(model.joint.joint_net[-1], old_to_new=row_map) as trace:
        for index, (chunk, lengths) in enumerate(buffer):
            _eager_decoder_state(model)
            if index >= 32:
                raise ValueError("The single-clip streaming probe exceeded its 32-chunk bound")
            before = [tensor_evidence(value) for value in caches]
            if steps and before != steps[-1]["cache_output"]:
                raise ValueError("A returned encoder cache was not carried into the next chunk")
            prompt_before, head_before = prompt["calls"], trace["calls"]
            drop = 0 if index == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded
            with torch.inference_mode():
                result = model.conformer_stream_step(
                    processed_signal=chunk.to(dtype=torch.float32), processed_signal_length=lengths,
                    cache_last_channel=caches[0], cache_last_time=caches[1], cache_last_channel_len=caches[2],
                    keep_all_outputs=buffer.is_buffer_empty(), previous_hypotheses=previous_hypotheses,
                    previous_pred_out=previous_predictions, drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            if not isinstance(result, tuple) or len(result) != 6:
                raise ValueError("Native conformer_stream_step did not return its expected six outputs")
            previous_predictions, hypotheses, *rest = result
            caches = tuple(rest[:3])
            previous_hypotheses = rest[3]
            if prompt["calls"] <= prompt_before or trace["calls"] <= head_before:
                raise ValueError("A streaming chunk skipped the observable prompt or joint-head execution")
            steps.append({"step": index, "chunk": tensor_evidence(chunk), "chunk_lengths": lengths.cpu().tolist(),
                          "drop_extra_pre_encoded": drop, "last_chunk": buffer.is_buffer_empty(),
                          "partial_hypotheses_supplied": index > 0,
                          "cache_input": before, "cache_output": [tensor_evidence(value) for value in caches],
                          "cache_lengths": caches[2].cpu().tolist(),
                          "hypothesis": _hypothesis(hypotheses, mapping),
                          "prompt_calls": prompt["calls"] - prompt_before,
                          "joint_head_calls": trace["calls"] - head_before})
    if len(steps) < 2 or not any(steps[-1]["cache_lengths"]):
        raise ValueError("The recording did not exercise multiple cached streaming chunks")
    result = {"chunk_count": len(steps), "initial_cache": initial_cache, "streaming_cfg": streaming_cfg,
              "effective_decoding_config": _plain(model.cfg)["decoding"], "runtime_decoder": _eager_decoder_state(model),
              "prompt_evidence": prompt, "joint_head_calls": trace["calls"], "steps": steps,
              "final_hypothesis": steps[-1]["hypothesis"], "buffer_exhausted": buffer.is_buffer_empty()}
    return result, trace
