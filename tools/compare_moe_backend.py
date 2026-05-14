#!/usr/bin/env python3
"""
Compare default vs trtllm_fused_moe backend on a single prompt.

Usage on B200 (inside docker-env-lmdeploy-min):

    python tools/compare_moe_backend.py \
        --model /home/workcode/models/Qwen3-30B-A3B \
        --gpu 1

Run sequentially (NOT in parallel) — both backends share the same GPU.
"""

import argparse
import os
import sys


PROMPT = (
    "Solve this step by step:\n"
    "Alice has 3 boxes.  Each box contains 4 red balls and 7 blue balls.\n"
    "How many balls does Alice have in total?  Show the arithmetic.\n"
)
SAMPLING = dict(top_k=1, temperature=1.0, max_new_tokens=128)  # greedy


def run_one(model: str, backend: str, dtype: str = 'auto') -> str:
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

    cfg = TurbomindEngineConfig(
        tp=1,
        session_len=4096,
        cache_max_entry_count=0.30,
        moe_backend=backend,
        dtype=dtype,
        # leave attention_backend at 'default' so the only varying axis is moe
    )
    pipe = pipeline(model, backend_config=cfg)
    out = pipe([PROMPT], gen_config=GenerationConfig(**SAMPLING))
    text = out[0].text if hasattr(out[0], 'text') else str(out[0])
    pipe.close()
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--dtype', default='auto', choices=['auto', 'float16', 'bfloat16'])
    # `--only` lets you run a single backend (useful when one side is
    # known-broken on the model — e.g. `default` cutlass MoE has no FP8
    # kernel for some hidden/intermediate sizes on SM100 today, but the
    # trtllm path may still work).
    ap.add_argument('--only', choices=['default', 'trtllm_fused_moe'], default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f'CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}', flush=True)
    print(f'dtype = {args.dtype}', flush=True)

    print(f'\n=== prompt ===\n{PROMPT}', flush=True)

    out_default = out_trtllm = None
    if args.only != 'trtllm_fused_moe':
        print('\n=== running with moe_backend=default ===', flush=True)
        try:
            out_default = run_one(args.model, 'default', dtype=args.dtype)
            print(f'--- output (default) ---\n{out_default}', flush=True)
        except Exception as e:
            print(f'--- default backend FAILED: {type(e).__name__}: {e} ---', flush=True)
            if args.only == 'default':
                return 2

    if args.only != 'default':
        print('\n=== running with moe_backend=trtllm_fused_moe ===', flush=True)
        try:
            out_trtllm = run_one(args.model, 'trtllm_fused_moe', dtype=args.dtype)
            print(f'--- output (trtllm_fused_moe) ---\n{out_trtllm}', flush=True)
        except Exception as e:
            print(f'--- trtllm_fused_moe backend FAILED: {type(e).__name__}: {e} ---', flush=True)
            if args.only == 'trtllm_fused_moe':
                return 3

    if out_default is None or out_trtllm is None:
        print('\n=== diff ===\nSKIPPED (only one backend ran)', flush=True)
        return 0

    print('\n=== diff ===', flush=True)
    if out_default == out_trtllm:
        print('IDENTICAL ✓', flush=True)
        return 0
    print('DIFFER ✗', flush=True)
    # Quick char-level overlap report
    common = 0
    for a, b in zip(out_default, out_trtllm):
        if a != b:
            break
        common += 1
    print(f'identical prefix: {common} chars', flush=True)
    print(f'len(default)={len(out_default)}  len(trtllm)={len(out_trtllm)}', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
