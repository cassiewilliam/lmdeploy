#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Fetch FlashInfer cubin artifacts for the FlashInfer integration.

Driven by CMake at build time (see flashinfer/attention/CMakeLists.txt).
Resolves the artifactory base URL + per-version cubin hash from CLI args,
walks `checksums.txt`, and downloads any `.cubin` whose local copy is
missing or fails SHA-256 verification.

Stdlib-only on purpose — we don't want to add `requests` / `flashinfer`
to the build-host pip set just to mirror public cubin artifacts.

Examples:
    # Download every FMHA cubin listed in checksums.txt (a few hundred MB):
    python download_cubins.py \
        --hash 1d876ee612888821b168c25ffa75a9dcbb963aaa \
        --dest ~/.cache/flashinfer/cubins

    # Download a non-FMHA cubin artifact, preserving the same cache layout
    # used by FlashInfer's cubin callback:
    python download_cubins.py \
        --hash 39a9d28268f43475a757d5700af135e1e58c9849 \
        --artifact-path '{hash}/batched_gemm-5ee61af-2b9855b' \
        --dest ~/.cache/flashinfer/cubins

    # Download only cubins whose name matches one of the regex patterns,
    # e.g. for a single-dtype/single-headdim build:
    python download_cubins.py \
        --hash 1d876ee612888821b168c25ffa75a9dcbb963aaa \
        --dest ~/.cache/flashinfer/cubins \
        --filter 'QkvFp16OFp16H128PagedKv|QkvBfloat16OBfloat16H128PagedKv' \
        --filter '_Q128Kv128'
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = (
    'https://edge.urm.nvidia.com/artifactory/'
    'sw-kernelinferencelibrary-public-generic-local'
)


def fetch_text(url: str, retries: int = 3, timeout: int = 30) -> str:
    """GET text resource with simple retry/backoff."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode('utf-8')
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries:
                time.sleep(2 * attempt)
    raise SystemExit(f'Failed to fetch {url}: {last}')


def fetch_binary(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> bytes:
    """GET bytes, write to dest atomically (.tmp → rename)."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + '.tmp')
            tmp.write_bytes(data)
            tmp.replace(dest)
            return data
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries:
                time.sleep(2 * attempt)
    raise SystemExit(f'Failed to fetch {url}: {last}')


def parse_checksums(text: str) -> list[tuple[str, str]]:
    """Yield (sha256_hex, filename) tuples from a `sha256  name\n` listing."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        out.append((parts[0], parts[1]))
    return out


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hash', required=True,
                    help='FlashInfer cubin artifact hash (e.g. e7afc4134…)')
    ap.add_argument('--dest', required=True, type=Path,
                    help='Local cubin cache dir; files go in <dest>/<artifact-path>/')
    ap.add_argument('--artifact-path', default='{hash}/fmha/trtllm-gen',
                    help='Artifact path below --base-url and --dest. Supports {hash}. '
                         'Default: {hash}/fmha/trtllm-gen')
    ap.add_argument('--base-url', default=DEFAULT_BASE_URL,
                    help='Artifactory base URL (override for mirrors / testing)')
    ap.add_argument('--filter', action='append', default=[],
                    help='Restrict downloads to filenames matching at least one regex. '
                         'Pass multiple times to AND-combine.  Without --filter every '
                         '.cubin in checksums.txt is downloaded.')
    ap.add_argument('--workers', type=int, default=8,
                    help='Parallel download workers (default 8)')
    ap.add_argument('--quiet', action='store_true',
                    help='Suppress per-file logging; only print summary')
    args = ap.parse_args()

    artifact_path = args.artifact_path.format(hash=args.hash).strip('/')
    if artifact_path.startswith('/') or '..' in Path(artifact_path).parts:
        raise SystemExit(f'Unsafe artifact path: {artifact_path!r}')

    cubin_dir = args.dest / artifact_path
    cubin_dir.mkdir(parents=True, exist_ok=True)

    base = f'{args.base_url}/{artifact_path}'
    checksums_path = cubin_dir / 'checksums.txt'

    # Always refresh checksums.txt (it's tiny; ensures we don't miss new
    # cubins after a hash bump).
    if not args.quiet:
        print(f'[cubins] fetching {base}/checksums.txt')
    checksums_path.write_bytes(fetch_text(f'{base}/checksums.txt').encode('utf-8'))
    entries = parse_checksums(checksums_path.read_text())

    # Apply name filters (AND).
    if args.filter:
        compiled = [re.compile(p) for p in args.filter]
        entries = [(s, n) for s, n in entries
                   if all(c.search(n) for c in compiled)]

    if not args.quiet:
        print(f'[cubins] {len(entries)} entries selected (cubins + metainfo headers)')

    # Pre-scan to skip already-up-to-date files.
    todo: list[tuple[str, str]] = []
    skipped = 0
    for sha, name in entries:
        local = cubin_dir / name
        if local.exists() and sha256_of(local) == sha:
            skipped += 1
            continue
        todo.append((sha, name))
    if not args.quiet:
        print(f'[cubins] {skipped} already cached, {len(todo)} to fetch')

    if not todo:
        return 0

    failed: list[str] = []

    def _one(item: tuple[str, str]) -> str | None:
        sha, name = item
        url = f'{base}/{name}'
        local = cubin_dir / name
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetch_binary(url, local)
        except SystemExit as e:
            return f'{name}: {e}'
        actual = sha256_of(local)
        if actual != sha:
            try:
                local.unlink()
            except OSError:
                pass
            return f'{name}: sha mismatch (expected {sha[:12]}…, got {actual[:12]}…)'
        if not args.quiet:
            print(f'[cubins] OK {name}')
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for err in ex.map(_one, todo):
            if err is not None:
                failed.append(err)

    if failed:
        print(f'[cubins] {len(failed)} failed:', file=sys.stderr)
        for f in failed:
            print(f'  {f}', file=sys.stderr)
        return 1

    if not args.quiet:
        print(f'[cubins] all {len(todo)} cubins downloaded into {cubin_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
