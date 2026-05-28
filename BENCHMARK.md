# KV Cache Benchmark Report

**Date (UTC):** 2026-05-28 10:57:41Z
**Model:** granite4.1:3b
**Binary:** build_optimized/ollama (built via `make -f Makefile.build build`)
**Runtime Flags:** OLLAMA_FLASH_ATTENTION=1, OLLAMA_GPU_LAYERS=999
**KV Policy:** key fixed at q8_0, value varied

## Executive Summary

| Config | Small Prompt tok/s | Small Gen tok/s | Medium Prompt tok/s | Medium Gen tok/s | GPU Validated | Status |
|---|---:|---:|---:|---:|---|---|
| q8_0/q8_0 | 173.00 | 87.46 | 191.20 | 85.29 | yes | ok |
| q8_0/turbo3_0 | 73.07 | 32.14 | 79.20 | 32.51 | yes | ok |
| q8_0/planar3 | 67.40 | 31.01 | 74.61 | 30.87 | yes | ok |
| q8_0/iso3 | 59.09 | 25.59 | 60.98 | 24.99 | yes | ok |

## Detailed Results

| Config | Test | Status | Prompt Tokens | Prompt s | Prompt tok/s | Gen Tokens | Gen s | Gen tok/s | Total s | CUDA Detected | CUDA Tensors | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| q8_0/q8_0 | small | ok | 16 | 0.0925 | 173.00 | 16 | 0.1829 | 87.46 | 1.7602 | 1 | 1 |  |
| q8_0/q8_0 | medium | ok | 16 | 0.0837 | 191.20 | 16 | 0.1876 | 85.29 | 1.7219 | 1 | 1 |  |
| q8_0/turbo3_0 | small | ok | 16 | 0.2190 | 73.07 | 16 | 0.4979 | 32.14 | 2.1049 | 1 | 1 |  |
| q8_0/turbo3_0 | medium | ok | 16 | 0.2020 | 79.20 | 16 | 0.4922 | 32.51 | 2.1209 | 1 | 1 |  |
| q8_0/planar3 | small | ok | 16 | 0.2374 | 67.40 | 16 | 0.5159 | 31.01 | 2.1408 | 1 | 1 |  |
| q8_0/planar3 | medium | ok | 16 | 0.2144 | 74.61 | 16 | 0.5182 | 30.87 | 2.0588 | 1 | 1 |  |
| q8_0/iso3 | small | ok | 16 | 0.2708 | 59.09 | 16 | 0.6252 | 25.59 | 2.2388 | 1 | 1 |  |
| q8_0/iso3 | medium | ok | 16 | 0.2624 | 60.98 | 16 | 0.6402 | 24.99 | 2.2201 | 1 | 1 |  |

## qwen3.6 turbo coherence

**Model:** qwen3.6:27b
**Context:** 131072
**Predict:** 1
**Batch:** 1
**Prompt:** Return exactly one lowercase word: apple.

| Test | KV Config | Status | Prompt Tokens | Prompt s | Prompt tok/s | Gen Tokens | Gen s | Gen tok/s | Total s | CUDA Validated | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| turbo2_turbo2 | turbo2_0/turbo2_0 | failed | 18 | 141.5018 | 0.13 | 1 | 0.0000 | 0.00 | 172.9223 | no |  |
| turbo3_turbo3 | turbo3_0/turbo3_0 | failed | 18 | 135.2497 | 0.13 | 1 | 0.0000 | 0.00 | 162.6854 | no |  |
| turbo4_turbo4 | turbo4_0/turbo4_0 | failed | 18 | 134.6301 | 0.13 | 1 | 0.0000 | 0.00 | 161.6165 | no |  |
| q8_0_turbo2 | q8_0/turbo2_0 | failed | 18 | 134.5736 | 0.13 | 1 | 0.0000 | 0.00 | 160.9967 | no |  |

## Interpretation

- One or more baseline configurations failed; see Detailed Results and Error column for exact failure points.
- Prompt tok/s corresponds to prefill throughput.
- Gen tok/s corresponds to decode throughput.
- qwen3.6 turbo pairs are reported separately below because they use a different model and KV layout.

## Method

- Single unified script: `benchmark-all.sh`
- Build phase: `make -f Makefile.build build`
- For each KV config and test size: fresh server start, single generate request, metrics parsed from response JSON.
- GPU validation required: CUDA inference log line and CUDA model tensor placement log line.

## Log Artifacts

- Raw benchmark TSV: /home/build/ollama/tmp-logs/bench_all_20260528_104550/results.tsv
- Per-run server and response logs are in the same run directory under `tmp-logs`.
