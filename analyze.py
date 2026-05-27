#!/usr/bin/env python3
import json
import os
from pathlib import Path

log_dir = Path("/home/build/ollama/tmp-logs")

results = {}

for kv_type in ["q8_0", "turbo3", "turbo2", "iso3", "planar3"]:
    json_file = log_dir / f"{kv_type}.json"
    
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        load_duration_ns = data.get("load_duration", 0)
        total_duration_ns = data.get("total_duration", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        
        # Convert to ms
        eval_ms = eval_duration_ns / 1_000_000
        load_ms = load_duration_ns / 1_000_000
        total_ms = total_duration_ns / 1_000_000
        
        # Throughput
        throughput = (1000 * eval_count) / eval_ms if eval_ms > 0 else 0
        
        results[kv_type] = {
            "eval_count": eval_count,
            "prompt_count": prompt_eval_count,
            "eval_ms": eval_ms,
            "load_ms": load_ms,
            "total_ms": total_ms,
            "throughput": throughput,
        }

# Print results
print("KV Cache Type Benchmark Results")
print("=" * 90)
print()
print(f"{'KV Type':<12} | {'Eval Count':>10} | {'Eval (ms)':>10} | {'Throughput':>12} | {'Load (ms)':>10} | {'Total (ms)':>10}")
print("-" * 90)

for kv_type in ["q8_0", "turbo3", "turbo2", "iso3", "planar3"]:
    if kv_type in results:
        r = results[kv_type]
        print(f"{kv_type:<12} | {r['eval_count']:>10} | {r['eval_ms']:>10.1f} | {r['throughput']:>12.1f} tok/s | {r['load_ms']:>10.1f} | {r['total_ms']:>10.1f}")

print()
print("Performance Summary")
print("-" * 90)

# Find best throughput
best_type = max(results.keys(), key=lambda k: results[k]["throughput"])
best_throughput = results[best_type]["throughput"]
baseline_throughput = results["q8_0"]["throughput"]

print(f"Baseline (q8_0):     {baseline_throughput:.1f} tok/s")
print(f"Best performer:      {best_type}: {best_throughput:.1f} tok/s")
print()

# Compare against baseline
print("Speedup vs q8_0 baseline:")
for kv_type in ["turbo3", "turbo2", "iso3", "planar3"]:
    if kv_type in results:
        speedup = results[kv_type]["throughput"] / baseline_throughput
        improvement = (speedup - 1) * 100
        print(f"  {kv_type:<12}: {speedup:>6.2f}x ({improvement:>+6.1f}%)")

print()
print("Memory Impact (lower load time = less memory activity):")
baseline_load = results["q8_0"]["load_ms"]
for kv_type in ["turbo3", "turbo2", "iso3", "planar3"]:
    if kv_type in results:
        load_saving = ((baseline_load - results[kv_type]["load_ms"]) / baseline_load) * 100
        print(f"  {kv_type:<12}: {results[kv_type]['load_ms']:>8.1f} ms ({load_saving:>+6.1f}% vs baseline)")
