#!/bin/bash
# Phase 0.1 acceptance suite. Run from repo root.
set -euo pipefail

echo "=== 1. CPU validator (no GPU deps) ==="
python3 tests/test_validator.py

echo "=== 2. single-GPU DAG equivalence ==="
python3 tests/test_events_single_gpu.py 0
python3 tests/test_events_single_gpu.py 1
python3 tests/test_events_single_gpu.py 2

echo "=== 3. dual-GPU ping-pong / fan-in / fan-out ==="
torchrun --nproc_per_node=2 tests/test_events_dual_gpu.py 1000

echo "=== 4. latency benchmark (acceptance) ==="
python3 tests/bench_notify_latency.py 10000
torchrun --nproc_per_node=2 tests/bench_notify_latency.py 10000

echo "PHASE 0.1 ALL PASS"
