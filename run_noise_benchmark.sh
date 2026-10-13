#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, inside the server's Python/conda environment.
python noise_benchmark.py run "$@"
