#!/usr/bin/env bash
set -e
CONFIG='{"num-supernodes":10,"backend":{"client_resources":{"num_cpus":4,"num_gpus":0}}}'
echo ">>> Configuring Flower simulation: max 2 concurrent actors (4 CPUs each)"
~/.local/bin/flwr federation simulation-config local-simulation "$CONFIG"
echo "Done. Run 'flwr run .' normally — concurrency is now limited to 2 clients at a time."
