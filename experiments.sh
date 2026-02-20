#!/usr/bin/env bash
set -e

AGENTS_LIST="4 8 16"
SEED_LIST="1 2 3"

for NA in $AGENTS_LIST; do
  for SEED in $SEED_LIST; do
    echo "Running experiment: num_agents=$NA, seed=$SEED"

    # Write config.json
    cat << CONFIG > config.json
{
  "world_width": 30,
  "world_height": 20,
  "obstacle_density": 0.12,
  "num_victims": 10,
  "num_agents": $NA,
  "comm_range": 5,
  "sensing_range": 3,
  "max_steps": 300,
  "seed": $SEED,
  "render_interval": 0
}
CONFIG

    # Run simulation
    python main.py
  done
done

echo "All experiments completed. Check logs/metrics_*.csv."
