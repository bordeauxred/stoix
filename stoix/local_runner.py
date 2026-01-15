#!/usr/bin/env python3
"""
Local multi-seed experiment runner for Stoix.

This script runs experiments locally with multiple seeds, either sequentially
or in parallel using multiprocessing. It automatically enables JSON logging
for downstream plotting.

Usage:
    # Run a single algorithm on a single environment with 5 seeds
    python stoix/local_runner.py \
        --algorithm stoix/systems/ppo/anakin/ff_ppo.py \
        --env gymnax/cartpole \
        --seeds 42 43 44 45 46 \
        --experiment_name ppo_cartpole_baseline

    # Run multiple algorithms for comparison
    python stoix/local_runner.py \
        --algorithm stoix/systems/ppo/anakin/ff_ppo.py stoix/systems/q_learning/ff_dqn.py \
        --env gymnax/cartpole \
        --seeds 42 43 44 \
        --experiment_name algo_comparison

    # Run in parallel (be careful with GPU memory)
    python stoix/local_runner.py \
        --algorithm stoix/systems/ppo/anakin/ff_ppo.py \
        --env gymnax/cartpole \
        --seeds 42 43 44 45 46 \
        --parallel 2 \
        --experiment_name parallel_test
"""

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def run_single_experiment(
    algorithm: str,
    env: str,
    seed: int,
    experiment_name: str,
    extra_args: List[str],
    results_dir: str = "results",
) -> Tuple[str, int, bool, str]:
    """Run a single experiment.

    Args:
        algorithm: Path to the algorithm script
        env: Environment config name
        seed: Random seed
        experiment_name: Name for grouping experiments
        extra_args: Additional hydra overrides
        results_dir: Base results directory

    Returns:
        Tuple of (algorithm, seed, success, output/error message)
    """
    # Build the command with JSON logging enabled
    cmd = [
        sys.executable,
        algorithm,
        f"env={env}",
        f"arch.seed={seed}",
        # Enable JSON logging for plotting
        "logger.loggers.json.enabled=True",
        f"logger.loggers.json.path={experiment_name}",
        f"logger.base_exp_path={results_dir}",
    ]

    # Add any extra arguments
    cmd.extend(extra_args)

    algo_name = Path(algorithm).stem
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {algo_name} | env={env} | seed={seed}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {algo_name} | seed={seed}")
        return (algorithm, seed, True, result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED: {algo_name} | seed={seed}")
        return (algorithm, seed, False, f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")


def run_experiments(
    algorithms: List[str],
    envs: List[str],
    seeds: List[int],
    experiment_name: str,
    parallel: int = 1,
    extra_args: Optional[List[str]] = None,
    results_dir: str = "results",
) -> None:
    """Run multiple experiments.

    Args:
        algorithms: List of algorithm script paths
        envs: List of environment config names
        seeds: List of random seeds
        experiment_name: Name for grouping experiments
        parallel: Number of parallel workers (1 = sequential)
        extra_args: Additional hydra overrides
        results_dir: Base results directory
    """
    extra_args = extra_args or []

    # Create all experiment combinations
    experiments = list(itertools.product(algorithms, envs, seeds))
    total = len(experiments)

    print(f"\n{'='*60}")
    print(f"Running {total} experiments")
    print(f"  Algorithms: {[Path(a).stem for a in algorithms]}")
    print(f"  Environments: {envs}")
    print(f"  Seeds: {seeds}")
    print(f"  Experiment name: {experiment_name}")
    print(f"  Parallel workers: {parallel}")
    print(f"  Results dir: {results_dir}")
    print(f"{'='*60}\n")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    results = []
    failed = []

    if parallel <= 1:
        # Sequential execution
        for i, (algo, env, seed) in enumerate(experiments):
            print(f"\n[{i+1}/{total}]")
            algo_name, seed_val, success, msg = run_single_experiment(
                algo, env, seed, experiment_name, extra_args, results_dir
            )
            results.append((algo_name, seed_val, success))
            if not success:
                failed.append((algo_name, seed_val, msg))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    run_single_experiment,
                    algo,
                    env,
                    seed,
                    experiment_name,
                    extra_args,
                    results_dir,
                ): (algo, seed)
                for algo, env, seed in experiments
            }

            for future in as_completed(futures):
                algo_name, seed_val, success, msg = future.result()
                results.append((algo_name, seed_val, success))
                if not success:
                    failed.append((algo_name, seed_val, msg))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results) - len(failed)}/{total} experiments succeeded")
    if failed:
        print(f"\nFailed experiments:")
        for algo, seed, msg in failed:
            print(f"  - {Path(algo).stem} seed={seed}")
            if "--verbose" in sys.argv:
                print(f"    Error: {msg[:500]}...")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}/json/{experiment_name}/")
    print(f"To plot results, run:")
    print(f"  python plotting/plot_seeds.py --path {results_dir}/json/{experiment_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stoix experiments with multiple seeds locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--algorithm", "-a",
        nargs="+",
        required=True,
        help="Path(s) to algorithm script(s)",
    )
    parser.add_argument(
        "--env", "-e",
        nargs="+",
        required=True,
        help="Environment config name(s)",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--experiment_name", "-n",
        required=True,
        help="Name for this experiment group (used for results directory)",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--results_dir", "-r",
        default="results",
        help="Base results directory (default: results)",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra hydra overrides (e.g., system.epochs=10)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed error messages",
    )

    args = parser.parse_args()

    run_experiments(
        algorithms=args.algorithm,
        envs=args.env,
        seeds=args.seeds,
        experiment_name=args.experiment_name,
        parallel=args.parallel,
        extra_args=args.extra,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
