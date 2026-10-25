#!/usr/bin/env python3

import argparse
import subprocess
import sys

RECORDINGS = [
    "t_20130819_center_out_reaching",
    "t_20130820_random_target_reaching",
    "t_20130821_center_out_reaching",
    "t_20130822_random_target_reaching",
    "t_20130823_center_out_reaching",
    "t_20130830_random_target_reaching",
    "t_20130903_center_out_reaching",
    "t_20130904_random_target_reaching",
    "t_20130905_center_out_reaching",
    "t_20130906_random_target_reaching",
    "t_20130909_center_out_reaching",
    "t_20130910_random_target_reaching",
    "spike_20121023_isometric_wrist",
    "spike_20121025_isometric_wrist",
    "spike_20121105_isometric_wrist",
    "spike_20121107_isometric_wrist",
    "spike_20121108_isometric_wrist",
    "spike_20121109_isometric_wrist",
    "spike_20121112_isometric_wrist",
    "jango_20150905_isometric_wrist",
    "jango_20150908_isometric_wrist",
    "jango_20151029_isometric_wrist",
    "jango_20151102_isometric_wrist",
    "jenkins_20090912_maze",
    "jenkins_20090916_maze",
    "jenkins_20090918_maze",
    "jenkins_20090923_maze",
]

MODELS = {
    "poyo": ("sweep", ["model=poyo"]),
    "ndt": ("sweep", ["model=ndt"]),
    "linear": ("sweep_baselines", ["model=linear", "trainer=baselines"]),
    "mlp": ("sweep_baselines", ["model=mlp", "trainer=baselines"]),
    "gru": ("sweep_baselines", ["model=gru", "trainer=baselines"]),
}


def data_config(recording):
    if "jenkins" in recording:
        return "churchland_ft"
    if "jango" in recording or "spike" in recording:
        return "ma_miller_ft"
    return "perich_miller_ft"


def main():
    parser = argparse.ArgumentParser(
        description="Run a single-session hyperparameter sweep locally."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument(
        "--project", help="WandB project (default: beemo-sweep-<model>)"
    )
    parser.add_argument("--recordings", nargs="+", default=RECORDINGS)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--data-root", help="Override data_root from the config")
    parser.add_argument("--gpu-req", type=float, default=0.25)
    parser.add_argument("--cpu-req", type=int, default=4)
    parser.add_argument("--num-samples", type=int, help="Optuna trials per recording")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_name, model_args = MODELS[args.model]
    project = args.project or f"beemo-sweep-{args.model}"

    failed = []
    for rec in args.recordings:
        cli = [
            "python tune.py",
            f"--config-name {config_name}",
            f"wandb.project={project}",
            f"wandb.name={rec}",
            f"data={data_config(rec)}",
            f"data.args.recording_ids=\"['{rec}']\"",
            "ray.address=local",
            f"ray.gpu_req={args.gpu_req}",
            f"ray.cpu_req={args.cpu_req}",
            "ckpt.enable=false",
            f"seed={args.seed}",
        ] + model_args

        if args.num_samples is not None:
            cli.append(f"ray.num_samples={args.num_samples}")
        if args.data_root:
            cli.append(f"data_root={args.data_root}")

        cmd = " ".join(cli)

        if args.dry_run:
            print(cmd)
            continue

        print(f"\n=== {rec} ===", flush=True)
        if subprocess.run(cmd, shell=True).returncode != 0:
            print(f"FAILED {rec}", file=sys.stderr)
            failed.append(rec)

    if failed:
        print(f"\n{len(failed)} recording(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
