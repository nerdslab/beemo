#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

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

ENCODERS = {
    "poyo": ("src.model.beemo_poyo_behavior_only.BeemoPOYOBehaviorOnly", []),
    "ndt": (
        "src.model.beemo_ndt_behavior_only.BeemoNDTBehaviorOnly",
        ["+model.args.bin_size=0.02"],
    ),
}


def data_config(recording):
    if "jenkins" in recording:
        return "churchland_ft", 64
    if "jango" in recording or "spike" in recording:
        return "ma_miller_ft", 32
    return "perich_miller_ft", 32


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a neural encoder onto a behaviour-only checkpoint."
    )
    parser.add_argument(
        "--ckpt", required=True, help="Path to the behaviour-only checkpoint"
    )
    parser.add_argument("--encoder", default="poyo", choices=sorted(ENCODERS))
    parser.add_argument(
        "--project", help="WandB project (default: beemo-neural-ft-<encoder>)"
    )
    parser.add_argument("--recordings", nargs="+", default=RECORDINGS)
    parser.add_argument("--unfreeze-at-epoch", type=int, default=30)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--data-root", help="Override data_root from the config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    model_name, encoder_args = ENCODERS[args.encoder]
    project = args.project or f"beemo-neural-ft-{args.encoder}"

    model_overrides = ["model=backbone"]
    ckpt = args.ckpt
    path = Path(args.ckpt)
    tag = f"{path.parent.name}_{path.stem}".strip("_")

    model_overrides += [
        f"model.name={model_name}",
        "model.args.ctx_duration=1",
        f"model.args.depth={args.depth}",
        f"model.args.num_heads={args.num_heads}",
    ] + encoder_args

    failed = []
    for rec in args.recordings:
        data_name, batch_size = data_config(rec)

        cli = [
            "python train.py",
            "--config-name neural_finetune",
            f"wandb.project={project}",
            f"wandb.name={rec}_neural_ft_{tag}",
            "ckpt.enable=false",
            f"ckpt.load_from={ckpt}",
            f"data={data_name}",
            f"data.args.recording_ids=\"['{rec}']\"",
            "finetuning.enable=true",
            f"finetuning.unfreeze_at_epoch={args.unfreeze_at_epoch}",
            f"num_epochs={args.num_epochs}",
            f"batch_size={batch_size}",
        ] + model_overrides

        if args.data_root:
            cli.append(f"data_root={args.data_root}")
        cli += args.extra

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
