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

DEFAULT_MODEL = "src.model.beemo_poyo.BeemoPOYO"


def data_config(recording):
    if "jenkins" in recording:
        return "churchland_ft", 64
    if "jango" in recording or "spike" in recording:
        return "ma_miller_ft", 32
    return "perich_miller_ft", 32


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained checkpoint on single recordings."
    )
    parser.add_argument(
        "--ckpt", required=True, help="Path to the checkpoint to fine-tune from"
    )
    parser.add_argument("--model", default="backbone", help="Hydra model config")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Model class path")
    parser.add_argument(
        "--project", default="beemo-finetune", help="WandB project to log to"
    )
    parser.add_argument("--recordings", nargs="+", default=RECORDINGS)
    parser.add_argument("--unfreeze-at-epoch", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--data-root", help="Override data_root from the config")
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Use the few-shot split (valid 0.2 / test 0.6)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_overrides = [f"model={args.model}", f"model.name={args.model_name}"]
    ckpt = args.ckpt
    path = Path(args.ckpt)
    tag = f"{path.parent.name}_{path.stem}".strip("_")

    failed = []
    for rec in args.recordings:
        data_name, batch_size = data_config(rec)

        cli = [
            "python train.py",
            "--config-name finetune",
            f"wandb.project={args.project}",
            f"wandb.name={rec}_{tag}_unfreeze_at_{args.unfreeze_at_epoch}",
            "ckpt.enable=false",
            f"ckpt.load_from={ckpt}",
            f"data={data_name}",
            f"data.args.recording_ids=\"['{rec}']\"",
            "finetuning.enable=true",
            f"finetuning.unfreeze_at_epoch={args.unfreeze_at_epoch}",
            f"num_epochs={args.num_epochs}",
            f"batch_size={batch_size}",
            "model.args.ctx_duration=1",
        ] + model_overrides

        if args.few_shot:
            cli += ["+valid_size=0.2", "+test_size=0.6"]
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
