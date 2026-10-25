# BEEMO

Official codebase for BEEMO as presented in "Scaling Neural Motor Decoding via Decoupled Behavioral Pretraining"

BEEMO learns a behavior prior across many sessions and reuses it when decoding behavior from spikes on a new recording. The prior can be learned together with the neural data, or decoupled from it and learned on behavior alone, with a neural encoder attached afterwards.

This repository supports two neural encoders: POYO, a latent-bottleneck transformer with spike tokenization, and NDT, which uses binned tokenization.

## Usage

This project has been developed on Python3.11, with environment management using [`uv`](https://docs.astral.sh/uv/getting-started/installation/). To setup the environment, do:

```bash
source utils/venv_setup.sh
```

This also installs `xformers`, which needs a CUDA GPU. Runs use `precision=bf16` or `fp16`; fp32 is not supported by the attention kernels.

Follow the steps below to pretrain and finetune your own BEEMO model.

### 1. Preprocessing Data

To preprocess datasets, run `brainsets prepare` for each brainset:

```bash
brainsets prepare perich_miller_population_2018 --raw-dir ../data/raw/ ../data/processed/ --cores 8
brainsets prepare gallego_carracedo_lfp_2022     --raw-dir ../data/raw/ ../data/processed/ --cores 8
brainsets prepare ma_miller_adversarial_bci_2023 --raw-dir ../data/raw/ ../data/processed/ --cores 8
brainsets prepare churchland_shenoy_neural_2012  --raw-dir ../data/raw/ ../data/processed/ --cores 8
```

Pretraining uses the first three. Churchland is only needed for the Jenkins finetuning recordings.

### 2. Pretraining

To pretrain on behavior and neural data together:

```bash
python train.py ckpt.enable=true
```

To pretrain on behavior alone, which is the first stage of the decoupled route:

```bash
python train.py --config-name beemo_behavior_only_pretrain ckpt.enable=true
```

To pretrain on neural data alone, without behavioral pretraining:

```bash
python train.py --config-name base_pretrain ckpt.enable=true
```

- We use [Hydra](https://hydra.cc/) for managing configs.
- Options for `model=` can be found in `configs/model/*.yaml`. E.g. `model=beemo_ndt` to use binned tokenization instead of spike tokenization.
- `model.args.num_neural_sessions` sets how many sessions contribute spikes. The rest contribute behavior only. This is what the scaling study varies.
- The data would be read from `../data/processed` by default. Set `data_root` to change this.
- The checkpoints would be stored in `../ckpt/<run_id>/epoch_<num>.pt` by default. Set `ckpt.dir` or `$CKPT_DIR` to change this.
- Pretraining runs at `batch_size=256` and `num_epochs=100`. The learning rate in the config is scaled by `sqrt(batch_size)` at startup.
- `train.py` would use one process per visible GPU. Use `CUDA_VISIBLE_DEVICES` to restrict it, or `ddp.force=true` to use DDP on a single GPU.
- Logging goes to your own wandb account. Set `wandb.entity` to log elsewhere, or `wandb.mode=disabled` to turn it off.
- Other available configurations can be found in `configs/base.yaml` and `configs/beemo_pretrain.yaml`.

### 3. Finetuning

The training script would print a "run_id" for the corresponding run. Use this to finetune the checkpoint on each held-out recording:

```bash
python utils/run_finetune.py --ckpt ../ckpt/<run_id>/epoch_299.pt
```

For the decoupled route, use a behavior-only checkpoint and train a fresh neural encoder on top of it:

```bash
python utils/run_neural_finetune.py --ckpt ../ckpt/<run_id>/epoch_299.pt --encoder poyo
```

- Both scripts loop over all 27 recordings. Pass `--recordings` to run a subset, and `--dry-run` to print the commands without running them.
- `--unfreeze-at-epoch` controls gradual unfreezing. `-1` unfreezes everything immediately.
- `run_finetune.py --few-shot` switches to the few-shot split.
- `run_neural_finetune.py --encoder ndt` uses the NDT encoder.

To finetune on a single recording:

```bash
python train.py --config-name finetune \
    ckpt.load_from=../ckpt/<run_id>/epoch_299.pt \
    data=perich_miller_ft data.args.recording_ids="['t_20130820_random_target_reaching']" \
    finetuning.enable=true finetuning.unfreeze_at_epoch=-1
```

### 4. Single-session Baselines

To train one recording from scratch:

```bash
python train.py --config-name single_session data.args.recording_ids="['t_20130819_center_out_reaching']"
```

To sweep hyperparameters per recording with Optuna:

```bash
python utils/run_sweep.py --model poyo
```

- Options for `--model` are `poyo`, `ndt`, `linear`, `mlp` and `gru`.
- Ray runs locally. `--gpu-req` and `--cpu-req` set the per-trial resource share, and `--num-samples` the number of trials per recording.
- Each model defines its own search space as `create_search_space` and `process_tunable_params`.

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation

```bib
TODO
```

## Acknowledgments

> [!NOTE]
> This project utilizes code from and was influenced by [nuclr](https://github.com/nerdslab/nuclr) by Vinam Arora. The baselines code was adapted from [ibl-bwb](https://github.com/brainbench-org/ibl-bwb) by Alexander Andre and Shivasriganesh Mahato.
