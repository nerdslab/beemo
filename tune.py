from typing import Dict, Any
from functools import partial
import ray
import hydra
from ray import tune, air, train
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
import optuna

from src.utils import get_class_from_path
from omegaconf import DictConfig, OmegaConf
import copy
import torch.nn as nn


def _fill_config(
    tune_params: dict, cfg: DictConfig, process_params: bool = True
) -> DictConfig:
    # process tunable parameters for the model
    model_cls = get_class_from_path(cfg.model.name)
    if process_params and model_cls is not None and issubclass(model_cls, nn.Module):
        tune_params = model_cls.process_tunable_params(tune_params)

    # fill config with tuned parameters
    config = copy.deepcopy(cfg)
    for key, value in tune_params.items():
        OmegaConf.update(config, key, value)
    return config


def run_training_hyperparameter(tune_params, cfg):
    from train import train_ddp

    cfg = _fill_config(tune_params, cfg)

    # hidden_dim = cfg.model.args.dim
    # num_heads = cfg.model.args.num_heads
    # head_dim = hidden_dim // num_heads

    # # # FlashAttention constraint
    # if head_dim % 8 != 0:
    #     raise optuna.exceptions.TrialPruned()

    cfg.num_workers = int(cfg.ray.cpu_req)

    metrics = train_ddp(cfg)  # type: Dict[str, Any]  # type: ignore

    train.report(metrics)
    session.report(metrics)

    return metrics


@hydra.main(version_base="1.3", config_path="./configs", config_name="sweep.yaml")
def main(cfg):
    # trials run from their own directories, so relative paths must be resolved here
    cfg.data_root = hydra.utils.to_absolute_path(cfg.data_root)

    ray.init(address=cfg.ray.address, log_to_driver=cfg.ray.log_to_driver)
    resources_arr = {"cpu": cfg.ray.cpu_req, "gpu": cfg.ray.gpu_req}
    num_samples = cfg.ray.num_samples

    model_cls = get_class_from_path(cfg.model.name)

    search_space = partial(model_cls.create_search_space, cfg=cfg)
    algo = OptunaSearch(
        space=search_space,
        metric="best/val/avg",
        mode="max",
    )

    print(f"Will tune for {num_samples} trials")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(run_training_hyperparameter, cfg=cfg)),
            resources=resources_arr,
        ),
        tune_config=tune.TuneConfig(
            metric="best/val/avg",
            mode="max",
            search_alg=algo,
            num_samples=num_samples,
            max_concurrent_trials=cfg.ray.max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            failure_config=tune.FailureConfig(max_failures=0),
        ),
    )

    results = tuner.fit()
    best_trial = results.get_best_result(metric="best/val/avg", mode="max")

    print("Best trial config: {}".format(best_trial.config))
    print(best_trial)
    print("Best trial directory path:", best_trial.path)


if __name__ == "__main__":
    main()
