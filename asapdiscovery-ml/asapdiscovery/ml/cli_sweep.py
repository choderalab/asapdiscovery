import json
from glob import glob
from pathlib import Path

import click
import pydantic
import torch
from asapdiscovery.data.util.utils import MOONSHOT_CDD_ID_REGEX, MPRO_ID_REGEX
from asapdiscovery.ml.cli_args import (
    ds_cache_overwrite,
    ds_config_cache_overwrite,
    ds_split_args,
    e3nn_args,
    es_args,
    gat_args,
    graph_ds_args,
    grouped,
    loss_args,
    model_config_cache,
    model_rand_seed,
    mtenn_args,
    optim_args,
    output_dir,
    overwrite_args,
    schnet_args,
    struct_ds_args,
    trainer_args,
    trainer_config_cache,
    visnet_args,
    wandb_args,
    weights_path,
)
from asapdiscovery.ml.config import (
    DatasetConfig,
    DatasetSplitterType,
    EarlyStoppingType,
    LossFunctionType,
    OptimizerType,
)
from asapdiscovery.ml.trainer import Trainer
from mtenn.config import CombinationConfig, ModelType, ReadoutConfig, StrategyConfig


@click.group()
def sweep():
    pass

@sweep.command(name="gat")
@output_dir
@weights_path
@trainer_config_cache
@optim_args
@wandb_args
@model_config_cache
@model_rand_seed
@mtenn_args
@gat_args
@es_args
@graph_ds_args
@ds_split_args
@loss_args
@trainer_args
@overwrite_args
def build_gat(
    output_dir: Path | None = None,
    weights_path: Path | None = None,
    trainer_config_cache: Path | None = None,
    optimizer_type: OptimizerType | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    dampening: float | None = None,
    b1: float | None = None,
    b2: float | None = None,
    eps: float | None = None,
    rho: float | None = None,
    optimizer_config_cache: Path | None = None,
    use_wandb: bool | None = None,
    sweep: bool | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: list[str] | None = None,
    grouped: bool | None = None,
    strategy: StrategyConfig | None = None,
    pred_readout: ReadoutConfig | None = None,
    combination: CombinationConfig | None = None,
    comb_readout: ReadoutConfig | None = None,
    max_comb_neg: bool | None = None,
    max_comb_scale: float | None = None,
    pred_substrate: float | None = None,
    pred_km: float | None = None,
    comb_substrate: float | None = None,
    comb_km: float | None = None,
    model_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
    in_feats: int | None = None,
    num_layers: int | None = None,
    hidden_feats: str | None = None,
    num_heads: str | None = None,
    feat_drops: str | None = None,
    attn_drops: str | None = None,
    alphas: str | None = None,
    residuals: str | None = None,
    agg_modes: str | None = None,
    biases: str | None = None,
    allow_zero_in_degree: bool | None = None,
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_config_cache: Path | None = None,
    exp_file: Path | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_config_cache: Path | None = None,
    loss_type: LossFunctionType | None = None,
    semiquant_fill: float | None = None,
    loss_config_cache: Path | None = None,
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    overwrite_loss_config_cache: bool = False,
):
    # First check if Trainer cache exists and skip everything else if so
    if (
        trainer_config_cache
        and trainer_config_cache.exists()
        and (not overwrite_trainer_config_cache)
    ):
        print("loaded trainer from cache", flush=True)
        t = Trainer(**json.loads(trainer_config_cache.read_text()))
    else:
        # Build each dict and pass to Trainer
        optim_config = {
            "cache": optimizer_config_cache,
            "overwrite_cache": overwrite_optimizer_config_cache,
            "optimizer_type": optimizer_type,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "dampening": dampening,
            "b1": b1,
            "b2": b2,
            "eps": eps,
            "rho": rho,
        }
        model_config = {
            "cache": model_config_cache,
            "overwrite_cache": overwrite_model_config_cache,
            "model_type": ModelType.GAT,
            "rand_seed": model_rand_seed,
            "weights_path": weights_path,
            "grouped": grouped,
            "strategy": strategy,
            "pred_readout": pred_readout,
            "combination": combination,
            "comb_readout": comb_readout,
            "max_comb_neg": max_comb_neg,
            "max_comb_scale": max_comb_scale,
            "pred_substrate": pred_substrate,
            "pred_km": pred_km,
            "comb_substrate": comb_substrate,
            "comb_km": comb_km,
            "in_feats": in_feats,
            "num_layers": num_layers,
            "hidden_feats": hidden_feats,
            "num_heads": num_heads,
            "feat_drops": feat_drops,
            "attn_drops": attn_drops,
            "alphas": alphas,
            "residuals": residuals,
            "agg_modes": agg_modes,
            "biases": biases,
            "allow_zero_in_degree": allow_zero_in_degree,
        }
        es_config = {
            "cache": es_config_cache,
            "overwrite_cache": overwrite_es_config_cache,
            "es_type": es_type,
            "patience": es_patience,
            "n_check": es_n_check,
            "divergence": es_divergence,
        }
        ds_config = {
            "cache": ds_config_cache,
            "overwrite_cache": overwrite_ds_config_cache,
            "exp_file": exp_file,
            "is_structural": False,
            "cache_file": ds_cache,
            "overwrite": overwrite_ds_cache,
        }

        ds_splitter_config = {
            "cache": ds_split_config_cache,
            "overwrite_cache": overwrite_ds_split_config_cache,
            "split_type": ds_split_type,
            "grouped": grouped,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "enforce_one": enforce_one,
            "rand_seed": ds_rand_seed,
        }
        loss_config = {
            "cache": loss_config_cache,
            "overwrite_cache": overwrite_loss_config_cache,
            "loss_type": loss_type,
            "semiquant_fill": semiquant_fill,
        }

        # Parse loss_dict
        if loss_dict:
            loss_dict = json.loads(loss_dict.read_text())

        # Filter out None Trainer kwargs
        trainer_kwargs = {
            "optimizer_config": optim_config,
            "model_config": model_config,
            "es_config": es_config,
            "ds_config": ds_config,
            "ds_splitter_config": ds_splitter_config,
            "loss_config": loss_config,
            "auto_init": auto_init,
            "start_epoch": start_epoch,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "target_prop": target_prop,
            "cont": cont,
            "loss_dict": loss_dict,
            "device": device,
            "output_dir": output_dir,
            "use_wandb": use_wandb,
            "sweep": sweep,
            "wandb_project": wandb_project,
            "wandb_name": wandb_name,
            "extra_config": extra_config,
        }
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

        try:
            t = Trainer(**trainer_kwargs)
        except pydantic.ValidationError as exc:
            # Only want to handle missing values, so if anything else went wrong just raise
            #  the pydantic error
            if any([err["type"] != "value_error.missing" for err in exc.errors()]):
                raise exc

            # Gather all missing values
            missing_vals = [err["loc"][0] for err in exc.errors()]

            raise ValueError(
                "Tried to build Trainer but missing required values: ["
                + ", ".join(missing_vals)
                + "]"
            )

        # Save Trainer
        if trainer_config_cache and (
            (not trainer_config_cache.exists()) or overwrite_trainer_config_cache
        ):
            trainer_config_cache.write_text(t.json())