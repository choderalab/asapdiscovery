import collections
import json
from glob import glob
from pathlib import Path

import click
import torch
from asapdiscovery.data.util.utils import MOONSHOT_CDD_ID_REGEX, MPRO_ID_REGEX
from asapdiscovery.ml.cli_args import (
    ds_cache_overwrite,
    ds_config_cache_overwrite,
    ds_split_args,
    es_args,
    general_ds_args,
    graph_ds_args,
    kvp_list_to_dict,
    loss_args,
    model_config_cache,
    model_rand_seed,
    model_tag,
    mtenn_args,
    optim_args,
    output_dir,
    overwrite_args,
    representation_config_cache_args,
    s3_args,
    save_weights,
    struct_ds_args,
    trainer_args,
    trainer_config_cache,
    wandb_args,
    weights_path,
)

# from asapdiscovery.ml.cli_mlops import mlops
# from asapdiscovery.ml.cli_sweep import sweep
from asapdiscovery.ml.config import (
    DatasetConfig,
    DatasetSplitterType,
    DatasetType,
    EarlyStoppingType,
    OptimizerType,
)
from asapdiscovery.ml.trainer import Trainer
from mtenn.config import CombinationConfig, ModelType, ReadoutConfig, StrategyConfig
from pydantic import ValidationError


@click.group()
def ml():
    """Tools to build and train ML models and run inference."""
    pass


# Function for training using an already built Trainer
@ml.command()
def train():
    pass


# Functions for just building a Dataset and DatasetConfig
@ml.command(name="build-dataset")
@general_ds_args
@graph_ds_args
@struct_ds_args
@ds_cache_overwrite
@ds_config_cache_overwrite
def build_ds(
    dataset_type: DatasetType | None = None,
    export_input_data: bool | None = None,
    export_exp_data: bool | None = None,
    grouped_dataset: bool | None = None,
    e3nn_dataset: bool | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    ds_random_iter: bool | None = None,
    exp_file: Path | None = None,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
):
    if dataset_type is None:
        raise ValueError(
            "A value must be specified for --dataset-type when building a dataset."
        )

    is_structural = dataset_type != DatasetType.graph

    if not _check_ds_args(
        exp_file=exp_file,
        structures=structures,
        ds_cache=ds_cache,
        ds_config_cache=ds_config_cache,
        is_structural=is_structural,
        config_overwrite=overwrite_ds_config_cache,
    ):
        raise ValueError("Invalid combination of dataset args.")

    if ds_config_cache and ds_config_cache.exists() and (not overwrite_ds_config_cache):
        print("loading from cache", flush=True)
        return DatasetConfig(**json.loads(ds_config_cache.read_text()))

    config_kwargs = {
        "cache_file": ds_cache,
        "grouped": grouped_dataset,
        "for_e3nn": e3nn_dataset,
        "overwrite": overwrite_ds_cache,
        "export_input_data": export_input_data,
        "export_exp_data": export_exp_data,
        "random_iter": ds_random_iter,
    }
    config_kwargs = {
        k: v
        for k, v in config_kwargs.items()
        if not ((v is None) or (isinstance(v, tuple) and len(v) == 0))
    }

    # Pick correct DatasetType
    if is_structural:
        if (xtal_regex is None) or (cpd_regex is None):
            raise ValueError(
                "Must pass values for xtal_regex and cpd_regex if building a "
                "structure-based dataset."
            )
        ds_config = DatasetConfig.from_str_files(
            ds_type=dataset_type,
            structures=structures,
            xtal_regex=xtal_regex,
            cpd_regex=cpd_regex,
            for_training=True,
            exp_file=exp_file,
            **config_kwargs,
        )
    else:
        ds_config = DatasetConfig.from_exp_file(
            exp_file,
            **config_kwargs,
        )

    # Save file if desired
    if ds_config_cache:
        ds_config_cache.write_text(ds_config.model_dump_json())

    ds_config.build()


def _check_ds_args(
    exp_file, structures, ds_cache, ds_config_cache, is_structural, config_overwrite
):
    """
    Helper function to check that all necessary dataset files were passed.

    Parameters
    ----------
    exp_file : Path
        JSON file giving a list of ExperimentalDataCompound objects
    structures : Path
        Glob or directory containing PDB files
    ds_cache : Path
        Dataset cache file
    ds_config_cache : Path
        Dataset config cache function
    is_structural : bool
        Is this a structure-based dataset
    config_overwrite : bool
        Should any existing DatasetConfig JSON file be ignored/overwritten

    Returns
    -------
    bool
        Whether an appropriate combination of args was passed
    """
    # Can just load from the config cache file so don't need anything else
    if ds_config_cache and ds_config_cache.exists() and (not config_overwrite):
        return True

    # Otherwise need to load data so make sure they all exist
    if (not exp_file) or (not exp_file.exists()):
        return False
    if is_structural:
        if not structures:
            return False
        if Path(structures).is_dir():
            # Make sure there's at least one PDB file
            try:
                _ = next(iter(Path(structures).glob("*.pdb")))
            except StopIteration:
                return False
        else:
            # Make sure there's at least one file that matches the glob
            try:
                _ = next(iter(glob(structures)))
            except StopIteration:
                return False

    # Nothing has failed so we should be good to go
    return True


# Functions for just building a Trainer and then dumping it
@ml.command()
@output_dir
@save_weights
@weights_path
@trainer_config_cache
@optim_args
@wandb_args
@model_config_cache
@representation_config_cache_args
@model_rand_seed
@model_tag
@mtenn_args
@es_args
@general_ds_args
@graph_ds_args
@struct_ds_args
@ds_split_args
@loss_args
@trainer_args
@overwrite_args
@s3_args
def build(
    output_dir: Path | None = None,
    save_weights: str | None = None,
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
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: tuple[str] | None = None,
    model_type: ModelType | None = None,
    representation: str | None = None,
    complex_representation: str | None = None,
    ligand_representation: str | None = None,
    protein_representation: str | None = None,
    strategy: StrategyConfig | None = None,
    strategy_layer_norm: bool | None = None,
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
    representation_config_cache: Path | None = None,
    complex_representation_config_cache: Path | None = None,
    ligand_representation_config_cache: Path | None = None,
    protein_representation_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
    model_tag: str | None = None,
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_burnin: int | None = None,
    es_config_cache: Path | None = None,
    dataset_type: DatasetType | None = None,
    export_input_data: bool | None = None,
    export_exp_data: bool | None = None,
    grouped_dataset: bool | None = None,
    e3nn_dataset: bool | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    ds_random_iter: bool | None = None,
    exp_file: Path | None = None,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_dict: Path | None = None,
    ds_split_config_cache: Path | None = None,
    loss: tuple[str] = (),
    loss_weights: tuple[float] = (),
    eval_loss_weights: tuple[float] = (),
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    data_aug: tuple[str] = (),
    trainer_weight_decay: float | None = None,
    batch_norm: bool | None = None,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    s3_path: str | None = None,
    upload_to_s3: bool | None = None,
):
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
    representation = (
        kvp_list_to_dict(representation) if representation is not None else {}
    )
    representation |= {"cache": representation_config_cache}
    complex_representation = (
        kvp_list_to_dict(complex_representation)
        if complex_representation is not None
        else {}
    )
    complex_representation |= {"cache": complex_representation_config_cache}
    ligand_representation = (
        kvp_list_to_dict(ligand_representation)
        if ligand_representation is not None
        else {}
    )
    ligand_representation |= {"cache": ligand_representation_config_cache}
    protein_representation = (
        kvp_list_to_dict(protein_representation)
        if protein_representation is not None
        else {}
    )
    protein_representation |= {"cache": protein_representation_config_cache}
    model_config = {
        "cache": model_config_cache,
        "overwrite_cache": overwrite_model_config_cache,
        "model_type": model_type,
        "representation": representation,
        "complex_representation": complex_representation,
        "ligand_representation": ligand_representation,
        "protein_representation": protein_representation,
        "strategy": strategy,
        "strategy_layer_norm": strategy_layer_norm,
        "pred_readout": pred_readout,
        "combination": combination,
        "comb_readout": comb_readout,
        "max_comb_neg": max_comb_neg,
        "max_comb_scale": max_comb_scale,
        "pred_substrate": pred_substrate,
        "pred_km": pred_km,
        "comb_substrate": comb_substrate,
        "comb_km": comb_km,
        "rand_seed": model_rand_seed,
    }
    es_config = {
        "cache": es_config_cache,
        "overwrite_cache": overwrite_es_config_cache,
        "es_type": es_type,
        "patience": es_patience,
        "n_check": es_n_check,
        "divergence": es_divergence,
        "burnin": es_burnin,
    }
    if dataset_type is not None:
        is_structural = dataset_type != DatasetType.graph
    elif structures is not None:
        # Assume structural since structures were passed
        is_structural = True
    else:
        # Don't know what to do so punt and hope there's an appropriate cache file
        is_structural = None
    ds_config = {
        "cache": ds_config_cache,
        "overwrite_cache": overwrite_ds_config_cache,
        "ds_type": dataset_type,
        "structures": structures,
        "xtal_regex": xtal_regex,
        "cpd_regex": cpd_regex,
        "exp_file": exp_file,
        "is_structural": is_structural,
        "cache_file": ds_cache,
        "overwrite": overwrite_ds_cache,
        "export_input_data": export_input_data,
        "export_exp_data": export_exp_data,
        "grouped": grouped_dataset,
        "for_e3nn": e3nn_dataset,
        "random_iter": ds_random_iter,
    }

    ds_splitter_config = {
        "cache": ds_split_config_cache,
        "overwrite_cache": overwrite_ds_split_config_cache,
        "split_type": ds_split_type,
        "grouped": model_type == ModelType.grouped,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "enforce_one": enforce_one,
        "rand_seed": ds_rand_seed,
        "split_dict": ds_split_dict,
    }
    loss_configs = [kvp_list_to_dict(loss_str) for loss_str in loss]
    data_aug_configs = [kvp_list_to_dict(aug_str) for aug_str in data_aug]

    # Parse loss_dict
    if loss_dict:
        loss_dict = json.loads(loss_dict.read_text())

    # Gather all the configs
    trainer_kwargs = {
        "optimizer_config": optim_config,
        "mtenn_model_config": model_config,
        "es_config": es_config,
        "ds_config": ds_config,
        "ds_splitter_config": ds_splitter_config,
        "loss_configs": loss_configs,
        "loss_weights": loss_weights,
        "eval_loss_weights": eval_loss_weights,
        "data_aug_configs": data_aug_configs,
        "weight_decay": trainer_weight_decay,
        "batch_norm": batch_norm,
        "auto_init": auto_init,
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_prop": target_prop,
        "cont": cont,
        "loss_dict": loss_dict,
        "device": device,
        "output_dir": output_dir,
        "save_weights": save_weights,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
        "extra_config": Trainer.parse_extra_config(extra_config),
        "s3_path": s3_path,
        "upload_to_s3": upload_to_s3,
        "model_tag": model_tag,
    }

    _build_trainer(trainer_kwargs, trainer_config_cache, overwrite_trainer_config_cache)


# Functions for building a Trainer and subsequently training the model
@ml.command(name="build-and-train")
@output_dir
@save_weights
@weights_path
@trainer_config_cache
@optim_args
@wandb_args
@model_config_cache
@representation_config_cache_args
@model_rand_seed
@model_tag
@mtenn_args
@es_args
@general_ds_args
@graph_ds_args
@struct_ds_args
@ds_split_args
@loss_args
@trainer_args
@overwrite_args
@s3_args
def build_and_train(
    output_dir: Path | None = None,
    save_weights: str | None = None,
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
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: tuple[str] | None = None,
    model_type: ModelType | None = None,
    representation: str | None = None,
    complex_representation: str | None = None,
    ligand_representation: str | None = None,
    protein_representation: str | None = None,
    strategy: StrategyConfig | None = None,
    strategy_layer_norm: bool | None = None,
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
    representation_config_cache: Path | None = None,
    complex_representation_config_cache: Path | None = None,
    ligand_representation_config_cache: Path | None = None,
    protein_representation_config_cache: Path | None = None,
    model_rand_seed: int | None = None,
    model_tag: str | None = None,
    es_type: EarlyStoppingType | None = None,
    es_patience: int | None = None,
    es_n_check: int | None = None,
    es_divergence: float | None = None,
    es_burnin: int | None = None,
    es_config_cache: Path | None = None,
    exp_file: Path | None = None,
    structures: str | None = None,
    xtal_regex: str = MPRO_ID_REGEX,
    cpd_regex: str = MOONSHOT_CDD_ID_REGEX,
    dataset_type: DatasetType | None = None,
    export_input_data: bool | None = None,
    export_exp_data: bool | None = None,
    grouped_dataset: bool | None = None,
    e3nn_dataset: bool | None = None,
    ds_cache: Path | None = None,
    ds_config_cache: Path | None = None,
    ds_random_iter: bool | None = None,
    ds_split_type: DatasetSplitterType | None = None,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    enforce_one: bool | None = None,
    ds_rand_seed: int | None = None,
    ds_split_dict: Path | None = None,
    ds_split_config_cache: Path | None = None,
    loss: tuple[str] = (),
    loss_weights: tuple[float] = (),
    eval_loss_weights: tuple[float] = (),
    auto_init: bool | None = None,
    start_epoch: int | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    target_prop: str | None = None,
    cont: bool | None = None,
    loss_dict: dict | None = None,
    device: torch.device | None = None,
    data_aug: tuple[str] = (),
    trainer_weight_decay: float | None = None,
    batch_norm: bool | None = None,
    overwrite_trainer_config_cache: bool = False,
    overwrite_optimizer_config_cache: bool = False,
    overwrite_model_config_cache: bool = False,
    overwrite_es_config_cache: bool = False,
    overwrite_ds_config_cache: bool = False,
    overwrite_ds_cache: bool = False,
    overwrite_ds_split_config_cache: bool = False,
    s3_path: str | None = None,
    upload_to_s3: bool | None = None,
):
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
    representation = (
        kvp_list_to_dict(representation) if representation is not None else {}
    )
    representation |= {"cache": representation_config_cache}
    complex_representation = (
        kvp_list_to_dict(complex_representation)
        if complex_representation is not None
        else {}
    )
    complex_representation |= {"cache": complex_representation_config_cache}
    ligand_representation = (
        kvp_list_to_dict(ligand_representation)
        if ligand_representation is not None
        else {}
    )
    ligand_representation |= {"cache": ligand_representation_config_cache}
    protein_representation = (
        kvp_list_to_dict(protein_representation)
        if protein_representation is not None
        else {}
    )
    protein_representation |= {"cache": protein_representation_config_cache}
    model_config = {
        "cache": model_config_cache,
        "overwrite_cache": overwrite_model_config_cache,
        "model_type": model_type,
        "representation": representation,
        "complex_representation": complex_representation,
        "ligand_representation": ligand_representation,
        "protein_representation": protein_representation,
        "strategy": strategy,
        "strategy_layer_norm": strategy_layer_norm,
        "pred_readout": pred_readout,
        "combination": combination,
        "comb_readout": comb_readout,
        "max_comb_neg": max_comb_neg,
        "max_comb_scale": max_comb_scale,
        "pred_substrate": pred_substrate,
        "pred_km": pred_km,
        "comb_substrate": comb_substrate,
        "comb_km": comb_km,
        "rand_seed": model_rand_seed,
    }
    es_config = {
        "cache": es_config_cache,
        "overwrite_cache": overwrite_es_config_cache,
        "es_type": es_type,
        "patience": es_patience,
        "n_check": es_n_check,
        "divergence": es_divergence,
        "burnin": es_burnin,
    }
    if dataset_type is not None:
        is_structural = dataset_type != DatasetType.graph
    elif structures is not None:
        # Assume structural since structures were passed
        is_structural = True
    else:
        # Don't know what to do so punt and hope there's an appropriate cache file
        is_structural = None
    ds_config = {
        "cache": ds_config_cache,
        "overwrite_cache": overwrite_ds_config_cache,
        "ds_type": dataset_type,
        "structures": structures,
        "xtal_regex": xtal_regex,
        "cpd_regex": cpd_regex,
        "exp_file": exp_file,
        "is_structural": is_structural,
        "cache_file": ds_cache,
        "overwrite": overwrite_ds_cache,
        "export_input_data": export_input_data,
        "export_exp_data": export_exp_data,
        "grouped": grouped_dataset,
        "for_e3nn": e3nn_dataset,
        "random_iter": ds_random_iter,
    }

    ds_splitter_config = {
        "cache": ds_split_config_cache,
        "overwrite_cache": overwrite_ds_split_config_cache,
        "split_type": ds_split_type,
        "grouped": model_type == ModelType.grouped,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "enforce_one": enforce_one,
        "rand_seed": ds_rand_seed,
        "split_dict": ds_split_dict,
    }
    loss_configs = [kvp_list_to_dict(loss_str) for loss_str in loss]
    data_aug_configs = [kvp_list_to_dict(aug_str) for aug_str in data_aug]

    # Parse loss_dict
    if loss_dict:
        loss_dict = json.loads(loss_dict.read_text())

    # Gather all the configs
    trainer_kwargs = {
        "optimizer_config": optim_config,
        "mtenn_model_config": model_config,
        "es_config": es_config,
        "ds_config": ds_config,
        "ds_splitter_config": ds_splitter_config,
        "loss_configs": loss_configs,
        "loss_weights": loss_weights,
        "eval_loss_weights": eval_loss_weights,
        "data_aug_configs": data_aug_configs,
        "weight_decay": trainer_weight_decay,
        "batch_norm": batch_norm,
        "auto_init": auto_init,
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "target_prop": target_prop,
        "cont": cont,
        "loss_dict": loss_dict,
        "device": device,
        "output_dir": output_dir,
        "save_weights": save_weights,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_name": wandb_name,
        "extra_config": Trainer.parse_extra_config(extra_config),
        "s3_path": s3_path,
        "upload_to_s3": upload_to_s3,
        "model_tag": model_tag,
    }

    t = _build_trainer(
        trainer_kwargs, trainer_config_cache, overwrite_trainer_config_cache
    )
    t.initialize()
    t.train()


def _build_trainer(
    trainer_kwargs: dict,
    trainer_config_cache: Path = None,
    overwrite_trainer_config_cache: bool = False,
):
    """
    Helper function to build a Trainer from kwargs and (optionally) a JSON Trainer
    config file. If a config file is given, those args will be used as the default, to
    be overwritten by anything in trainer_kwargs.

    Parameters
    ----------
    trainer_kwargs : dict
        Args to be passed to the Trainer constructor. These will supersede anything in
        trainer_config_cache
    trainer_config_cache : Path, optional
        Trainer Config JSON cache file. Any other CLI args that are passed will
        supersede anything in this file
    overwrite_trainer_config_cache : bool, default=False
        Overwrite any existing Trainer JSON cache file

    Returns
    -------
    Trainer
    """

    # Filter out None Trainer kwargs
    trainer_kwargs = {
        k: v
        for k, v in trainer_kwargs.items()
        if not ((v is None) or (isinstance(v, tuple) and len(v) == 0))
    }

    def update_potentially_nested_dict(orig_dict, update_dict):
        update_dict = {
            k: v
            for k, v in update_dict.items()
            if not ((v is None) or (isinstance(v, tuple) and len(v) == 0))
        }
        for k, v in update_dict.items():
            if isinstance(v, collections.abc.Mapping):
                orig_dict[k] = update_potentially_nested_dict(orig_dict.get(k, {}), v)
            else:
                orig_dict[k] = v
        return orig_dict

    # If we got a config for the Trainer, load those args and merge with CLI args
    if trainer_config_cache and trainer_config_cache.exists():
        config_trainer_kwargs = json.loads(trainer_config_cache.read_text())

        for config_name, config_val in config_trainer_kwargs.items():
            # Arg wasn't passed at all, so got filtered out before
            if config_name not in trainer_kwargs:
                continue

            if isinstance(config_val, dict):
                update_dict = {
                    k: v
                    for k, v in trainer_kwargs[config_name].items()
                    if not ((v is None) or (isinstance(v, tuple) and len(v) == 0))
                }
                update_potentially_nested_dict(config_val, update_dict)
            elif (isinstance(config_val, list) or isinstance(config_val, tuple)) and (
                len(trainer_kwargs[config_name]) == 0
            ):
                # If no values are passed to CLI keep config values
                pass
            else:
                config_trainer_kwargs[config_name] = trainer_kwargs[config_name]

        trainer_kwargs = config_trainer_kwargs

    try:
        t = Trainer(**trainer_kwargs)
    except ValidationError as exc:
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
        trainer_config_cache.write_text(t.model_dump_json())

    return t
