import itertools as it
import json
import pickle as pkl

import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.ml.cli import ml as cli
from asapdiscovery.ml.config import DatasetConfig
from asapdiscovery.ml.schema import TrainingPredictionTracker
from asapdiscovery.ml.trainer import Trainer
from click.testing import CliRunner


@pytest.fixture(scope="session")
def exp_file():
    exp_file = fetch_test_file("ml_testing/exp_file.json")
    return exp_file


@pytest.fixture(scope="session")
def docked_files():
    docked_files = fetch_test_file(
        [
            "ml_testing/docked/AAR-POS-5507155c-1_Mpro-P0018_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-RCN-390aeb1f-1_Mpro-P3074_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-RCN-67438d21-1_Mpro-P3074_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-POS-d2a4d1df-27_Mpro-P0238_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-POS-8a4e0f60-7_Mpro-P0053_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-RCN-28a8122f-1_Mpro-P2005_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-RCN-37d0aa00-1_Mpro-P3074_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-RCN-748c104b-1_Mpro-P3074_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-RCN-521d1733-1_Mpro-P2005_0A_0_bound_best.pdb",
            "ml_testing/docked/AAR-RCN-845f9611-1_Mpro-P2005_0A_0_bound_best.pdb",
        ]
    )
    return docked_files


@pytest.mark.parametrize(
    "export_input_data,export_exp_data",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_build_ds_graph(exp_file, tmp_path, export_input_data, export_exp_data):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-dataset",
            "--dataset-type",
            "graph",
            "--export-input-data",
            export_input_data,
            "--export-exp-data",
            export_exp_data,
            "--exp-file",
            exp_file,
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
        ],
    )
    assert result.exit_code == 0

    # Make sure files exist
    ds_cache = tmp_path / "ds_cache.pkl"
    ds_config_cache = tmp_path / "ds_config_cache.json"
    assert ds_cache.exists()
    assert ds_config_cache.exists()

    # Load and check stuff
    ds = pkl.loads(ds_cache.read_bytes())
    assert len(ds) == 10

    ds_config = DatasetConfig(**json.loads(ds_config_cache.read_text()))
    assert ds_config.ds_type == "graph"
    if export_input_data:
        assert len(ds_config.input_data) == 10
    else:
        assert ds_config.input_data is None
    if export_exp_data:
        assert len(ds_config.exp_data) > 0
    else:
        assert len(ds_config.exp_data) == 0
    assert ds_config.cache_file == ds_cache
    assert not ds_config.overwrite


# All 3-permutations of True and False
@pytest.mark.parametrize(
    "export_input_data,export_exp_data,random_iter", it.product([True, False], repeat=3)
)
def test_build_ds_schnet(
    exp_file, docked_files, tmp_path, export_input_data, export_exp_data, random_iter
):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-dataset",
            "--dataset-type",
            "structural",
            "--export-input-data",
            export_input_data,
            "--export-exp-data",
            export_exp_data,
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--ds-random-iter",
            random_iter,
        ],
    )
    assert result.exit_code == 0

    # Make sure files exist
    ds_cache = tmp_path / "ds_cache.pkl"
    ds_config_cache = tmp_path / "ds_config_cache.json"
    assert ds_cache.exists()
    assert ds_config_cache.exists()

    # Load and check stuff
    ds = pkl.loads(ds_cache.read_bytes())
    assert len(ds) == 10

    ds_config = DatasetConfig(**json.loads(ds_config_cache.read_text()))
    assert ds_config.ds_type == "structural"
    if export_input_data:
        assert len(ds_config.input_data) == 10
    else:
        assert ds_config.input_data is None
    if export_exp_data:
        assert len(ds_config.exp_data) > 0
    else:
        assert len(ds_config.exp_data) == 0
    assert ds_config.cache_file == ds_cache
    assert not ds_config.grouped
    assert not ds_config.for_e3nn
    assert not ds_config.overwrite
    assert ds.random_iter == random_iter


@pytest.mark.parametrize(
    "export_input_data,export_exp_data",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_build_ds_e3nn(
    exp_file, docked_files, tmp_path, export_input_data, export_exp_data
):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-dataset",
            "--dataset-type",
            "structural",
            "--e3nn-dataset",
            "True",
            "--export-input-data",
            export_input_data,
            "--export-exp-data",
            export_exp_data,
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
        ],
    )
    assert result.exit_code == 0

    # Make sure files exist
    ds_cache = tmp_path / "ds_cache.pkl"
    ds_config_cache = tmp_path / "ds_config_cache.json"
    assert ds_cache.exists()
    assert ds_config_cache.exists()

    # Load and check stuff
    ds = pkl.loads(ds_cache.read_bytes())
    assert len(ds) == 10

    ds_config = DatasetConfig(**json.loads(ds_config_cache.read_text()))
    assert ds_config.ds_type == "structural"
    if export_input_data:
        assert len(ds_config.input_data) == 10
    else:
        assert ds_config.input_data is None
    if export_exp_data:
        assert len(ds_config.exp_data) > 0
    else:
        assert len(ds_config.exp_data) == 0
    assert ds_config.cache_file == ds_cache
    assert not ds_config.grouped
    assert ds_config.for_e3nn
    assert not ds_config.overwrite


def test_build_trainer_graph(exp_file, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "ligand",
            "--representation",
            "representation_type:gat",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--export-input-data",
            "True",
            "--export-exp-data",
            "True",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    assert trainer_config_cache.exists()
    assert not output_dir.exists()
    assert not (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    t = Trainer(**json.loads(trainer_config_cache.read_text()))
    assert t.n_epochs == 1
    assert t.output_dir == output_dir
    assert not t.use_wandb
    assert not t._is_initialized
    assert t.model is None
    assert t.optimizer is None
    assert t.es is None
    assert t.ds is None
    assert t.ds_train is None
    assert t.ds_val is None
    assert t.ds_test is None
    assert t.loss_funcs is None

    assert t.ds_config.ds_type == "graph"
    assert not t.ds_config.for_e3nn


def test_build_trainer_schnet(exp_file, docked_files, tmp_path):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "model",
            "--representation",
            "representation_type:schnet",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--export-input-data",
            "True",
            "--export-exp-data",
            "True",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    assert trainer_config_cache.exists()
    assert not output_dir.exists()
    assert not (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    t = Trainer(**json.loads(trainer_config_cache.read_text()))
    assert t.n_epochs == 1
    assert t.output_dir == output_dir
    assert not t.use_wandb
    assert not t._is_initialized
    assert t.model is None
    assert t.optimizer is None
    assert t.es is None
    assert t.ds is None
    assert t.ds_train is None
    assert t.ds_val is None
    assert t.ds_test is None
    assert t.loss_funcs is None

    assert t.ds_config.ds_type == "structural"
    assert not t.ds_config.for_e3nn


def test_build_trainer_e3nn(exp_file, docked_files, tmp_path):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "model",
            "--representation",
            "representation_type:e3nn,irreps_hidden:0:5",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--export-input-data",
            "True",
            "--export-exp-data",
            "True",
            "--e3nn-dataset",
            "True",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    assert trainer_config_cache.exists()
    assert not output_dir.exists()
    assert not (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    t = Trainer(**json.loads(trainer_config_cache.read_text()))
    assert t.n_epochs == 1
    assert t.output_dir == output_dir
    assert not t.use_wandb
    assert not t._is_initialized
    assert t.model is None
    assert t.optimizer is None
    assert t.es is None
    assert t.ds is None
    assert t.ds_train is None
    assert t.ds_val is None
    assert t.ds_test is None
    assert t.loss_funcs is None

    assert t.ds_config.ds_type == "structural"
    assert t.ds_config.for_e3nn

    assert t.mtenn_model_config.representation.irreps_hidden == "5x0o+5x0e"


def test_build_and_train_graph(exp_file, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-and-train",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "ligand",
            "--representation",
            "representation_type:gat",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    tpt_path = output_dir / "pred_tracker.json"
    assert trainer_config_cache.exists()
    assert output_dir.exists()
    assert (output_dir / "init.th").exists()
    for i in range(1):
        assert (output_dir / f"{i}.th").exists()
    assert (output_dir / "final.th").exists()
    assert tpt_path.exists()
    assert (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    pred_tracker = TrainingPredictionTracker(**json.loads(tpt_path.read_text()))
    assert {"train", "test", "val"} == set(pred_tracker.split_dict.keys())
    assert len(pred_tracker.split_dict["train"]) == 8
    assert len(pred_tracker.split_dict["val"]) == 1
    assert len(pred_tracker.split_dict["test"]) == 1


def test_build_and_train_schnet(exp_file, docked_files, tmp_path):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-and-train",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "model",
            "--representation",
            "representation_type:schnet",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    tpt_path = output_dir / "pred_tracker.json"
    assert trainer_config_cache.exists()
    assert output_dir.exists()
    assert (output_dir / "init.th").exists()
    for i in range(1):
        assert (output_dir / f"{i}.th").exists()
    assert (output_dir / "final.th").exists()
    assert tpt_path.exists()
    assert (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    pred_tracker = TrainingPredictionTracker(**json.loads(tpt_path.read_text()))
    assert {"train", "test", "val"} == set(pred_tracker.split_dict.keys())
    assert len(pred_tracker.split_dict["train"]) == 8
    assert len(pred_tracker.split_dict["val"]) == 1
    assert len(pred_tracker.split_dict["test"]) == 1


def test_build_and_train_e3nn(exp_file, docked_files, tmp_path):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-and-train",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "model",
            "--representation",
            "representation_type:e3nn,irreps_hidden:0:5",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    tpt_path = output_dir / "pred_tracker.json"
    assert trainer_config_cache.exists()
    assert output_dir.exists()
    assert (output_dir / "init.th").exists()
    for i in range(1):
        assert (output_dir / f"{i}.th").exists()
    assert (output_dir / "final.th").exists()
    assert tpt_path.exists()
    assert (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    pred_tracker = TrainingPredictionTracker(**json.loads(tpt_path.read_text()))
    assert {"train", "test", "val"} == set(pred_tracker.split_dict.keys())
    assert len(pred_tracker.split_dict["train"]) == 8
    assert len(pred_tracker.split_dict["val"]) == 1
    assert len(pred_tracker.split_dict["test"]) == 1


def test_build_and_train_schnet_jitter(exp_file, docked_files, tmp_path):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-and-train",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "model",
            "--representation",
            "representation_type:schnet",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--data-aug",
            "aug_type:jitter_fixed",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    tpt_path = output_dir / "pred_tracker.json"
    assert trainer_config_cache.exists()
    assert output_dir.exists()
    assert (output_dir / "init.th").exists()
    for i in range(1):
        assert (output_dir / f"{i}.th").exists()
    assert (output_dir / "final.th").exists()
    assert tpt_path.exists()
    assert (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    pred_tracker = TrainingPredictionTracker(**json.loads(tpt_path.read_text()))
    assert {"train", "test", "val"} == set(pred_tracker.split_dict.keys())
    assert len(pred_tracker.split_dict["train"]) == 8
    assert len(pred_tracker.split_dict["val"]) == 1
    assert len(pred_tracker.split_dict["test"]) == 1


def test_build_and_train_e3nn_jitter(exp_file, docked_files, tmp_path):
    docked_dir = docked_files[0].parent

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build-and-train",
            "--output-dir",
            tmp_path / "model_out",
            "--model-type",
            "model",
            "--representation",
            "representation_type:e3nn,irreps_hidden:0:5",
            "--trainer-config-cache",
            tmp_path / "trainer.json",
            "--ds-split-type",
            "temporal",
            "--exp-file",
            exp_file,
            "--structures",
            str(docked_dir),
            "--ds-cache",
            tmp_path / "ds_cache.pkl",
            "--ds-config-cache",
            tmp_path / "ds_config_cache.json",
            "--data-aug",
            "aug_type:jitter_fixed",
            "--loss",
            "loss_type:mse_step",
            "--device",
            "cpu",
            "--n-epochs",
            "1",
            "--use-wandb",
            "False",
        ],
    )
    assert result.exit_code == 0

    # Make sure the right files exist
    trainer_config_cache = tmp_path / "trainer.json"
    output_dir = tmp_path / "model_out"
    tpt_path = output_dir / "pred_tracker.json"
    assert trainer_config_cache.exists()
    assert output_dir.exists()
    assert (output_dir / "init.th").exists()
    for i in range(1):
        assert (output_dir / f"{i}.th").exists()
    assert (output_dir / "final.th").exists()
    assert tpt_path.exists()
    assert (tmp_path / "ds_cache.pkl").exists()
    assert (tmp_path / "ds_config_cache.json").exists()

    # Load and check stuff
    pred_tracker = TrainingPredictionTracker(**json.loads(tpt_path.read_text()))
    assert {"train", "test", "val"} == set(pred_tracker.split_dict.keys())
    assert len(pred_tracker.split_dict["train"]) == 8
    assert len(pred_tracker.split_dict["val"]) == 1
    assert len(pred_tracker.split_dict["test"]) == 1
