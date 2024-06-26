import os
from pathlib import Path

import pytest
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.docking.openeye import POSITDocker


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="Docking tests slow on GHA on macOS"
)
class TestDocking:
    def test_docking(self, docking_input_pair):
        docker = POSITDocker(use_omega=False)  # save compute
        results = docker.dock([docking_input_pair])
        assert len(results) == 1
        assert results[0].probability > 0.0

    @pytest.mark.parametrize("omega_dense", [True, False])
    def test_docking_omega(self, docking_input_pair, omega_dense):
        docker = POSITDocker(use_omega=True, omega_dense=omega_dense)
        results = docker.dock([docking_input_pair])
        assert len(results) == 1
        assert results[0].probability > 0.0

    def test_docking_omega_dense_fails_no_omega(self):
        with pytest.raises(
            ValueError, match="Cannot use omega_dense without use_omega"
        ):
            _ = POSITDocker(use_omega=False, omega_dense=True)

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("return_for_disk_backend", [True, False])
    def test_docking_dask(
        self,
        docking_input_pair,
        docking_input_pair_simple,
        use_dask,
        return_for_disk_backend,
        tmp_path,
    ):
        docker = POSITDocker(use_omega=False)  # save compute
        results = docker.dock(
            [docking_input_pair, docking_input_pair_simple],
            use_dask=use_dask,
            return_for_disk_backend=return_for_disk_backend,
            output_dir=tmp_path / "docking_results",
        )
        assert len(results) == 2

    def test_docking_with_file_write(self, results_simple, tmp_path):
        docker = POSITDocker(use_omega=False)
        docker.write_docking_files(results_simple, tmp_path)

    def test_docking_with_cache(self, docking_input_pair, tmp_path, caplog):
        import logging

        caplog.set_level(logging.DEBUG)
        docker = POSITDocker(use_omega=False)
        results = docker.dock(
            [docking_input_pair], output_dir=tmp_path / "docking_results"
        )
        assert len(results) == 1
        assert results[0].probability > 0.0
        assert Path(tmp_path / "docking_results").exists()

        results2 = docker.dock(
            [docking_input_pair], output_dir=tmp_path / "docking_results"
        )
        assert len(results2) == 1
        assert results2 == results
        assert "already exists, reading from disk" in caplog.text

    def test_multireceptor_docking(self, docking_multi_structure):
        assert len(docking_multi_structure.complexes) == 2
        docker = POSITDocker(use_omega=False)  # save compute
        results = docker.dock([docking_multi_structure])
        assert len(results) == 1
        assert results[0].input_pair.complex.target.target_name == "Mpro-x0354"
        assert results[0].probability > 0.0

    def test_results_to_df(self, results_simple):
        df = results_simple[0].to_df()
        assert DockingResultCols.SMILES in df.columns
        assert DockingResultCols.LIGAND_ID in df.columns
