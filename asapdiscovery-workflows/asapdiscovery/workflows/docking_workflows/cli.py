import logging
from typing import Optional, Union

import click
from asapdiscovery.cli.cli_args import (
    active_site_chain,
    cache_dir,
    dask_args,
    fragalysis_dir,
    input_json,
    ligands,
    loglevel,
    md_args,
    ml_score,
    output_dir,
    overwrite,
    pdb_file,
    postera_args,
    ref_chain,
    save_to_cache,
    structure_dir,
    target,
    use_only_cache,
)
from asapdiscovery.data.operators.selectors.selector_list import StructureSelector
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.util.dask_utils import DaskType, FailureMode
from asapdiscovery.docking.openeye import POSIT_METHOD, POSIT_RELAX_MODE
from asapdiscovery.simulation.simulate import OpenMMPlatform
from asapdiscovery.workflows.docking_workflows.cross_docking import (
    CrossDockingWorkflowInputs,
    cross_docking_workflow,
)
from asapdiscovery.workflows.docking_workflows.large_scale_docking import (
    LargeScaleDockingInputs,
    large_scale_docking_workflow,
)
from asapdiscovery.workflows.docking_workflows.ligand_transfer_docking import (
    LigandTransferDockingWorkflowInputs,
    ligand_transfer_docking_workflow,
)
from asapdiscovery.workflows.docking_workflows.small_scale_docking import (
    SmallScaleDockingInputs,
    small_scale_docking_workflow,
)
from asapdiscovery.workflows.docking_workflows.symexp_crystal_packing import (
    SymExpCrystalPackingInputs,
    symexp_crystal_packing_workflow,
)


@click.group()
def docking():
    """Run docking and scoring workflows for protein-ligand complexes."""
    pass


@docking.command()
@target
@click.option(
    "--n-select",
    type=int,
    default=5,
    help="The number of targets to dock each ligand against, sorted by MCS",
)
@click.option(
    "--top-n",
    type=int,
    default=500,
    help="The maximum number of docking results to return, ordered by docking score",
)
@click.option(
    "--posit-confidence-cutoff",
    type=float,
    default=0.7,
    help="The confidence cutoff for POSIT results to be considered",
)
@click.option(
    "--use-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@click.option(
    "--allow-posit-retries",
    is_flag=True,
    default=False,
    help="Whether to allow POSIT to retry with relaxed parameters if docking fails (slower, more likely to succeed)",
)
@ligands
@postera_args
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@dask_args
@output_dir
@overwrite
@input_json
@ml_score
@loglevel
@ref_chain
@active_site_chain
def large_scale(
    target: TargetTags,
    n_select: int = 10,
    top_n: int = 500,
    posit_confidence_cutoff: float = 0.5,
    use_omega: bool = False,
    allow_posit_retries: bool = False,
    ligands: Optional[str] = None,
    postera: bool = False,
    postera_molset_name: Optional[str] = None,
    postera_upload: bool = False,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    dask_n_workers: Optional[int] = None,
    failure_mode: FailureMode = FailureMode.SKIP,
    ml_score: Optional[bool] = True,
    loglevel: Union[int, str] = logging.INFO,
    ref_chain: Optional[str] = None,
    active_site_chain: Optional[str] = None,
):
    """
    Run large scale docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = LargeScaleDockingInputs.from_json_file(input_json)

    else:
        inputs = LargeScaleDockingInputs(
            postera=postera,
            postera_upload=postera_upload,
            target=target,
            n_select=n_select,
            top_n=top_n,
            use_dask=use_dask,
            dask_type=dask_type,
            dask_n_workers=dask_n_workers,
            failure_mode=failure_mode,
            posit_confidence_cutoff=posit_confidence_cutoff,
            use_omega=use_omega,
            allow_posit_retries=allow_posit_retries,
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            postera_molset_name=postera_molset_name,
            cache_dir=cache_dir,
            save_to_cache=save_to_cache,
            ml_score=ml_score,
            output_dir=output_dir,
            overwrite=overwrite,
            loglevel=loglevel,
            ref_chain=ref_chain,
            active_site_chain=active_site_chain,
        )

    large_scale_docking_workflow(inputs)


@docking.command()
@target
@click.option(
    "--use-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@click.option(
    "--omega-dense",
    is_flag=True,
    default=False,
    help="Whether to use dense conformer enumeration with OEOmega (slower, more accurate)",
)
@click.option(
    "--posit-method",
    type=click.Choice(POSIT_METHOD.get_names(), case_sensitive=False),
    default="all",
    help="The set of methods POSIT can use. Defaults to all.",
)
@click.option(
    "--relax-mode",
    type=click.Choice(POSIT_RELAX_MODE.get_names(), case_sensitive=False),
    default="none",
    help="When to check for relaxation either, 'clash', 'all', 'none'",
)
@click.option(
    "--allow-retries",
    is_flag=True,
    default=False,
    help="Whether to allow POSIT to retry with relaxed parameters if docking fails (slower, more likely to succeed)",
)
@click.option(
    "--allow-final-clash",
    is_flag=True,
    default=False,
    help="Allow clashing poses in last stage of docking",
)
@click.option(
    "--multi-reference",
    is_flag=True,
    default=False,
    help="Whether to pass multiple references to the docker for each ligand instead of just one at a time",
)
@click.option(
    "--structure-selector",
    type=click.Choice(StructureSelector.get_values(), case_sensitive=False),
    default=StructureSelector.LEAVE_SIMILAR_OUT,
    help="The type of structure selector to use.",
)
@click.option("--num-poses", type=int, default=1, help="Number of poses to generate")
@ligands
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@use_only_cache
@dask_args
@output_dir
@overwrite
@input_json
@loglevel
def cross_docking(
    target: TargetTags,
    multi_reference: bool = False,
    structure_selector: StructureSelector = StructureSelector.LEAVE_SIMILAR_OUT,
    use_omega: bool = False,
    omega_dense: bool = False,
    posit_method: Optional[str] = POSIT_METHOD.ALL.name,
    relax_mode: Optional[str] = POSIT_RELAX_MODE.NONE.name,
    num_poses: int = 1,
    allow_retries: bool = False,
    allow_final_clash: bool = False,
    ligands: Optional[str] = None,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    use_only_cache: bool = False,
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    dask_n_workers: Optional[int] = None,
    failure_mode: FailureMode = FailureMode.SKIP,
    loglevel: Union[int, str] = logging.INFO,
):
    """
    Run cross docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = CrossDockingWorkflowInputs.from_json_file(input_json)

    else:
        inputs = CrossDockingWorkflowInputs(
            target=target,
            multi_reference=multi_reference,
            structure_selector=structure_selector,
            use_dask=use_dask,
            dask_type=dask_type,
            dask_n_workers=dask_n_workers,
            failure_mode=failure_mode,
            use_omega=use_omega,
            omega_dense=omega_dense,
            posit_method=POSIT_METHOD[posit_method],
            relax_mode=POSIT_RELAX_MODE[relax_mode],
            num_poses=num_poses,
            allow_retries=allow_retries,
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            cache_dir=cache_dir,
            use_only_cache=use_only_cache,
            save_to_cache=save_to_cache,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_final_clash=allow_final_clash,
            loglevel=loglevel,
        )

    cross_docking_workflow(inputs)


@docking.command()
@target
@click.option(
    "--posit-confidence-cutoff",
    type=float,
    default=0.1,
    help="The confidence cutoff for POSIT results to be considered",
)
@click.option(
    "--n-select",
    type=int,
    default=20,
    help="The number of targets to dock each ligand against, sorted by MCS",
)
@click.option("--allow-dask-cuda/--no-allow-dask-cuda", default=True)
@click.option(
    "--no-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@ligands
@postera_args
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@dask_args
@output_dir
@overwrite
@input_json
@ml_score
@md_args
@loglevel
@ref_chain
@active_site_chain
def small_scale(
    target: TargetTags,
    posit_confidence_cutoff: float = 0.1,
    n_select: int = 20,
    allow_dask_cuda: bool = True,
    no_omega: bool = False,
    ligands: Optional[str] = None,
    postera: bool = False,
    postera_molset_name: Optional[str] = None,
    postera_upload: bool = False,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    dask_n_workers: Optional[int] = None,
    failure_mode: FailureMode = FailureMode.SKIP,
    ml_score: Optional[bool] = True,
    md: bool = False,
    md_steps: int = 2500000,  # 10 ns @ 4.0 fs timestep
    md_openmm_platform: OpenMMPlatform = OpenMMPlatform.Fastest,
    loglevel: Union[int, str] = logging.INFO,
    ref_chain: Optional[str] = None,
    active_site_chain: Optional[str] = None,
):
    """
    Run small scale docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = SmallScaleDockingInputs.from_json_file(input_json)

    else:
        inputs = SmallScaleDockingInputs(
            postera=postera,
            postera_upload=postera_upload,
            target=target,
            use_dask=use_dask,
            dask_type=dask_type,
            dask_n_workers=dask_n_workers,
            failure_mode=failure_mode,
            posit_confidence_cutoff=posit_confidence_cutoff,
            n_select=n_select,
            allow_dask_cuda=allow_dask_cuda,
            use_omega=not no_omega,
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            postera_molset_name=postera_molset_name,
            cache_dir=cache_dir,
            save_to_cache=save_to_cache,
            ml_score=ml_score,
            output_dir=output_dir,
            overwrite=overwrite,
            md=md,
            md_steps=md_steps,
            md_openmm_platform=md_openmm_platform,
            loglevel=loglevel,
            ref_chain=ref_chain,
            active_site_chain=active_site_chain,
        )

    small_scale_docking_workflow(inputs)


@docking.command()
@target
@click.option(
    "--vdw-radii-fudgefactor",
    type=float,
    default=0.9,
    help="The fudge factor multiplier to apply to VDW radii",
)
@click.option(
    "--symexp-clash-thresh",
    type=int,
    default=0,
    help="The number of clashes to consider a ligand as clashing",
)
@ligands
@postera_args
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@dask_args
@output_dir
@overwrite
@input_json
@loglevel
def symexp_crystal_packing(
    target: TargetTags,
    vdw_radii_fudgefactor: float = 0.9,
    symexp_clash_thresh: int = 0,
    ligands: Optional[str] = None,
    postera: bool = False,
    postera_molset_name: Optional[str] = None,
    postera_upload: bool = False,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    dask_type: DaskType = DaskType.LOCAL,
    dask_n_workers: Optional[int] = None,
    failure_mode: FailureMode = FailureMode.SKIP,
    loglevel: Union[int, str] = logging.INFO,
):
    """
    Check for overlaps between docked ligands and crystal neighbouts
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = SymExpCrystalPackingInputs.from_json_file(input_json)

    else:
        inputs = SymExpCrystalPackingInputs(
            postera=postera,
            postera_upload=postera_upload,
            target=target,
            vdw_radii_fudgefactor=vdw_radii_fudgefactor,
            use_dask=use_dask,
            dask_type=dask_type,
            dask_n_workers=dask_n_workers,
            failure_mode=failure_mode,
            ligands=ligands,
            pdb_file=pdb_file,
            fragalysis_dir=fragalysis_dir,
            structure_dir=structure_dir,
            postera_molset_name=postera_molset_name,
            cache_dir=cache_dir,
            save_to_cache=save_to_cache,
            ml_score=ml_score,
            output_dir=output_dir,
            overwrite=overwrite,
            loglevel=loglevel,
            symexp_clash_thresh=symexp_clash_thresh,
        )

    symexp_crystal_packing_workflow(inputs)


@docking.command()
@click.option(
    "--use-omega",
    is_flag=True,
    default=False,
    help="Whether to use OEOmega conformer enumeration before docking (slower, more accurate)",
)
@click.option(
    "--omega-dense",
    is_flag=True,
    default=False,
    help="Whether to use dense conformer enumeration with OEOmega (slower, more accurate)",
)
@click.option(
    "--allow-retries",
    is_flag=True,
    default=False,
    help="Whether to allow POSIT to retry with relaxed parameters if docking fails (slower, more likely to succeed)",
)
@click.option(
    "--allow-final-clash",
    is_flag=True,
    default=False,
    help="Allow clashing poses in last stage of docking",
)
@click.option("--num-poses", type=int, default=1, help="Number of poses to generate")
@click.option(
    "--ref-pdb-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a pdb file containing a reference structure",
)
@click.option(
    "--ref-fragalysis-dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to a fragalysis-formatted directory to use as reference structures.",
)
@click.option(
    "--ref-structure-dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, dir_okay=True),
    help="Path to a directory containing apo structures to dock to.",
)
@click.option(
    "--posit-confidence-cutoff",
    type=float,
    default=0.1,
    help="The confidence cutoff for POSIT results to be considered",
)
@pdb_file
@fragalysis_dir
@structure_dir
@save_to_cache
@cache_dir
@use_only_cache
@dask_args
@output_dir
@overwrite
@input_json
@ml_score
@md_args
@loglevel
@ref_chain
@active_site_chain
@click.option(
    "--seqres-yaml",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a seqres yaml file to mutate to, if not specified will use the default for the target",
)
@click.option(
    "--loop-db",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="Path to a loop database to use for prepping",
)
@target
@click.option("--allow-dask-cuda/--no-allow-dask-cuda", default=True)
def ligand_transfer_docking(
    ref_structure_dir: Optional[str] = None,
    ref_pdb_file: Optional[str] = None,
    ref_fragalysis_dir: Optional[str] = None,
    pdb_file: Optional[str] = None,
    fragalysis_dir: Optional[str] = None,
    structure_dir: Optional[str] = None,
    posit_confidence_cutoff: float = 0.1,
    use_omega: bool = False,
    omega_dense: bool = False,
    num_poses: int = 1,
    allow_retries: bool = False,
    allow_final_clash: bool = False,
    use_only_cache: bool = False,
    save_to_cache: Optional[bool] = True,
    cache_dir: Optional[str] = None,
    output_dir: str = "output",
    overwrite: bool = True,
    input_json: Optional[str] = None,
    use_dask: bool = False,
    allow_dask_cuda: bool = True,
    dask_type: DaskType = DaskType.LOCAL,
    dask_n_workers: Optional[int] = None,
    failure_mode: FailureMode = FailureMode.SKIP,
    loglevel: Union[int, str] = logging.INFO,
    ml_score: Optional[bool] = True,
    md: bool = False,
    md_steps: int = 2500000,  # 10 ns @ 4.0 fs timestep
    md_openmm_platform: OpenMMPlatform = OpenMMPlatform.Fastest,
    ref_chain: Optional[str] = "A",
    active_site_chain: Optional[str] = "A",
    seqres_yaml: Optional[str] = None,
    loop_db: Optional[str] = None,
    target=target,
):
    """
    Run ligand transfer docking on a set of ligands, against a set of targets.
    """

    if input_json is not None:
        print("Loading inputs from json file... Will override all other inputs.")
        inputs = LigandTransferDockingWorkflowInputs.from_json_file(input_json)

    else:
        inputs = LigandTransferDockingWorkflowInputs(
            reference_complex_dir=ref_structure_dir,
            reference_pdb_file=ref_pdb_file,
            reference_fragalysis_dir=ref_fragalysis_dir,
            target_pdb_file=pdb_file,
            target_fragalysis_dir=fragalysis_dir,
            target_structure_dir=structure_dir,
            use_dask=use_dask,
            allow_dask_cuda=allow_dask_cuda,
            dask_type=dask_type,
            dask_n_workers=dask_n_workers,
            failure_mode=failure_mode,
            posit_confidence_cutoff=posit_confidence_cutoff,
            use_omega=use_omega,
            omega_dense=omega_dense,
            num_poses=num_poses,
            allow_retries=allow_retries,
            cache_dir=cache_dir,
            use_only_cache=use_only_cache,
            save_to_cache=save_to_cache,
            output_dir=output_dir,
            overwrite=overwrite,
            allow_final_clash=allow_final_clash,
            loglevel=loglevel,
            ml_score=ml_score,
            md=md,
            md_steps=md_steps,
            md_openmm_platform=md_openmm_platform,
            ref_chain=ref_chain,
            active_site_chain=active_site_chain,
            seqres_yaml=seqres_yaml,
            loop_db=loop_db,
            target=target,
        )

    ligand_transfer_docking_workflow(inputs)


if __name__ == "__main__":
    docking()
