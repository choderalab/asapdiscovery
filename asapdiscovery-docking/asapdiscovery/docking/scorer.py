import abc
import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

import numpy as np
import pandas as pd
from asapdiscovery.data.backend.openeye import oedocking
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand, LigandIdentifiers
from asapdiscovery.data.schema.target import TargetIdentifiers
from asapdiscovery.data.util.dask_utils import (
    BackendType,
    FailureMode,
    backend_wrapper,
    dask_vmap,
)
from asapdiscovery.docking.docking import DockingResult
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from multimethod import multimethod
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class ScoreType(str, Enum):
    """
    Enum for score types.
    """

    chemgauss4 = "chemgauss4"
    FINT = "FINT"
    GAT_pIC50 = "GAT-pIC50"
    GAT_LogD = "GAT-LogD"
    schnet_pIC50 = "schnet-pIC50"
    e3nn_pIC50 = "e3nn-pIC50"
    sym_clash = "sym_clash"
    INVALID = "INVALID"


class ScoreUnits(str, Enum):
    """
    Enum for score units.
    """

    arbitrary = "arbitrary"
    kcal_mol = "kcal/mol"
    pIC50 = "pIC50"
    INVALID = "INVALID"


# TODO: this is a massive kludge, need to refactor


_SCORE_MANIFOLD_ALIAS = {
    ScoreType.chemgauss4: DockingResultCols.DOCKING_SCORE_POSIT.value,
    ScoreType.FINT: DockingResultCols.FITNESS_SCORE_FINT.value,
    ScoreType.GAT_pIC50: DockingResultCols.COMPUTED_GAT_PIC50.value,
    ScoreType.GAT_LogD: DockingResultCols.COMPUTED_GAT_LOGD.value,
    ScoreType.schnet_pIC50: DockingResultCols.COMPUTED_SCHNET_PIC50.value,
    ScoreType.e3nn_pIC50: DockingResultCols.COMPUTED_E3NN_PIC50.value,
    ScoreType.INVALID: None,
    ScoreType.sym_clash: DockingResultCols.SYMEXP_CLASH_NUM.value,
    "target_name": DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
    "compound_name": DockingResultCols.LIGAND_ID.value,
    "smiles": DockingResultCols.SMILES.value,
    "ligand_inchikey": DockingResultCols.INCHIKEY.value,
    "probability": DockingResultCols.DOCKING_CONFIDENCE_POSIT.value,
}


class Score(BaseModel):
    """
    Result of scoring, we don't embed the input result because it can be large,
    instead we just store the input result ids.
    """

    score_type: ScoreType
    score: float
    compound_name: Optional[str]
    smiles: Optional[str]
    ligand_identifiers: Optional[LigandIdentifiers]
    ligand_inchikey: Optional[str]
    target_name: Optional[str]
    target_identifiers: Optional[TargetIdentifiers]
    complex_ligand_smiles: Optional[str]
    probability: Optional[float]
    pose_id: Optional[int]
    units: ScoreUnits
    input: Optional[Any] = None

    @classmethod
    def from_score_and_docking_result(
        cls,
        score: float,
        score_type: ScoreType,
        units: ScoreUnits,
        docking_result: DockingResult,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=docking_result.posed_ligand.compound_name,
            smiles=docking_result.posed_ligand.smiles,
            ligand_inchikey=docking_result.posed_ligand.inchikey,
            ligand_ids=docking_result.posed_ligand.ids,
            target_name=docking_result.input_pair.complex.target.target_name,
            target_ids=docking_result.input_pair.complex.target.ids,
            complex_ligand_smiles=docking_result.input_pair.complex.ligand.smiles,
            probability=docking_result.probability,
            pose_id=docking_result.pose_id,
            units=units,
            input=docking_result,
        )

    @classmethod
    def from_score_and_complex(
        cls,
        score: float,
        score_type: ScoreType,
        units: ScoreUnits,
        complex: Complex,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=complex.ligand.compound_name,
            smiles=complex.ligand.smiles,
            ligand_inchikey=complex.ligand.inchikey,
            ligand_ids=complex.ligand.ids,
            target_name=complex.target.target_name,
            target_ids=complex.target.ids,
            complex_ligand_smiles=complex.ligand.smiles,
            probability=None,
            units=units,
            input=complex,
        )

    @classmethod
    def from_score_and_smiles(
        cls,
        score: float,
        smiles: str,
        score_type: ScoreType,
        units: ScoreUnits,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=None,
            smiles=smiles,
            ligand_inchikey=None,
            ligand_ids=None,
            target_name=None,
            target_ids=None,
            complex_ligand_smiles=None,
            probability=None,
            units=units,
            input=smiles,
        )

    @classmethod
    def from_score_and_ligand(
        cls,
        score: float,
        ligand: Ligand,
        score_type: ScoreType,
        units: ScoreUnits,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=ligand.compound_name,
            smiles=ligand.smiles,
            ligand_inchikey=ligand.inchikey,
            ligand_ids=ligand.ids,
            target_name=None,
            target_ids=None,
            complex_ligand_smiles=None,
            probability=None,
            units=units,
            input=ligand,
        )

    @staticmethod
    def _combine_and_pivot_scores_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """ """
        df = pd.concat(dfs)
        made_json = False
        if "input" in df.columns:
            # find the type of the input and cast to the appropriate type
            if isinstance(df["input"].iloc[0], str):
                made_json = False  # already a string
            else:  # cast to JSON
                dtype = type(df["input"].iloc[0])
                df["input"] = df["input"].apply(lambda x: x.json())
                made_json = True
        indices = set(df.columns) - {"score_type", "score", "units"}
        df = df.pivot(
            index=indices,
            columns=["score_type"],
            values="score",
        ).reset_index()

        if made_json:
            df["input"] = df["input"].apply(lambda x: dtype.from_json(x))

        df.rename(columns=_SCORE_MANIFOLD_ALIAS, inplace=True)
        return df


class ScorerBase(BaseModel):
    """
    Base class for scoring functions.
    """

    score_type: ScoreType = Field(ScoreType.INVALID, description="Type of score")
    score_units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    @abc.abstractmethod
    def _score(self) -> list[DockingResult]: ...

    def score(
        self,
        inputs: Union[
            list[DockingResult], list[Complex], list[Path], list[str], list[Ligand]
        ],
        use_dask: bool = False,
        dask_client=None,
        failure_mode=FailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        return_df: bool = False,
        pivot: bool = True,
        return_for_disk_backend: bool = False,
    ) -> list[Score]:
        """
        Score the inputs. Most of the work is done in the _score method, this method is in turn dispatched based on type to various methods.


        Parameters
        ----------
        inputs : Union[list[DockingResult], list[Complex], list[Path], list[str], list[Ligand]],
            List of inputs to score
        use_dask : bool, optional
            Whether to use dask, by default False
        dask_client : dask.distributed.Client, optional
            Dask client to use, by default None
        failure_mode : FailureMode, optional
            How to handle dask failures, by default FailureMode.SKIP
        backend : BackendType, optional
            Backend to use, by default BackendType.IN_MEMORY
        reconstruct_cls : Optional[Callable], optional
            Function to use to reconstruct the inputs, by default None
        return_df : bool, optional
            Whether to return a dataframe, by default False
        pivot : bool, optional
            Whether to pivot the dataframe, by default True
        return_for_disk_backend : bool, optional
            Whether to return paths to inputs for disk backend, by default False
        """

        outputs = self._score(
            inputs=inputs,
            use_dask=use_dask,
            dask_client=dask_client,
            failure_mode=failure_mode,
            backend=backend,
            reconstruct_cls=reconstruct_cls,
            return_for_disk_backend=return_for_disk_backend,
        )

        if return_df:
            df = self.scores_to_df(outputs)
            if pivot:
                return Score._combine_and_pivot_scores_df([df])
            else:
                return df
        else:
            return outputs

    @staticmethod
    def scores_to_df(scores: list[Score]) -> pd.DataFrame:
        """
        Convert a list of scores to a dataframe.

        Parameters
        ----------
        scores : list[Score]
            List of scores

        Returns
        -------
        pd.DataFrame
            Dataframe of scores
        """
        # gather some fields from the input
        data_list = []
        # flatten the list of scores
        scores = np.ravel(scores)
        for score in scores:
            dct = score.dict()
            dct["score_type"] = score.score_type.value  # convert to string
            # we don't want the unpacked version of the input
            dct.pop("input")
            dct["input"] = score.input
            # ok better
            data_list.append(dct)
        # convert to a dataframe
        df = pd.DataFrame(data_list)

        return df


class ChemGauss4Scorer(ScorerBase):
    """
    Scoring using ChemGauss.

    Overloaded to accept DockingResults, Complexes, or Paths to PDB files.

    """

    score_type: ScoreType = Field(ScoreType.chemgauss4, description="Type of score")
    units: ClassVar[ScoreUnits.arbitrary] = ScoreUnits.arbitrary

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(
        self, inputs, return_for_disk_backend: bool = False, **kwargs
    ) -> list[Score]:
        """
        Score the inputs, dispatching based on type.
        """
        return self._dispatch(
            inputs, return_for_disk_backend=return_for_disk_backend, **kwargs
        )

    @multimethod
    def _dispatch(
        self,
        inputs: list[DockingResult],
        return_for_disk_backend: bool = False,
        **kwargs,
    ) -> list[Score]:
        """
        Dispatch for DockingResults
        """
        results = []
        for inp in inputs:
            posed_mol = inp.posed_ligand.to_oemol()
            pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
            du = inp.input_pair.complex.target.to_oedu()
            pose_scorer.Initialize(du)
            chemgauss_score = pose_scorer.ScoreLigand(posed_mol)

            sc = Score.from_score_and_docking_result(
                chemgauss_score, self.score_type, self.units, inp
            )

            # overwrite the input with the path to the file
            if return_for_disk_backend:
                sc.input = _get_disk_path_from_docking_result(inp)

            results.append(sc)
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Complex], **kwargs) -> list[Score]:
        """
        Dispatch for Complexes
        """
        results = []
        for inp in inputs:
            posed_mol = inp.ligand.to_oemol()
            pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
            receptor = inp.target.to_oemol()
            pose_scorer.Initialize(receptor)
            chemgauss_score = pose_scorer.ScoreLigand(posed_mol)
            results.append(
                Score.from_score_and_complex(
                    chemgauss_score, self.score_type, self.units, inp
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Path], **kwargs) -> list[Score]:
        """
        Dispatch for PDB files from disk
        """
        # assuming reading PDB files from disk
        complexes = [
            Complex.from_pdb(
                p,
                ligand_kwargs={"compound_name": f"{p.stem}_ligand"},
                target_kwargs={"target_name": f"{p.stem}_target"},
            )
            for p in inputs
        ]
        return self._dispatch(complexes)


class MetaScorer(BaseModel):
    """
    Score from a combination of other scorers, the scorers must share an input type,
    """

    scorers: list[ScorerBase] = Field(..., description="Scorers to score with")

    def score(
        self,
        inputs: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        failure_mode=FailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        return_df: bool = False,
        return_for_disk_backend: bool = False,
    ) -> list[Score]:
        """
        Score the inputs using all the scorers provided in the constructor
        """
        results = []
        for scorer in self.scorers:
            vals = scorer.score(
                inputs=inputs,
                use_dask=use_dask,
                dask_client=dask_client,
                failure_mode=failure_mode,
                backend=backend,
                reconstruct_cls=reconstruct_cls,
                return_df=return_df,
                pivot=False,
                return_for_disk_backend=return_for_disk_backend,
            )
            results.append(vals)

        if return_df:
            return Score._combine_and_pivot_scores_df(results)

        return np.ravel(results).tolist()


def _get_disk_path_from_docking_result(docking_result: DockingResult) -> Path:
    if docking_result.provenance is None:
        raise ValueError("DockingResult does not have provenance")
    disk_path = docking_result.provenance.get("on_disk_location", None)
    if not disk_path:
        raise ValueError("DockingResult provenance does not have on_disk_location")
    return disk_path
