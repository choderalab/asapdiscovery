import pytest
from asapdiscovery.docking.selectors.mcs_selector import (
    MCSSelector,
    RascalMCESSelector,
)
from asapdiscovery.docking.selectors.pairwise_selector import (
    LeaveOneOutSelector,
    LeaveSimilarOutSelector,
    PairwiseSelector,
    SelfDockingSelector,
)
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.docking.docking import DockingInputPair  # TODO: move to data
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.services.cdd.cdd_api import CDDAPI
from asapdiscovery.data.services.services_config import CDDSettings
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.schema import PreppedComplex

@pytest.fixture(scope="session")
def all_mpro_fns():
    return [
        "aligned/Mpro-x11041_0A/Mpro-x11041_0A_bound.pdb",
        "aligned/Mpro-x1425_0A/Mpro-x1425_0A_bound.pdb",
        "aligned/Mpro-x11894_0A/Mpro-x11894_0A_bound.pdb",
        "aligned/Mpro-x1002_0A/Mpro-x1002_0A_bound.pdb",
        "aligned/Mpro-x10155_0A/Mpro-x10155_0A_bound.pdb",
        "aligned/Mpro-x0354_0A/Mpro-x0354_0A_bound.pdb",
        "aligned/Mpro-x11271_0A/Mpro-x11271_0A_bound.pdb",
        "aligned/Mpro-x1101_1A/Mpro-x1101_1A_bound.pdb",
        "aligned/Mpro-x1187_0A/Mpro-x1187_0A_bound.pdb",
        "aligned/Mpro-x10338_0A/Mpro-x10338_0A_bound.pdb",
    ]


@pytest.fixture(scope="session")
def complexes(all_mpro_fns):
    all_pdbs = [fetch_test_file(f"frag_factory_test/{fn}") for fn in all_mpro_fns]
    return [
        Complex.from_pdb(
            struct,
            target_kwargs={"target_name": "test"},
            ligand_kwargs={"compound_name": "test"},
        )
        for struct in all_pdbs
    ]

@pytest.fixture(scope="session")
def prepped_complexes(complexes):
    # kinda expensive to make, so let's just do the first 2
    return [PreppedComplex.from_complex(c) for c in complexes[:2]]


@pytest.fixture(scope="module")
def smiles():
    # smiles for the ligands in the first 4  test pdb files
    return [
        "Cc1ccncc1N(C)C(=O)Cc2cccc(c2)Cl",
        "CC(=O)N1CCN(CC1)c2ccc(cc2)OC",
        "c1cc(sc1)C(=O)NC(Cc2ccc(s2)N3CCOCC3)C=O",
        "c1cc[nH]c(=O)c1",
    ]


@pytest.fixture(scope="session")
def moonshot_sdf():
    sdf = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf")
    return sdf


@pytest.fixture
def sdf_file():
    return fetch_test_file("Mpro_combined_labeled.sdf")


@pytest.fixture(scope="module")
def ligands(smiles):
    return [Ligand.from_smiles(s, compound_name="test") for s in smiles]


@pytest.fixture(scope="module")
def ligands_from_complexes(complexes):
    # get ligands from 3d structure to ensure the added hydrogens make sense, using top 4 to match the smiles
    return [c.ligand for c in complexes[0:4]]


@pytest.fixture()
def mocked_cdd_api():
    """A cdd_api configured with dummy data which should have the requests mocked."""
    settings = CDDSettings(CDD_API_KEY="my-key", CDD_VAULT_NUMBER=1)
    return CDDAPI.from_settings(settings=settings)


@pytest.fixture(scope="module")
def multipose_ligand():
    return fetch_test_file("multiconf.sdf")


def test_pairwise_selector(ligands_from_complexes, complexes):
    selector = PairwiseSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 40


def test_leave_one_out_selector(ligands_from_complexes, complexes):
    selector = LeaveOneOutSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 36


def test_leave_similar_out_selector(ligands_from_complexes, complexes):
    selector = LeaveSimilarOutSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 36


def test_self_docking_selector(ligands_from_complexes, complexes):
    selector = SelfDockingSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 4


@pytest.mark.parametrize("use_dask", [True, False])
def test_pairwise_selector_prepped(ligands_from_complexes, prepped_complexes, use_dask):
    selector = PairwiseSelector()
    pairs = selector.select(
        ligands_from_complexes, prepped_complexes, use_dask=use_dask
    )
    assert len(pairs) == 8


@pytest.mark.parametrize("approximate", [True, False])
@pytest.mark.parametrize("structure_based", [True, False])
def test_mcs_selector(ligands_from_complexes, complexes, approximate, structure_based):
    selector = MCSSelector(approximate=approximate, structure_based=structure_based)
    pairs = selector.select(ligands_from_complexes, complexes, n_select=1)
    # should be 4 pairs
    assert len(pairs) == 4
    # as we matched against the exact smiles of the first 4 complex ligands_from_complexes, they should be in order
    assert pairs[0] == CompoundStructurePair(
        ligand=ligands_from_complexes[0], complex=complexes[0]
    )
    assert pairs[1] == CompoundStructurePair(
        ligand=ligands_from_complexes[1], complex=complexes[1]
    )
    assert pairs[2] == CompoundStructurePair(
        ligand=ligands_from_complexes[2], complex=complexes[2]
    )
    assert pairs[3] == CompoundStructurePair(
        ligand=ligands_from_complexes[3], complex=complexes[3]
    )


@pytest.mark.parametrize("use_dask", [True, False])
def test_rascalMCES_selector(ligands_from_complexes, complexes, use_dask):
    selector = RascalMCESSelector()
    pairs = selector.select(
        ligands_from_complexes, complexes, n_select=1, use_dask=use_dask
    )
    # should be 4 pairs
    assert len(pairs) == 4
    # as we matched against the exact smiles of the first 4 complex ligands_from_complexes, they should be in order
    assert pairs[0] == CompoundStructurePair(
        ligand=ligands_from_complexes[0], complex=complexes[0]
    )
    assert pairs[1] == CompoundStructurePair(
        ligand=ligands_from_complexes[1], complex=complexes[1]
    )
    assert pairs[2] == CompoundStructurePair(
        ligand=ligands_from_complexes[2], complex=complexes[2]
    )
    assert pairs[3] == CompoundStructurePair(
        ligand=ligands_from_complexes[3], complex=complexes[3]
    )


def test_mcs_select_prepped(ligands_from_complexes, prepped_complexes):
    selector = MCSSelector()
    pairs = selector.select(ligands_from_complexes, prepped_complexes, n_select=1)
    # should be 4 pairs
    assert len(pairs) == 4
    assert pairs[0] == DockingInputPair(
        ligand=ligands_from_complexes[0], complex=prepped_complexes[0]
    )
    assert pairs[1] == DockingInputPair(
        ligand=ligands_from_complexes[1], complex=prepped_complexes[1]
    )
    assert pairs[2] == DockingInputPair(
        ligand=ligands_from_complexes[2], complex=prepped_complexes[1]
    )
    assert pairs[3] == DockingInputPair(
        ligand=ligands_from_complexes[3], complex=prepped_complexes[0]
    )


def test_mcs_selector_nselect(ligands_from_complexes, complexes):
    selector = MCSSelector()
    pairs = selector.select(ligands_from_complexes, complexes, n_select=2)
    # should be 8 pairs
    assert len(pairs) == 8
    assert (
        pairs[0].complex.ligand.smiles == "Cc1ccncc1N(C)C(=O)Cc2cccc(c2)Cl"
    )  # exact match
    assert (
        pairs[1].complex.ligand.smiles == "Cc1ccncc1NC(=O)Cc2cc(cc(c2)Cl)OC"
    )  # clearly related


def test_mcs_selector_no_match(prepped_complexes):
    lig = Ligand.from_smiles("Si", compound_name="test_no_match")
    selector = MCSSelector()
    _ = selector.select([lig], prepped_complexes, n_select=1)
