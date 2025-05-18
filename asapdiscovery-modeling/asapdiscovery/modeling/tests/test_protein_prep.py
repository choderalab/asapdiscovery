import os

import pytest
from pydantic.v1 import ValidationError

from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.sequence import seqres_by_target
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.protein_prep import ProteinPrepper
from asapdiscovery.modeling.schema import PreppedComplex, PreppedTarget
from asapdiscovery.data.backend.openeye import load_openeye_design_unit
from asapdiscovery.data.schema.identifiers import TargetIdentifiers


@pytest.fixture
def loop_db():
    return fetch_test_file("fragalysis-mpro_spruce.loop_db")

@pytest.fixture(scope="session")
def complex_pdb():
    pdb = fetch_test_file("Mpro-P2660_0A_bound.pdb")
    return pdb

@pytest.fixture(scope="session")
def cmplx():
    return Complex.from_pdb(
        fetch_test_file("structure_dir/Mpro-x0354_0A_bound.pdb"),
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test2"},
    )


@pytest.fixture(scope="session")
def prep_complex():
    return Complex.from_pdb(
        fetch_test_file("SARS2_Mac1A-A1013.pdb"),
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test2"},
    )

@pytest.fixture(scope="session")
def complex_oedu():
    oedu = fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu")
    return oedu


@pytest.fixture(scope="session")
def all_structure_dir_fns():
    return [
        "structure_dir/Mpro-x0354_0A_bound.pdb",
        "structure_dir/Mpro-x1002_0A_bound.pdb",
    ]


@pytest.fixture(scope="session")
def structure_dir(all_structure_dir_fns):
    all_paths = [fetch_test_file(f) for f in all_structure_dir_fns]
    return all_paths[0].parent, all_paths

@pytest.fixture(scope="session")
def oedu_file():
    oedu = fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17_prepped_receptor_0.oedu")
    return oedu

@pytest.fixture(scope="session")
def du_cache_files():
    return ["du_cache/Mpro-x0354_0A_bound.oedu", "du_cache/Mpro-x1002_0A_bound.oedu"]


@pytest.fixture(scope="session")
def du_cache(du_cache_files):
    all_paths = [fetch_test_file(f) for f in du_cache_files]
    return all_paths[0].parent, all_paths


@pytest.fixture(scope="session")
def json_cache():
    """A mock json cache of prepared proteins"""
    return fetch_test_file("protein_json_cache/Mpro-x0354_0A_bound.json")


@pytest.mark.parametrize("use_dask", [True, False])
def test_protein_prep(prep_complex, use_dask, loop_db):
    target = TargetTags["SARS-CoV-2-Mac1"]
    prepper = ProteinPrepper(loop_db=loop_db, seqres_yaml=seqres_by_target(target))
    pcs = prepper.prep([prep_complex], use_dask=use_dask)
    assert len(pcs) == 1
    assert pcs[0].target.target_name == "test"
    assert pcs[0].ligand.compound_name == "test2"


def test_cache_load(json_cache):
    """Test loading cached PreppedComplex files."""

    cached_complexs = ProteinPrepper.load_cache(cache_dir=json_cache.parent)
    assert len(cached_complexs) == 1
    assert (
        cached_complexs[0].hash
        == "9e2ea19d1a175314647dacb9d878138a80b8443cff5faf56031bf4af61179a0a+GIIIJZOPGUFGBF-QXYFZJGFNA-O"
    )


def test_prepped_complex_from_complex(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = PreppedComplex.from_complex(c1, prep_kwargs={})
    du = c2.target.to_oedu()
    assert du.HasReceptor()
    assert du.HasLigand()
    assert c2.target.target_name == "test"
    assert c2.ligand.compound_name == "test"


def test_prepped_complex_from_oedu_file(complex_oedu):
    c = PreppedComplex.from_oedu_file(
        complex_oedu,
        target_kwargs={"target_name": "test", "target_hash": "test hash"},
        ligand_kwargs={"compound_name": "test"},
    )
    assert c.target.target_name == "test"
    assert c.ligand.compound_name == "test"


def test_prepped_complex_hash(complex_pdb):
    comp = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "receptor1"},
        ligand_kwargs={"compound_name": "ligand1"},
    )
    pc = PreppedComplex.from_complex(comp)
    assert (
        pc.target.target_hash
        == "843587eb7f589836d67da772b11584da4fa02fba63d6d3f3062e98c177306abb"
    )
    assert (
        pc.hash
        == "843587eb7f589836d67da772b11584da4fa02fba63d6d3f3062e98c177306abb+JZJCSVMJFIAMQB-DLYUOGNHNA-N"
    )


def test_preppedtarget_from_oedu_file(oedu_file):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    oedu = pt.to_oedu()
    assert oedu.GetTitle() == "(AB) > LIG(A-403)"  # from one of the old files


def test_preppedtarget_from_oedu_file_at_least_one_id(oedu_file):
    with pytest.raises(ValidationError):
        # neither id is set
        PreppedTarget.from_oedu_file(oedu_file)


def test_preppedtarget_to_pdb_file(oedu_file, tmpdir):
    """Make sure a target can be saved to pdb file for vis"""

    with tmpdir.as_cwd():
        pt = PreppedTarget.from_oedu_file(
            oedu_file, target_name="PreppedTargetTest", target_hash="mock-hash"
        )
        file_name = "test_protein.pdb"
        pt.to_pdb_file(file_name)
        assert os.path.exists(file_name) is True


def test_preppedtarget_from_oedu_file_at_least_one_target_id(oedu_file):
    with pytest.raises(ValidationError):
        _ = PreppedTarget.from_oedu_file(oedu_file, ids=TargetIdentifiers())


def test_prepped_target_from_oedu_file_bad_file():
    with pytest.raises(FileNotFoundError):
        # neither id is set
        _ = PreppedTarget.from_oedu_file(
            "bad_file", target_name="PreppedTargetTestName"
        )


def test_prepped_target_from_oedu(oedu_file):
    loaded_oedu = load_openeye_design_unit(oedu_file)
    loaded_oedu.SetTitle("PreppedTargetTestName")
    pt = PreppedTarget.from_oedu(
        loaded_oedu, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    oedu = pt.to_oedu()
    assert oedu.GetTitle() == "PreppedTargetTestName"


def test_prepped_target_from_oedu_file_roundtrip(oedu_file, tmp_path):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    pt.to_oedu_file(tmp_path / "test.oedu")
    pt2 = PreppedTarget.from_oedu_file(
        tmp_path / "test.oedu",
        target_name="PreppedTargetTestName",
        target_hash="mock-hash",
    )
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)


def test_prepped_target_from_oedu_roundtrip(oedu_file):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    du = pt.to_oedu()
    pt2 = PreppedTarget.from_oedu(
        du, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)


def test_prepped_target_json_roundtrip(oedu_file):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    js = pt.json()
    pt2 = PreppedTarget.from_json(js)
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)
    du = pt2.to_oedu()
    assert du.GetTitle() == "(AB) > LIG(A-403)"


def test_prepped_target_json_file_roundtrip(oedu_file, tmp_path):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    path = tmp_path / "test.json"
    pt.to_json_file(path)
    pt2 = PreppedTarget.from_json_file(path)
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)
    du = pt2.to_oedu()
    assert du.GetTitle() == "(AB) > LIG(A-403)"
