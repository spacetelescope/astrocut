import re
from copy import deepcopy

import asdf
import asdf.schema as asdf_schema
import numpy as np
import pytest

from astrocut.asdf_spectral_subset import _make_pickle_safe
from astrocut.exceptions import DataWarning, InvalidInputError, InvalidQueryError
from astrocut.roman_spectral_subset import RomanSpectralSubset


@pytest.fixture()
def spectral_files(tmp_path):
    # Create a temporary ASDF file with spectral data for testing
    with asdf.AsdfFile() as af:
        af["roman"] = {"meta": {}, "data": {}}

        # Add 100 sources to the ASDF file with dummy data
        for i in range(420007, 420007 + 100):
            af["roman"]["data"][str(i)] = {
                "flag": np.array([True, False, True, False, True, False]),
                "wl": np.array([500, 600, 700, 800, 900, 1000]),
                "flux": np.array([10, 20, 30, 40, 50, 60]),
                "flux_error": np.array([1, 2, 3, 4, 5, 6]),
                "number_coadds": 2,
            }

        # Add some metadata
        af["roman"]["meta"] = {"unit_flux": "W m**(-2) nm**(-1)", "unit_wl": "nm"}

        # Write the ASDF file to TWO temporary locations
        temp_file = tmp_path / "temp_spectral.asdf"
        af.write_to(temp_file)

        # Add slightly more data to a second file
        af_copy = asdf.AsdfFile(deepcopy(af.tree))
        af_copy["roman"]["data"][430000] = {
            "flag": np.array([False, True, False, True]),
            "wl": np.array([500, 600, 700, 800]),
            "flux": np.array([15, 25, 35, 45]),
            "flux_error": np.array([1.5, 2.5, 3.5, 4.5]),
            "number_coadds": 3,
        }
        temp_file2 = tmp_path / "temp_spectral_copy.asdf"
        af_copy.write_to(temp_file2)

        yield [str(temp_file), str(temp_file2)]


@pytest.fixture
def subset(spectral_files):
    # Create a RomanSpectralSubset instance for testing
    return RomanSpectralSubset(
        spectral_files=spectral_files,
        source_ids=[420007, 420008],
        lite=True,
        max_workers=1,
    )


def test_make_pickle_safe():
    # Test that _make_pickle_safe correctly converts non-pickle-safe data types to pickle-safe ones
    data = {
        "int": 1,
        "float": 1.0,
        "str": "test",
        "bool": True,
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "ndarray": np.array([7, 8, 9]),
        "generic": np.int64(10),
    }
    safe_data = _make_pickle_safe(data)

    assert safe_data["int"] == 1
    assert safe_data["float"] == 1.0
    assert safe_data["str"] == "test"
    assert safe_data["bool"] is True
    assert safe_data["list"] == [1, 2, 3]
    assert safe_data["tuple"] == (4, 5, 6)
    assert np.array_equal(safe_data["ndarray"], np.asarray([7, 8, 9]))
    assert safe_data["generic"] == 10


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_data(spectral_files, lite):
    # Data extraction with no wavelength range filtering
    subset = RomanSpectralSubset(
        spectral_files=spectral_files,  # Test with multiple files (same file for simplicity)
        source_ids=[420007, 420008],
        lite=lite,
        max_workers=1,
    )

    subset_data = subset.subset_data
    assert len(subset_data) == 2  # Should have subset data for both files

    subset_data_file = subset_data[spectral_files[0]]
    assert "420007" in subset_data_file
    assert "420008" in subset_data_file
    assert len(subset_data_file.keys()) == 2  # Should only contain the specified source IDs
    assert "flux" in subset_data_file["420007"]
    assert "wl" in subset_data_file["420007"]
    # In lite mode, the 'flag' key should be removed from the subset data
    if lite:
        assert "flag" not in subset_data_file["420007"]
        assert "number_coadds" not in subset_data_file["420007"]
    else:
        print(subset_data_file["420007"])
        assert "flag" in subset_data_file["420007"]
        assert "number_coadds" in subset_data_file["420007"]

    subset_wl = subset_data_file["420007"]["wl"]
    subset_flux = subset_data_file["420007"]["flux"]
    expected_wl = np.array([500, 600, 700, 800, 900, 1000])
    expected_flux = np.array([10, 20, 30, 40, 50, 60])
    assert np.array_equal(subset_wl, expected_wl)
    assert np.array_equal(subset_flux, expected_flux)

    if not lite:
        subset_flag = subset_data_file["420007"]["flag"]
        expected_flag = np.array([True, False, True, False, True, False])
        assert np.array_equal(subset_flag, expected_flag)

        subset_number_coadds = subset_data_file["420007"]["number_coadds"]
        assert subset_number_coadds == 2


def test_roman_spectral_subset_error(spectral_files):
    # Warn if wavelength region does not overlap with any data in the file
    with pytest.warns(DataWarning, match=r"Wavelength range \(810, 910\) is out of bounds for source ID 430000"):
        RomanSpectralSubset(
            spectral_files=spectral_files[1],
            source_ids=[420007, 420008, 430000],
            wl_range=(810, 910),
            lite=True,
            max_workers=1,
            verbose=True,
        )

    # Warn if source ID specified in get_asdf_subsets is not in one of the files but is in another file
    with pytest.warns(
        DataWarning,
        match=r"Source ID 430000 not found in file .*temp_spectral\.asdf\. " r"Skipping this source for this file\.",
    ):
        RomanSpectralSubset(
            spectral_files=spectral_files,
            source_ids=[420007, 420008, 430000],
            wl_range=(550, 850),
            lite=True,
            max_workers=1,
            verbose=True,
        )

    # Error if no subsets are created
    with pytest.raises(InvalidQueryError, match=r"No subsets were created."):
        RomanSpectralSubset(
            spectral_files=spectral_files,
            source_ids=[123456],  # Source ID that doesn't exist in either file
            wl_range=(810, 910),
            lite=True,
            max_workers=1,
            verbose=False,
        )


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_data_wavelength_filter(spectral_files, lite):
    # Test with wavelength range
    subset = RomanSpectralSubset(
        spectral_files=spectral_files,  # Test with multiple files (same file for simplicity)
        source_ids=[420007, 420008],
        wl_range=(550, 850),
        lite=lite,
        max_workers=1,
    )

    subset_data_file = subset.subset_data[spectral_files[0]]
    assert "flux" in subset_data_file["420007"]
    assert "wl" in subset_data_file["420007"]
    subset_wl = subset_data_file["420007"]["wl"]
    subset_flux = subset_data_file["420007"]["flux"]

    # Check that the wavelength range is correctly applied
    assert np.all((subset_wl >= 550) & (subset_wl <= 850))
    # Check that the flux values correspond to the correct wavelength range
    expected_flux = np.array([20, 30, 40])  # Corresponding to wl = 600, 700, 800
    assert np.array_equal(subset_flux, expected_flux)

    if not lite:
        subset_flag = subset_data_file["420007"]["flag"]
        expected_flag = np.array([False, True, False])  # Corresponding to wl = 600, 700, 800
        assert np.array_equal(subset_flag, expected_flag)

        subset_number_coadds = subset_data_file["420007"]["number_coadds"]
        assert subset_number_coadds == 2


def test_roman_spectral_subset_parallel(subset, spectral_files):
    # Test that parallel processing runs without error and produces the same results as non-parallel processing
    subset_parallel = RomanSpectralSubset(
        spectral_files=spectral_files,
        source_ids=[420007],
        lite=True,
        max_workers=None,  # Use parallel processing with default number of workers
    )

    parallel_flux = subset_parallel.subset_data[spectral_files[0]]["420007"]["flux"]
    non_parallel_flux = subset.subset_data[spectral_files[0]]["420007"]["flux"]
    assert np.array_equal(parallel_flux, non_parallel_flux)


def test_roman_spectral_subset_source_file_keys(spectral_files):
    # Test the get_source_file_keys method of RomanSpectralsubset
    subset = RomanSpectralSubset(
        spectral_files=spectral_files,
        source_ids=[420007, 420008, 430000],
        wl_range=(550, 850),
        lite=True,
        max_workers=1,
    )

    source_file_keys = subset.get_source_file_keys()
    assert len(source_file_keys) == 5  # Should have one key for each combination of source ID and file

    for subset_file, subset_source_id in source_file_keys.values():
        assert subset_file in spectral_files
        assert subset_source_id in ["420007", "420008", "430000"]

    print(subset._out_trees[spectral_files[1]])

    # Filter by source IDs
    source_file_keys = subset.get_source_file_keys(source_ids=["420007", "430000"])
    assert len(source_file_keys) == 3  # Should only have keys for the specified source IDs
    for subset_file, subset_source_id in source_file_keys.values():
        assert subset_source_id == "420007" or subset_source_id == "430000"

    # Filter by source files
    source_file_keys = subset.get_source_file_keys(spectral_files=[spectral_files[0]])
    assert len(source_file_keys) == 2  # Should only have keys for the specified file
    for subset_file, subset_source_id in source_file_keys.values():
        assert subset_file == spectral_files[0]
        assert subset_source_id in ["420007", "420008"]

    # Error if a file is not found in results
    with pytest.raises(
        InvalidQueryError, match=r"Spectral file .*temp_spectral_nonexistent\.asdf " r"not found in subset results\."
    ):
        subset.get_source_file_keys(
            spectral_files=[spectral_files[0], spectral_files[1], "temp_spectral_nonexistent.asdf"]
        )

    # Error if a source ID is not found in results
    with pytest.raises(InvalidQueryError, match=r"Source ID 999999 not found in subset results\."):
        subset.get_source_file_keys(source_ids=["420007", "999999"])


def test_roman_spectral_subset_source_file_keys_duplicate_stems(tmp_path):
    def _write_spectral_file(path):
        with asdf.AsdfFile() as af:
            af["roman"] = {
                "meta": {
                    "unit_flux": "W m**(-2) nm**(-1)",
                    "unit_wl": "nm",
                },
                "data": {
                    "402849": {
                        "wl": np.array([500, 600, 700]),
                        "flux": np.array([1, 2, 3]),
                        "flux_error": np.array([0.1, 0.2, 0.3]),
                    }
                },
            }
            af.write_to(path)

    prism_dir_1 = tmp_path / "obs_a"
    prism_dir_1.mkdir()
    prism_file_1 = prism_dir_1 / "spectrum.asdf"
    _write_spectral_file(prism_file_1)

    prism_dir_2 = tmp_path / "obs_b"
    prism_dir_2.mkdir()
    prism_file_2 = prism_dir_2 / "spectrum.asdf"
    _write_spectral_file(prism_file_2)

    subset = RomanSpectralSubset(
        spectral_files=[prism_file_1, prism_file_2],
        source_ids=[402849],
        wl_range=None,
        lite=True,
        max_workers=1,
    )

    source_file_keys = subset.get_source_file_keys()

    assert len(source_file_keys) == 2
    assert set(source_file_keys.values()) == {
        (str(prism_file_1), "402849"),
        (str(prism_file_2), "402849"),
    }

    keys = list(source_file_keys.keys())
    assert all(re.match(r"^402849_spectrum_[0-9a-f]{8}$", key) for key in keys)
    assert len(set(keys)) == 2

    asdf_subsets = subset.get_asdf_subsets(group_by="source_file")
    assert set(asdf_subsets.keys()) == set(source_file_keys.keys())


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_asdf_subsets_source_file(spectral_files, lite):
    # Test the get_asdf_subsets method of RomanSpectralsubset with group_by='source_file'
    subset = RomanSpectralSubset(spectral_files=spectral_files, source_ids=[420007, 420008], lite=lite, max_workers=1)

    asdf_subsets = subset.get_asdf_subsets(group_by="source_file")
    source_file_keys = subset.get_source_file_keys()
    # Should make a subset for every combination of source ID and file, so 2 sources x 2 files = 4 subsets
    assert len(asdf_subsets) == 4
    assert len(source_file_keys) == 4

    for key, subset_af in asdf_subsets.items():
        assert key in source_file_keys
        subset_file, subset_source_id = source_file_keys[key]
        assert subset_file in spectral_files
        assert subset_source_id in ["420007", "420008"]
        assert isinstance(subset_af, asdf.AsdfFile)
        # Check that the subset ASDF file contains the expected data structure
        assert "roman" in subset_af.tree
        assert "data" in subset_af.tree["roman"]
        assert "meta" in subset_af.tree["roman"]
        assert subset_af.tree["roman"]["meta"]["source_id"] == subset_source_id
        assert len(subset_af.tree["roman"]["meta"]) == 3
        # In lite mode, the 'flag' and 'number_coadds' keys should be removed from the subset data
        assert len(subset_af.tree["roman"]["data"]) == 3 if lite else 5

        # Check that history entry is added to the subset ASDF file
        assert "history" in subset_af.tree
        history_entry = subset_af.tree["history"]["entries"][-1]  # Get the last history entry
        expected_entry = f"Spectral subset created for source ID {subset_source_id} " f"from file {subset_file}."
        assert history_entry["description"] == expected_entry

    # Only select certain source files
    asdf_subsets = subset.get_asdf_subsets(group_by="source_file", spectral_files=[spectral_files[0]])
    source_file_keys = subset.get_source_file_keys(spectral_files=[spectral_files[0]])
    assert len(asdf_subsets) == 2  # Should only have subsets for the specified file
    for key in asdf_subsets.keys():
        subset_file, subset_source_id = source_file_keys[key]
        assert subset_file == spectral_files[0]
        assert subset_source_id in ["420007", "420008"]

    # Only select certain source IDs
    asdf_subsets = subset.get_asdf_subsets(group_by="source_file", source_ids=["420007"])
    source_file_keys = subset.get_source_file_keys(source_ids=["420007"])
    assert len(asdf_subsets) == 2  # Should only have subsets for the specified source ID
    for key in asdf_subsets.keys():
        subset_file, subset_source_id = source_file_keys[key]
        assert subset_source_id == "420007"
        assert subset_file in spectral_files

    # Only select certain source files and source IDs
    asdf_subsets = subset.get_asdf_subsets(
        group_by="source_file", spectral_files=[spectral_files[0]], source_ids=["420007"]
    )
    source_file_keys = subset.get_source_file_keys(spectral_files=[spectral_files[0]], source_ids=["420007"])
    assert len(asdf_subsets) == 1  # Should only have one subset for the specified file and source ID
    for key in asdf_subsets.keys():
        subset_file, subset_source_id = source_file_keys[key]
        assert subset_file == spectral_files[0]
        assert subset_source_id == "420007"


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_asdf_subsets_file(spectral_files, lite):
    # Test the get_asdf_subsets method of RomanSpectralsubset with group_by='file'
    subset = RomanSpectralSubset(spectral_files=spectral_files, source_ids=[420007, 420008], lite=lite, max_workers=1)

    asdf_subsets = subset.get_asdf_subsets(group_by="file")
    assert len(asdf_subsets) == 2  # Should have one subset per file

    for subset_file in asdf_subsets.keys():
        assert subset_file in spectral_files
        subset_af = asdf_subsets[subset_file]
        assert isinstance(subset_af, asdf.AsdfFile)
        # Check that the subset ASDF file contains the expected data structure
        assert "roman" in subset_af.tree
        assert "data" in subset_af.tree["roman"]
        assert len(subset_af.tree["roman"]["data"]) == 2  # Should have subset data for both source IDs
        assert "meta" in subset_af.tree["roman"]
        assert "source_ids" in subset_af.tree["roman"]["meta"]
        assert set(subset_af.tree["roman"]["meta"]["source_ids"]) == {"420007", "420008"}
        assert len(subset_af.tree["roman"]["meta"]) == 3
        # In lite mode, the 'flag' and 'number_coadds' keys should be removed from the subset data
        assert len(subset_af.tree["roman"]["data"]["420007"]) == 3 if lite else 5

        # Check that history entry is added to the subset ASDF file
        assert "history" in subset_af.tree
        history_entry = subset_af.tree["history"]["entries"][-1]  # Get the last history entry
        expected_entry = f"Spectral subset created for source IDs ['420007', '420008'] " f"from file {subset_file}."
        assert history_entry["description"] == expected_entry


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_asdf_subsets_combined(spectral_files, lite):
    # Test the get_asdf_subsets method of RomanSpectralsubset with group_by='combined'
    subset = RomanSpectralSubset(spectral_files=spectral_files, source_ids=[420007, 420008], lite=lite, max_workers=1)
    subset_af = subset.get_asdf_subsets(group_by="combined")
    assert isinstance(subset_af, asdf.AsdfFile)
    # Check that the subset ASDF file contains the expected data structure
    assert "roman" in subset_af.tree
    assert "data" in subset_af.tree["roman"]
    data = subset_af.tree["roman"]["data"]
    assert len(data) == 2  # Should have subset data for both files
    file_str = spectral_files[0]
    file_data = data[file_str]
    assert len(file_data) == 2  # Should have subset data for both source IDs in the file
    assert "meta" in subset_af.tree["roman"]
    meta = subset_af.tree["roman"]["meta"][file_str]
    assert len(meta) == 3
    assert "source_ids" in meta
    assert set(meta["source_ids"]) == {"420007", "420008"}
    # In lite mode, the 'flag' and 'number_coadds' keys should be removed from the subset data
    assert len(file_data["420007"]) == 3 if lite else 5

    # Check that history entry is added to the subset ASDF file
    assert "history" in subset_af.tree
    history_entry = subset_af.tree["history"]["entries"][-1]  # Get the last history entry
    assert "Spectral subset created for source IDs ['420007', '420008']" in history_entry["description"]


def test_roman_spectral_subset_asdf_subsets_error(spectral_files, tmp_path):
    subset = RomanSpectralSubset(
        spectral_files=spectral_files,
        source_ids=[420007, 420008, 430000],
        wl_range=(550, 850),
        lite=True,
        max_workers=1,
    )

    # Raise error if file specified in get_asdf_subsets is not in the original input files
    with pytest.raises(
        InvalidQueryError, match=r"Spectral file .*temp_spectral_nonexistent\.asdf " r"not found in subset results\."
    ):
        subset.get_asdf_subsets(group_by="source_file", spectral_files=[tmp_path / "temp_spectral_nonexistent.asdf"])

    # Raise error if source ID specified in get_asdf_subsets is not in the original input files
    with pytest.raises(InvalidQueryError, match=r"Source ID 999999 not found in subset results\."):
        subset.get_asdf_subsets(group_by="source_file", source_ids=["999999"])

    # Raise error if group_by value is invalid
    with pytest.raises(InvalidInputError, match="Invalid group_by value: 'invalid_group'."):
        subset.get_asdf_subsets(group_by="invalid_group")


def test_roman_spectral_subset_write_as_asdf_source_file(subset, tmp_path):
    # Test the write_as_asdf method of RomanSpectralsubset
    subset_files = subset.write_as_asdf(tmp_path, group_by="source_file")

    # Should write one subset file per source ID and input file combination, so 2 sources x 2 files = 4 subset files
    assert len(subset_files) == 4
    for subset_file in subset_files:
        assert "temp_spectral" in subset_file  # Check that the input file name is in the subset file name
        assert "420007" in subset_file or "420008" in subset_file  # Check that the source ID is in the file name
        assert subset_file.endswith(".asdf")
        assert (tmp_path / subset_file).exists()
    subset_file = subset_files[0]  # Just check the first subset file for simplicity

    # Read the written ASDF file and check that it contains the expected data structure
    with asdf.open(tmp_path / subset_file) as af:
        assert "roman" in af.tree
        assert "data" in af.tree["roman"]
        assert "meta" in af.tree["roman"]
        assert "source_id" in af.tree["roman"]["meta"]
        assert af.tree["roman"]["meta"]["source_id"] in ["420007", "420008"]
        assert "wl" in af.tree["roman"]["data"]
        assert "flux" in af.tree["roman"]["data"]


def test_roman_spectral_subset_write_as_asdf_file(subset, tmp_path):
    # Test the write_as_asdf method of RomanSpectralsubset with group_by='file'
    subset_files = subset.write_as_asdf(tmp_path, group_by="file")

    assert len(subset_files) == 2  # Should write one subset file per input file, so 2 files = 2 subset files
    for subset_file in subset_files:
        assert "temp_spectral" in subset_file  # Check that the input file name is in the subset file name
        assert subset_file.endswith(".asdf")
        assert (tmp_path / subset_file).exists()
    subset_file = subset_files[0]  # Just check the first subset file for simplicity

    # Read the written ASDF file and check that it contains the expected data structure
    with asdf.open(tmp_path / subset_file) as af:
        assert "roman" in af.tree
        assert "data" in af.tree["roman"]
        assert "meta" in af.tree["roman"]
        assert "source_ids" in af.tree["roman"]["meta"]
        assert set(af.tree["roman"]["meta"]["source_ids"]) == {"420007", "420008"}
        assert "wl" in af.tree["roman"]["data"]["420007"]
        assert "flux" in af.tree["roman"]["data"]["420007"]


def test_roman_spectral_subset_write_as_asdf_combined(subset, spectral_files, tmp_path):
    # Test the write_as_asdf method of RomanSpectralsubset with group_by='combined'
    subset_files = subset.write_as_asdf(tmp_path, group_by="combined")

    assert len(subset_files) == 1  # Should write all subsets to a single file
    subset_file = subset_files[0]
    assert "combined_spectral_subset" in subset_file  # Check that the file name indicates it's a combined subset
    assert subset_file.endswith(".asdf")
    assert (tmp_path / subset_file).exists()

    # Read the written ASDF file and check that it contains the expected data structure
    with asdf.open(tmp_path / subset_file) as af:
        assert "roman" in af.tree
        assert "data" in af.tree["roman"]
        assert "meta" in af.tree["roman"]
        file_str = spectral_files[0]
        assert "source_ids" in af.tree["roman"]["meta"][file_str]
        assert set(af.tree["roman"]["meta"][file_str]["source_ids"]) == {"420007", "420008"}
        assert file_str in af.tree["roman"]["data"]
        assert "wl" in af.tree["roman"]["data"][file_str]["420007"]
        assert "flux" in af.tree["roman"]["data"][file_str]["420007"]


def test_roman_spectral_subset_write_as_asdf_invalid_group_by(subset):
    with pytest.raises(InvalidInputError, match=r"Invalid group_by value: 'invalid_group'."):
        subset.write_as_asdf(group_by="invalid_group")


def test_roman_spectral_subset_write_as_asdf_skips_validation_by_default(subset, tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("schema.validate should not be called during default bulk writes")

    monkeypatch.setattr(asdf_schema, "validate", fail_if_called)

    subset_files = subset.write_as_asdf(tmp_path, group_by="source_file")

    assert len(subset_files) == 4
    for subset_file in subset_files:
        assert (tmp_path / subset_file).exists()


def test_roman_spectral_subset_write_as_asdf_can_enable_validation(subset, tmp_path, monkeypatch):
    call_count = {"count": 0}

    def count_calls(*args, **kwargs):
        call_count["count"] += 1

    monkeypatch.setattr(asdf_schema, "validate", count_calls)

    subset_files = subset.write_as_asdf(tmp_path, group_by="source_file", validate_output=True)

    assert len(subset_files) == 4
    assert call_count["count"] > 0
