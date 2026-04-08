from copy import deepcopy
from pathlib import Path

import asdf
import numpy as np
import pytest

from astrocut.exceptions import DataWarning, InvalidQueryError
from astrocut.roman_spectral_subset import RomanSpectralSubset, roman_spectral_subset


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
            }

        # Add some metadata
        af["roman"]["meta"] = {"unit_flux": "W m**(-2) nm**(-1)", "unit_wl": "nm"}

        # Write the ASDF file to TWO temporary locations
        temp_file = tmp_path / "temp_spectral.asdf"
        af.write_to(temp_file)

        # Add slightly more data to a second file
        af_copy = asdf.AsdfFile(deepcopy(af.tree))
        af_copy["roman"]["data"]["430000"] = {
            "flag": np.array([False, True, False, True]),
            "wl": np.array([500, 600, 700, 800]),
            "flux": np.array([15, 25, 35, 45]),
            "flux_error": np.array([1.5, 2.5, 3.5, 4.5]),
        }
        temp_file2 = tmp_path / "temp_spectral_copy.asdf"
        af_copy.write_to(temp_file2)

        yield [temp_file, temp_file2]


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_data(spectral_files, lite):
    # Data extraction with no wavelength range filtering
    subset = RomanSpectralSubset(
        spectral_files=spectral_files,  # Test with multiple files (same file for simplicity)
        source_ids=[420007, 420008],
        lite=lite,
    )

    subset_data = subset.subset_data
    assert len(subset_data) == 2  # Should have subset data for both files

    subset_data_file = subset_data[str(spectral_files[0])]
    assert "420007" in subset_data_file
    assert "420008" in subset_data_file
    assert len(subset_data_file.keys()) == 2  # Should only contain the specified source IDs
    assert "flux" in subset_data_file["420007"]
    assert "wl" in subset_data_file["420007"]
    # In lite mode, the 'flag' key should be removed from the subset data
    if lite:
        assert "flag" not in subset_data_file["420007"]
    else:
        assert "flag" in subset_data_file["420007"]

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


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_data_wavelength_filter(spectral_files, lite):
    # Test with wavelength range
    subset = RomanSpectralSubset(
        spectral_files=spectral_files,  # Test with multiple files (same file for simplicity)
        source_ids=[420007, 420008],
        wl_range=(550, 850),
        lite=lite,
    )

    subset_data_file = subset.subset_data[str(spectral_files[0])]
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


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_asdf_subsets_source_file(spectral_files, lite):
    # Test the get_asdf_subsets method of RomanSpectralsubset with group_by='source_file'
    subset = RomanSpectralSubset(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=lite
    )

    asdf_subsets = subset.get_asdf_subsets(group_by="source_file")
    # Should make a subset for every combination of source ID and file, so 2 sources x 2 files = 4 subsets
    assert len(asdf_subsets) == 4

    for subset_file, subset_source_id in asdf_subsets.keys():
        assert Path(subset_file) in spectral_files
        assert subset_source_id in ["420007", "420008"]
        subset_af = asdf_subsets[(subset_file, subset_source_id)]
        assert isinstance(subset_af, asdf.AsdfFile)
        # Check that the subset ASDF file contains the expected data structure
        assert "roman" in subset_af.tree
        assert "data" in subset_af.tree["roman"]
        assert "meta" in subset_af.tree["roman"]
        assert subset_af.tree["roman"]["meta"]["source_id"] == subset_source_id
        assert len(subset_af.tree["roman"]["meta"]) == 3
        # In lite mode, the 'flag' key should be removed from the subset data
        assert len(subset_af.tree["roman"]["data"]) == 3 if lite else 4

        # Check that history entry is added to the subset ASDF file
        assert "history" in subset_af.tree
        history_entry = subset_af.tree["history"]["entries"][-1]  # Get the last history entry
        expected_entry = (
            f"Spectral subset created for source ID {subset_source_id} "
            f"from file {subset_file} with wavelength range (550, 850)."
        )
        assert history_entry["description"] == expected_entry

    # Only select certain source files
    asdf_subsets = subset.get_asdf_subsets(group_by="source_file", spectral_files=[spectral_files[0]])
    assert len(asdf_subsets) == 2  # Should only have subsets for the specified file
    for subset_file, subset_source_id in asdf_subsets.keys():
        assert Path(subset_file) == spectral_files[0]
        assert subset_source_id in ["420007", "420008"]

    # Only select certain source IDs
    asdf_subsets = subset.get_asdf_subsets(group_by="source_file", source_ids=["420007"])
    assert len(asdf_subsets) == 2  # Should only have subsets for the specified source ID
    for subset_file, subset_source_id in asdf_subsets.keys():
        assert subset_source_id == "420007"
        assert Path(subset_file) in spectral_files

    # Only select certain source files and source IDs
    asdf_subsets = subset.get_asdf_subsets(
        group_by="source_file", spectral_files=[spectral_files[0]], source_ids=["420007"]
    )
    assert len(asdf_subsets) == 1  # Should only have one subset for the specified file and source ID
    for subset_file, subset_source_id in asdf_subsets.keys():
        assert Path(subset_file) == spectral_files[0]
        assert subset_source_id == "420007"


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_asdf_subsets_file(spectral_files, lite):
    # Test the get_asdf_subsets method of RomanSpectralsubset with group_by='file'
    subset = RomanSpectralSubset(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=lite
    )

    asdf_subsets = subset.get_asdf_subsets(group_by="file")
    assert len(asdf_subsets) == 2  # Should have one subset per file

    for subset_file in asdf_subsets.keys():
        assert Path(subset_file) in spectral_files
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
        # In lite mode, the 'flag' key should be removed from the subset data
        assert len(subset_af.tree["roman"]["data"]["420007"]) == 3 if lite else 4

        # Check that history entry is added to the subset ASDF file
        assert "history" in subset_af.tree
        history_entry = subset_af.tree["history"]["entries"][-1]  # Get the last history entry
        expected_entry = (
            f"Spectral subset created for source IDs ['420007', '420008'] "
            f"from file {subset_file} with wavelength range (550, 850)."
        )
        assert history_entry["description"] == expected_entry


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_subset_asdf_subsets_combined(spectral_files, lite):
    # Test the get_asdf_subsets method of RomanSpectralsubset with group_by='combined'
    subset = RomanSpectralSubset(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=lite
    )

    subset_af = subset.get_asdf_subsets(group_by="combined")
    assert isinstance(subset_af, asdf.AsdfFile)
    # Check that the subset ASDF file contains the expected data structure
    assert "roman" in subset_af.tree
    assert "data" in subset_af.tree["roman"]
    data = subset_af.tree["roman"]["data"]
    assert len(data) == 2  # Should have subset data for both files
    file_str = str(spectral_files[0])
    file_data = data[file_str]
    assert len(file_data) == 2  # Should have subset data for both source IDs in the file
    assert "meta" in subset_af.tree["roman"]
    meta = subset_af.tree["roman"]["meta"][file_str]
    assert len(meta) == 3
    assert "source_ids" in meta
    assert set(meta["source_ids"]) == {"420007", "420008"}
    # In lite mode, the 'flag' key should be removed from the subset data
    assert len(file_data["420007"]) == 3 if lite else 4

    # Check that history entry is added to the subset ASDF file
    assert "history" in subset_af.tree
    history_entry = subset_af.tree["history"]["entries"][-1]  # Get the last history entry
    assert "Spectral subset created for source IDs ['420007', '420008']" in history_entry["description"]


def test_roman_spectral_subset_error(spectral_files, tmp_path):
    # Warn if wavelength region does not overlap with any data in the file
    with pytest.warns(DataWarning, match=r"Wavelength range \(810, 910\) is out of bounds for source ID 430000"):
        RomanSpectralSubset(
            spectral_files=spectral_files[1],
            source_ids=[420007, 420008, 430000],
            wl_range=(810, 910),
            lite=True,
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
            verbose=True,
        )

    # Error if no subsets are created
    with pytest.raises(InvalidQueryError, match=r"No subsets were created."):
        RomanSpectralSubset(
            spectral_files=spectral_files,
            source_ids=[123456],  # Source ID that doesn't exist in either file
            wl_range=(810, 910),
            lite=True,
            verbose=False,
        )


def test_roman_spectral_subset_asdf_subsets_error(spectral_files, tmp_path):
    subset = RomanSpectralSubset(
        spectral_files=spectral_files,
        source_ids=[420007, 420008, 430000],
        wl_range=(550, 850),
        lite=True,
        verbose=False,
    )

    # Raise error if file specified in get_asdf_subsets is not in the original input files
    with pytest.raises(
        InvalidQueryError, match=r"Spectral file .*temp_spectral_nonexistent\.asdf " r"not found in subset results\."
    ):
        subset.get_asdf_subsets(group_by="source_file", spectral_files=[tmp_path / "temp_spectral_nonexistent.asdf"])

    # Raise error if source ID specified in get_asdf_subsets is not in the original input files
    with pytest.raises(InvalidQueryError, match=r"Source ID 999999 not found in subset results\."):
        subset.get_asdf_subsets(group_by="source_file", source_ids=["999999"])

    # Warn if source ID specified in get_asdf_subsets is not in one of the files but is in another file
    with pytest.warns(
        DataWarning,
        match=r"Source ID 430000 not found in file " r".*temp_spectral\.asdf\. Skipping this source for this file\.",
    ):
        subset.get_asdf_subsets(group_by="source_file", source_ids=["430000"])


def test_roman_spectral_subset_write_as_asdf_source_file(spectral_files, tmp_path):
    # Test the write_as_asdf method of RomanSpectralsubset
    subset = RomanSpectralSubset(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=True
    )
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


def test_roman_spectral_subset_write_as_asdf_file(spectral_files, tmp_path):
    # Test the write_as_asdf method of RomanSpectralsubset with group_by='file'
    subset = RomanSpectralSubset(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=True
    )
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


def test_roman_spectral_subset_write_as_asdf_combined(spectral_files, tmp_path):
    # Test the write_as_asdf method of RomanSpectralsubset with group_by='combined'
    subset = RomanSpectralSubset(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=True
    )
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
        file_str = str(spectral_files[0])
        assert "source_ids" in af.tree["roman"]["meta"][file_str]
        assert set(af.tree["roman"]["meta"][file_str]["source_ids"]) == {"420007", "420008"}
        file_str = str(spectral_files[0])
        assert file_str in af.tree["roman"]["data"]
        assert "wl" in af.tree["roman"]["data"][file_str]["420007"]
        assert "flux" in af.tree["roman"]["data"][file_str]["420007"]


def test_roman_spectral_cut(spectral_files, tmp_path):
    # Test the roman_spectral_cut function
    subset_files = roman_spectral_subset(
        spectral_files=spectral_files,
        source_ids=[i for i in range(420007, 420007 + 100)],
        wl_range=(550, 850),
        lite=True,
        output_dir=tmp_path,
    )

    # Check that the subset files were created
    assert len(subset_files) == 200  # Should create one subset file per source and file

    for subset_file in subset_files:
        assert subset_file.endswith(".asdf")
        assert (tmp_path / subset_file).exists()

    # Force an InvalidQueryError for one of the files
    subset_files = roman_spectral_subset(
        spectral_files=spectral_files,
        source_ids=[430000],  # Source ID that only exists in the second file
        wl_range=(550, 850),
        lite=True,
        output_dir=tmp_path,
    )
    assert len(subset_files) == 1  # Should only create a subset for the file that contains the source ID
    subset_file = subset_files[0]
    assert subset_file.endswith(".asdf")
    assert spectral_files[1].stem in subset_file  # Check that the subset file corresponds to the correct input file
    assert (tmp_path / subset_file).exists()
