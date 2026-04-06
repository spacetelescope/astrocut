from copy import deepcopy
from pathlib import Path

import asdf
import numpy as np
import pytest

from astrocut.exceptions import DataWarning, InvalidQueryError
from astrocut.roman_spectral_cutout import RomanSpectralCutout, roman_spectral_cut


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
        }
        temp_file2 = tmp_path / "temp_spectral_copy.asdf"
        af_copy.write_to(temp_file2)

        yield [temp_file, temp_file2]


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_cutout_data(spectral_files, lite):
    # Data extraction with no wavelength range filtering
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files,  # Test with multiple files (same file for simplicity)
        source_ids=[420007, 420008],
        lite=lite,
    )

    cutout_data = cutout.cutout_data
    assert len(cutout_data) == 2  # Should have cutout data for both files

    cutout_data_file = cutout_data[str(spectral_files[0])]
    assert "420007" in cutout_data_file
    assert "420008" in cutout_data_file
    assert len(cutout_data_file.keys()) == 2  # Should only contain the specified source IDs
    assert "flux" in cutout_data_file["420007"]
    assert "wl" in cutout_data_file["420007"]
    # In lite mode, the 'flag' key should be removed from the cutout data
    if lite:
        assert "flag" not in cutout_data_file["420007"]
    else:
        assert "flag" in cutout_data_file["420007"]

    cutout_wl = cutout_data_file["420007"]["wl"]
    cutout_flux = cutout_data_file["420007"]["flux"]
    expected_wl = np.array([500, 600, 700, 800, 900, 1000])
    expected_flux = np.array([10, 20, 30, 40, 50, 60])
    assert np.array_equal(cutout_wl, expected_wl)
    assert np.array_equal(cutout_flux, expected_flux)

    if not lite:
        cutout_flag = cutout_data_file["420007"]["flag"]
        expected_flag = np.array([True, False, True, False, True, False])
        assert np.array_equal(cutout_flag, expected_flag)


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_cutout_data_wavelength_filter(spectral_files, lite):
    # Test with wavelength range
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files,  # Test with multiple files (same file for simplicity)
        source_ids=[420007, 420008],
        wl_range=(550, 850),
        lite=lite,
    )

    cutout_data_file = cutout.cutout_data[str(spectral_files[0])]
    assert "flux" in cutout_data_file["420007"]
    assert "wl" in cutout_data_file["420007"]
    cutout_wl = cutout_data_file["420007"]["wl"]
    cutout_flux = cutout_data_file["420007"]["flux"]

    # Check that the wavelength range is correctly applied
    assert np.all((cutout_wl >= 550) & (cutout_wl <= 850))
    # Check that the flux values correspond to the correct wavelength range
    expected_flux = np.array([20, 30, 40])  # Corresponding to wl = 600, 700, 800
    assert np.array_equal(cutout_flux, expected_flux)

    if not lite:
        cutout_flag = cutout_data_file["420007"]["flag"]
        expected_flag = np.array([False, True, False])  # Corresponding to wl = 600, 700, 800
        assert np.array_equal(cutout_flag, expected_flag)


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_cutout_asdf_cutouts_source_file(spectral_files, lite):
    # Test the get_asdf_cutouts method of RomanSpectralCutout with group_by='source_file'
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=lite
    )

    asdf_cutouts = cutout.get_asdf_cutouts(group_by="source_file")
    # Should make a cutout for every combination of source ID and file, so 2 sources x 2 files = 4 cutouts
    assert len(asdf_cutouts) == 4

    for cutout_file, cutout_source_id in asdf_cutouts.keys():
        assert Path(cutout_file) in spectral_files
        assert cutout_source_id in ["420007", "420008"]
        cutout_af = asdf_cutouts[(cutout_file, cutout_source_id)]
        assert isinstance(cutout_af, asdf.AsdfFile)
        # Check that the cutout ASDF file contains the expected data structure
        assert "roman" in cutout_af.tree
        assert "data" in cutout_af.tree["roman"]
        assert "meta" in cutout_af.tree["roman"]
        assert cutout_af.tree["roman"]["meta"]["source_id"] == cutout_source_id

        # Check that the data contains the expected keys and that the lite mode has the correct structure
        if lite:
            assert len(cutout_af.tree["roman"]["meta"]) == 1  # In lite mode, only 'source_id' should be in the meta
            assert len(cutout_af.tree["roman"]["data"]) == 2  # In lite mode, only 'wl' and 'flux' should be in the data
        else:
            assert (
                len(cutout_af.tree["roman"]["meta"]) == 3
            )  # In full mode, 'source_id', 'unit_flux', and 'unit_wl' should be in the meta
            assert (
                len(cutout_af.tree["roman"]["data"]) == 3
            )  # In full mode, 'wl', 'flux', and 'flag' should be in the data

        # Check that history entry is added to the cutout ASDF file
        assert "history" in cutout_af.tree
        history_entry = cutout_af.tree["history"]["entries"][-1]  # Get the last history entry
        expected_entry = (
            f"Spectral cutout created for source ID {cutout_source_id} "
            f"from file {cutout_file} with wavelength range (550, 850)."
        )
        assert history_entry["description"] == expected_entry

    # Only select certain source files
    asdf_cutouts = cutout.get_asdf_cutouts(group_by="source_file", spectral_files=[spectral_files[0]])
    assert len(asdf_cutouts) == 2  # Should only have cutouts for the specified file
    for cutout_file, cutout_source_id in asdf_cutouts.keys():
        assert Path(cutout_file) == spectral_files[0]
        assert cutout_source_id in ["420007", "420008"]

    # Only select certain source IDs
    asdf_cutouts = cutout.get_asdf_cutouts(group_by="source_file", source_ids=["420007"])
    assert len(asdf_cutouts) == 2  # Should only have cutouts for the specified source ID
    for cutout_file, cutout_source_id in asdf_cutouts.keys():
        assert cutout_source_id == "420007"
        assert Path(cutout_file) in spectral_files

    # Only select certain source files and source IDs
    asdf_cutouts = cutout.get_asdf_cutouts(
        group_by="source_file", spectral_files=[spectral_files[0]], source_ids=["420007"]
    )
    assert len(asdf_cutouts) == 1  # Should only have one cutout for the specified file and source ID
    for cutout_file, cutout_source_id in asdf_cutouts.keys():
        assert Path(cutout_file) == spectral_files[0]
        assert cutout_source_id == "420007"


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_cutout_asdf_cutouts_file(spectral_files, lite):
    # Test the get_asdf_cutouts method of RomanSpectralCutout with group_by='file'
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=lite
    )

    asdf_cutouts = cutout.get_asdf_cutouts(group_by="file")
    assert len(asdf_cutouts) == 2  # Should have one cutout per file

    for cutout_file in asdf_cutouts.keys():
        assert Path(cutout_file) in spectral_files
        cutout_af = asdf_cutouts[cutout_file]
        assert isinstance(cutout_af, asdf.AsdfFile)
        # Check that the cutout ASDF file contains the expected data structure
        assert "roman" in cutout_af.tree
        assert "data" in cutout_af.tree["roman"]
        assert len(cutout_af.tree["roman"]["data"]) == 2  # Should have cutout data for both source IDs
        assert "meta" in cutout_af.tree["roman"]
        assert "source_ids" in cutout_af.tree["roman"]["meta"]
        assert set(cutout_af.tree["roman"]["meta"]["source_ids"]) == {"420007", "420008"}

        if lite:
            assert len(cutout_af.tree["roman"]["meta"]) == 1  # In lite mode, only 'source_ids' should be in the meta
            assert len(cutout_af.tree["roman"]["data"]["420007"]) == 2  # Only 'wl' and 'flux' should be in the data
        else:
            # In full mode, 'source_ids', 'unit_flux', and 'unit_wl' should be in the meta
            assert len(cutout_af.tree["roman"]["meta"]) == 3
            assert len(cutout_af.tree["roman"]["data"]["420007"]) == 3  # 'wl', 'flux', and 'flag' should be in the data

        # Check that history entry is added to the cutout ASDF file
        assert "history" in cutout_af.tree
        history_entry = cutout_af.tree["history"]["entries"][-1]  # Get the last history entry
        expected_entry = (
            f"Spectral cutout created for source IDs ['420007', '420008'] "
            f"from file {cutout_file} with wavelength range (550, 850)."
        )
        assert history_entry["description"] == expected_entry


@pytest.mark.parametrize("lite", [True, False])
def test_roman_spectral_cutout_asdf_cutouts_combined(spectral_files, lite):
    # Test the get_asdf_cutouts method of RomanSpectralCutout with group_by='combined'
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=lite
    )

    cutout_af = cutout.get_asdf_cutouts(group_by="combined")
    assert isinstance(cutout_af, asdf.AsdfFile)
    # Check that the cutout ASDF file contains the expected data structure
    assert "roman" in cutout_af.tree
    assert "data" in cutout_af.tree["roman"]
    data = cutout_af.tree["roman"]["data"]
    assert len(data) == 2  # Should have cutout data for both files
    file_str = str(spectral_files[0])
    file_data = data[file_str]
    assert len(file_data) == 2  # Should have cutout data for both source IDs in the file
    assert "meta" in cutout_af.tree["roman"]
    meta = cutout_af.tree["roman"]["meta"] if lite else cutout_af.tree["roman"]["meta"][file_str]
    assert "source_ids" in meta
    assert set(meta["source_ids"]) == {"420007", "420008"}

    if lite:
        assert len(meta) == 1  # In lite mode, only 'source_ids' should be in the meta
        assert len(file_data["420007"]) == 2  # In lite mode, only 'wl' and 'flux' should be in the data
    else:
        assert len(meta) == 3  # In full mode, 'source_ids', 'unit_flux', and 'unit_wl' should be in the meta
        assert len(file_data["420007"]) == 3  # In full mode, 'wl', 'flux', and 'flag' should be in the data

    # Check that history entry is added to the cutout ASDF file
    assert "history" in cutout_af.tree
    history_entry = cutout_af.tree["history"]["entries"][-1]  # Get the last history entry
    assert "Spectral cutout created for source IDs ['420007', '420008']" in history_entry["description"]


def test_roman_spectral_cutout_error(spectral_files, tmp_path):
    # Warn if wavelength region does not overlap with any data in the file
    with pytest.warns(DataWarning, match=r"Wavelength range \(810, 910\) is out of bounds for source ID 430000"):
        RomanSpectralCutout(
            spectral_files=spectral_files[1],
            source_ids=[420007, 420008, 430000],
            wl_range=(810, 910),
            lite=True,
            verbose=True,
        )

    # Warn if source ID specified in get_asdf_cutouts is not in one of the files but is in another file
    with pytest.warns(
        DataWarning,
        match=r"Source ID 430000 not found in file .*temp_spectral\.asdf\. " r"Skipping this source for this file\.",
    ):
        RomanSpectralCutout(
            spectral_files=spectral_files,
            source_ids=[420007, 420008, 430000],
            wl_range=(550, 850),
            lite=True,
            verbose=True,
        )

    # Error if no cutouts are created
    with pytest.raises(InvalidQueryError, match=r"No cutouts were created."):
        RomanSpectralCutout(
            spectral_files=spectral_files,
            source_ids=[123456],  # Source ID that doesn't exist in either file
            wl_range=(810, 910),
            lite=True,
            verbose=False,
        )


def test_roman_spectral_cutout_asdf_cutouts_error(spectral_files, tmp_path):
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files,
        source_ids=[420007, 420008, 430000],
        wl_range=(550, 850),
        lite=True,
        verbose=False,
    )

    # Raise error if file specified in get_asdf_cutouts is not in the original input files
    with pytest.raises(
        InvalidQueryError, match=r"Spectral file .*temp_spectral_nonexistent\.asdf " r"not found in cutout results\."
    ):
        cutout.get_asdf_cutouts(group_by="source_file", spectral_files=[tmp_path / "temp_spectral_nonexistent.asdf"])

    # Raise error if source ID specified in get_asdf_cutouts is not in the original input files
    with pytest.raises(InvalidQueryError, match=r"Source ID 999999 not found in cutout results\."):
        cutout.get_asdf_cutouts(group_by="source_file", source_ids=["999999"])

    # Warn if source ID specified in get_asdf_cutouts is not in one of the files but is in another file
    with pytest.warns(
        DataWarning,
        match=r"Source ID 430000 not found in file " r".*temp_spectral\.asdf\. Skipping this source for this file\.",
    ):
        cutout.get_asdf_cutouts(group_by="source_file", source_ids=["430000"])


def test_roman_spectral_cutout_write_as_asdf_source_file(spectral_files, tmp_path):
    # Test the write_as_asdf method of RomanSpectralCutout
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=True
    )
    cutout_files = cutout.write_as_asdf(tmp_path, group_by="source_file")

    # Should write one cutout file per source ID and input file combination, so 2 sources x 2 files = 4 cutout files
    assert len(cutout_files) == 4
    for cutout_file in cutout_files:
        assert "temp_spectral" in cutout_file  # Check that the input file name is in the cutout file name
        assert "420007" in cutout_file or "420008" in cutout_file  # Check that the source ID is in the file name
        assert cutout_file.endswith(".asdf")
        assert (tmp_path / cutout_file).exists()
    cutout_file = cutout_files[0]  # Just check the first cutout file for simplicity

    # Read the written ASDF file and check that it contains the expected data structure
    with asdf.open(tmp_path / cutout_file) as af:
        assert "roman" in af.tree
        assert "data" in af.tree["roman"]
        assert "meta" in af.tree["roman"]
        assert "source_id" in af.tree["roman"]["meta"]
        assert af.tree["roman"]["meta"]["source_id"] in ["420007", "420008"]
        assert "wl" in af.tree["roman"]["data"]
        assert "flux" in af.tree["roman"]["data"]


def test_roman_spectral_cutout_write_as_asdf_file(spectral_files, tmp_path):
    # Test the write_as_asdf method of RomanSpectralCutout with group_by='file'
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=True
    )
    cutout_files = cutout.write_as_asdf(tmp_path, group_by="file")

    assert len(cutout_files) == 2  # Should write one cutout file per input file, so 2 files = 2 cutout files
    for cutout_file in cutout_files:
        assert "temp_spectral" in cutout_file  # Check that the input file name is in the cutout file name
        assert cutout_file.endswith(".asdf")
        assert (tmp_path / cutout_file).exists()
    cutout_file = cutout_files[0]  # Just check the first cutout file for simplicity

    # Read the written ASDF file and check that it contains the expected data structure
    with asdf.open(tmp_path / cutout_file) as af:
        assert "roman" in af.tree
        assert "data" in af.tree["roman"]
        assert "meta" in af.tree["roman"]
        assert "source_ids" in af.tree["roman"]["meta"]
        assert set(af.tree["roman"]["meta"]["source_ids"]) == {"420007", "420008"}
        assert "wl" in af.tree["roman"]["data"]["420007"]
        assert "flux" in af.tree["roman"]["data"]["420007"]


def test_roman_spectral_cutout_write_as_asdf_combined(spectral_files, tmp_path):
    # Test the write_as_asdf method of RomanSpectralCutout with group_by='combined'
    cutout = RomanSpectralCutout(
        spectral_files=spectral_files, source_ids=[420007, 420008], wl_range=(550, 850), lite=True
    )
    cutout_files = cutout.write_as_asdf(tmp_path, group_by="combined")

    assert len(cutout_files) == 1  # Should write all cutouts to a single file
    cutout_file = cutout_files[0]
    assert "combined_spectral_cutout" in cutout_file  # Check that the file name indicates it's a combined cutout
    assert cutout_file.endswith(".asdf")
    assert (tmp_path / cutout_file).exists()

    # Read the written ASDF file and check that it contains the expected data structure
    with asdf.open(tmp_path / cutout_file) as af:
        assert "roman" in af.tree
        assert "data" in af.tree["roman"]
        assert "meta" in af.tree["roman"]
        assert "source_ids" in af.tree["roman"]["meta"]
        assert set(af.tree["roman"]["meta"]["source_ids"]) == {"420007", "420008"}
        file_str = str(spectral_files[0])
        assert file_str in af.tree["roman"]["data"]
        assert "wl" in af.tree["roman"]["data"][file_str]["420007"]
        assert "flux" in af.tree["roman"]["data"][file_str]["420007"]


def test_roman_spectral_cut(spectral_files, tmp_path):
    # Test the roman_spectral_cut function
    cutout_files = roman_spectral_cut(
        spectral_files=spectral_files,
        source_ids=[i for i in range(420007, 420007 + 100)],
        wl_range=(550, 850),
        lite=True,
        output_dir=tmp_path,
    )

    # Check that the cutout files were created
    assert len(cutout_files) == 200  # Should create one cutout file per source and file

    for cutout_file in cutout_files:
        assert cutout_file.endswith(".asdf")
        assert (tmp_path / cutout_file).exists()

    # Force an InvalidQueryError for one of the files
    cutout_files = roman_spectral_cut(
        spectral_files=spectral_files,
        source_ids=[430000],  # Source ID that only exists in the second file
        wl_range=(550, 850),
        lite=True,
        output_dir=tmp_path,
    )
    assert len(cutout_files) == 1  # Should only create a cutout for the file that contains the source ID
    cutout_file = cutout_files[0]
    assert cutout_file.endswith(".asdf")
    assert spectral_files[1].stem in cutout_file  # Check that the cutout file corresponds to the correct input file
    assert (tmp_path / cutout_file).exists()
