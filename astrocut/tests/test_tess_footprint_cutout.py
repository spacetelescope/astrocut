from pathlib import Path
import pytest
import re

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from spherical_geometry.polygon import SphericalPolygon

from ..cube_cutout import CubeCutout
from ..exceptions import InvalidInputError, InvalidQueryError
from ..footprint_cutout import get_ffis, ra_dec_crossmatch
from ..tess_footprint_cutout import TessFootprintCutout, cube_cut_from_footprint
from ..tess_cube_cutout import TessCubeCutout


@pytest.fixture
def cutout_size():
    """Fixture to return the cutout size"""
    return 5


@pytest.fixture
def coordinates():
    """Fixture to return the coordinates at the center of the images"""
    return SkyCoord('350 -80', unit='deg')


def check_output_tpf(tpf, ffi_type, sequences=[], cutout_size=5):
    """Helper function to check the validity of output cutout files"""
    tpf_table = tpf[1].data

    # SPOC cutouts have 1 extra columns in EXT 1
    ncols = 12 if ffi_type == 'SPOC' else 11
    assert len(tpf_table.columns) == ncols
    assert tpf_table[0]['FLUX'].shape == (cutout_size, cutout_size)

    # Check that sector matches a provided sequence
    if sequences:
        assert tpf[0].header['SECTOR'] in sequences

    # Close TPF
    tpf.close()


def test_s_region_to_polygon_unsupported_region():
    """Test that ValueError is raised if s_region is not a polygon"""
    s_region = 'CIRCLE'
    err = f'Unsupported s_region type: {s_region}'
    with pytest.raises(ValueError, match=err):
        TessFootprintCutout._s_region_to_polygon(s_region)


@pytest.mark.parametrize("lon, lat, center, expected", [
    ((345, 355, 355, 345), (-15, -15, -5, -5), (350, -10), True),  # intersecting
    ((335, 345, 345, 335), (-15, -15, -5, -5), (340, -10), False),  # non-intersecting
    ((340, 350, 350, 340), (-15, -15, -5, -5), (345, -10), True),  # edge object that intersects
    ((340, 349, 349, 340), (-15, -15, -5, -5), (345, -10), False),  # edge object that does not intersect
])
def test_ffi_intersect(lon, lat, center, expected):
    """Test that FFI intersection with cutout outputs proper results."""
    # SphericalPolygon object for cutout
    cutout_sp = SphericalPolygon.from_radec(lon=(350, 10, 10, 350),
                                            lat=(-10, -10, 10, 10),
                                            center=(0, 0))

    # Create a SphericalPolygon with the parametrized lon, lat, and center
    polygon = SphericalPolygon.from_radec(lon=lon, lat=lat, center=center)

    # Create a table with this polygon
    polygon_table = Table(names=['polygon'], dtype=[SphericalPolygon])
    polygon_table['polygon'] = [polygon]

    # Perform the intersection check
    intersection = TessFootprintCutout._ffi_intersect(polygon_table, cutout_sp)

    # Assert the intersection result matches the expected value
    assert intersection.value[0] == expected


@pytest.mark.parametrize('ffi_type', ['SPOC', 'TICA'])
def test_tess_footprint_cutout(cutout_size, caplog, ffi_type):
    """Test that a single data cube is created for a given sequence"""
    cutout = TessFootprintCutout('130 30', cutout_size, product=ffi_type, sequence=44, verbose=True)

    # Check cutouts attribute
    cutouts = cutout.cutouts
    assert len(cutouts) == 1
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], CubeCutout.CubeCutoutInstance)
    assert cutouts[0].shape[1:] == (5, 5)

    # Check cutouts_by_file attribute
    cutouts_by_file = cutout.cutouts_by_file
    assert len(cutouts_by_file) == 1
    assert isinstance(cutouts_by_file, dict)
    assert isinstance(list(cutouts_by_file.values())[0], CubeCutout.CubeCutoutInstance)

    # Check tpf_cutouts attribute
    tpf_cutouts = cutout.tpf_cutouts
    assert len(tpf_cutouts) == 1
    assert isinstance(tpf_cutouts, list)
    assert isinstance(tpf_cutouts[0], fits.HDUList)
    check_output_tpf(tpf_cutouts[0], ffi_type, [44])

    # Check tpf_cutouts_by_file
    tpf_cutouts_by_file = cutout.tpf_cutouts_by_file
    tpf_cutout = list(tpf_cutouts_by_file.values())[0]
    assert len(tpf_cutouts_by_file) == 1
    assert isinstance(tpf_cutouts_by_file, dict)
    assert isinstance(tpf_cutout, fits.HDUList)
    check_output_tpf(tpf_cutout, ffi_type, [44])

    # Check tess_cube attribute
    tess_cube = cutout.tess_cube_cutout
    assert isinstance(tess_cube, TessCubeCutout)
    
    # Assert that messages were printed
    captured = caplog.text
    assert 'Coordinates:' in captured
    assert 'Cutout size: [5 5]' in captured
    assert re.search(r'Found \d+ footprint files.', captured)
    assert re.search(r'Filtered to \d+ footprints for sequences: 44', captured)
    assert re.search(r'Found \d+ matching files.', captured)
    assert 'Generating cutouts...' in captured

    # Check that _extract_sequence_information works correctly
    # Should return empty dict if sector name does not match product
    sector_name = 'hlsp_tica_s' if ffi_type == 'SPOC' else 'tess_s'
    sector_name += '0044-4-1'
    info = cutout._extract_sequence_information(sector_name)
    assert info == {}


def test_tess_footprint_cutout_multi_sequence(coordinates, cutout_size):
    """Test that a cube is created for each sequence when multiple are provided"""
    sequences = [1, 13]
    cutout = TessFootprintCutout(coordinates, cutout_size, product='SPOC', sequence=sequences)
    cutout_tpfs = cutout.tpf_cutouts
    assert len(cutout_tpfs) == 2

    for tpf in cutout_tpfs:
        check_output_tpf(tpf, 'SPOC', sequences)


def test_tess_footprint_cutout_all_sequences(coordinates, cutout_size):
    """Test that cubes are created for all sequences that intersect the cutout"""
    # Create cutouts for all possible sequences
    cutout = TessFootprintCutout(coordinates, cutout_size, product='SPOC')
    cutout_tpfs = cutout.tpf_cutouts
    assert len(cutout_tpfs) >= 5

    # Crossmatch to get sectors that contain cutout
    all_ffis = get_ffis(cutout._s3_footprint_cache)
    cone_results = ra_dec_crossmatch(all_ffis, '350 -80', cutout_size, 21)
    seq_list = cutout._create_sequence_list(cone_results)
    sequences = [int(seq['sector']) for seq in seq_list]

    # Assert non-empty results
    assert len(seq_list) == len(cutout_tpfs)
    for tpf in cutout_tpfs:
        check_output_tpf(tpf, 'SPOC', sequences)


def test_tess_footprint_cutout_write_as_tpf(coordinates, cutout_size, tmpdir):
    """Test that TPF files are written to disk"""
    cutout = TessFootprintCutout(coordinates, cutout_size, sequence=[1, 13])
    paths = cutout.write_as_tpf(output_dir=tmpdir)

    for cutout_path in paths:
        path = Path(cutout_path)
        assert path.exists()
        assert path.suffix == '.fits'

        # Check that file can be opened
        with fits.open(path) as hdu:
            hdu.info()


def test_tess_footprint_cutout_invalid_sequence(coordinates, cutout_size):
    """Test that InvalidQueryError is raised if sequence does not have cube files"""
    err = 'No files were found for sequences: -1'
    with pytest.raises(InvalidQueryError, match=err):
        TessFootprintCutout(coordinates, cutout_size, sequence=-1)


def test_tess_footprint_cutout_outside_coords(coordinates, cutout_size):
    """Test that InvalidQueryError is raised if coordinates are not found in sequence"""
    err = 'The given coordinates were not found within the specified sequence(s).'
    with pytest.raises(InvalidQueryError, match=re.escape(err)):
        TessFootprintCutout(coordinates, cutout_size, sequence=2)


def test_tess_footprint_cutout_invalid_product(coordinates, cutout_size):
    """Test that InvalidQueryError is raised if an invalid product is given"""
    err = 'Product for TESS cube cutouts must be either "SPOC" or "TICA".'
    with pytest.raises(InvalidInputError, match=err):
        TessFootprintCutout(coordinates, cutout_size, product='invalid')
        

def test_cube_cut_from_footprint(coordinates, cutout_size, tmpdir):
    """Test that data cube is cut from FFI file using parallel processing"""
    # Writing to memory, should return cutouts as memory objects
    cutouts = cube_cut_from_footprint(coordinates, 
                                      cutout_size,
                                      sequence=13,
                                      memory_only=True)
    assert len(cutouts) == 1
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], fits.HDUList)

    # Writing to disk, should return cutout filepaths
    cutouts = cube_cut_from_footprint(coordinates, 
                                      cutout_size,
                                      sequence=13,
                                      output_dir=tmpdir)
    assert len(cutouts) == 1
    assert isinstance(cutouts, list)
    assert isinstance(cutouts[0], str)
    assert str(tmpdir) in cutouts[0]
