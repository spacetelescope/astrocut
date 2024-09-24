import re
import numpy as np
import pytest

from astrocut.exceptions import InvalidQueryError
from astrocut.footprint_cutouts import (cube_cut_from_footprint, _extract_sequence_information, _s_region_to_polygon, 
                                        _get_s3_ffis, _ffi_intersect, _ra_dec_crossmatch, _create_sequence_list)
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from spherical_geometry.polygon import SphericalPolygon


def check_output_file(path, ffi_type, sequences=[]):
    """Helper function to check the validity of output cutout files"""
    # Check that cutout path point to valid FITS file
    tpf = fits.open(path)
    tpf_table = tpf[1].data

    # SPOC cutouts have 1 extra columns in EXT 1
    ncols = 12 if ffi_type == 'SPOC' else 11
    assert len(tpf_table.columns) == ncols
    assert tpf_table[0]['FLUX'].shape == (5, 5)

    # Check that sector matches a provided sequence
    if sequences:
        assert tpf[0].header['SECTOR'] in sequences

    tpf.close()


def test_s_region_to_polygon_unsupported_region():
    """Test that ValueError is raised if s_region is not a polygon"""
    s_region = 'CIRCLE'
    err = 'Unsupported S_Region type.'
    with pytest.raises(ValueError, match=err):
        _s_region_to_polygon(s_region)


def test_ffi_intersect():
    """Test that FFI intersection with cutout outputs proper results"""
    # SphericalPolygon object for cutout
    cutout_sp = SphericalPolygon.from_radec(lon=(350, 10, 10, 350),
                                            lat=(-10, -10, 10, 10), 
                                            center=(0, 0))

    # Intersecting object
    intersecting = SphericalPolygon.from_radec(lon=(345, 355, 355, 345),
                                               lat=(-15, -15, -5, -5),
                                               center=(350, -10))

    # Non-intersecting object
    nonintersecting = SphericalPolygon.from_radec(lon=(335, 345, 345, 335),
                                                  lat=(-15, -15, -5, -5),
                                                  center=(340, -10))
    
    # Edge object that intersects
    edge_intersect = SphericalPolygon.from_radec(lon=(340, 350, 350, 340),
                                                 lat=(-15, -15, -5, -5),
                                                 center=(345, -10))
    
    # Edge object that does not intersect
    edge_nonintersect = SphericalPolygon.from_radec(lon=(340, 349, 349, 340),
                                                    lat=(-15, -15, -5, -5),
                                                    center=(345, -10))
    
    polygon_table = Table(names=['polygon'], dtype=[SphericalPolygon])
    polygon_table['polygon'] = [intersecting, nonintersecting, edge_intersect, edge_nonintersect]
    intersection = _ffi_intersect(polygon_table, cutout_sp)
    assert np.array_equal(intersection.value, np.array([True, False, True, False]))


def test_extract_sequence_information_unknown_product():
    """Test that an empty dict is returned if product is not recognized"""
    info = _extract_sequence_information('tess-s0044-4-1', product='UNKNOWN')
    assert info == {}

    info = _extract_sequence_information('tess-s0044-4-1', product=None)
    assert info == {}


def test_extract_sequence_information_no_match():
    """Test that an empty dict is returned if name does not match product pattern"""
    info = _extract_sequence_information('tess-s0044-4-1', product='TICA')
    assert info == {}


@pytest.mark.parametrize('ffi_type', ['SPOC', 'TICA'])
def test_cube_cut_from_footprint(tmpdir, capsys, ffi_type):
    """Test that data cube is cut from FFI file using parallel processing"""
    cutout = cube_cut_from_footprint(coordinates='130 30', 
                                     cutout_size=5,
                                     product=ffi_type,
                                     output_dir=tmpdir,
                                     sequence=44,
                                     verbose=True)
    
    # Assert that messages were printed
    captured = capsys.readouterr()
    output = captured.out
    assert 'Coordinates: ' in output
    assert 'Cutout size:  [5 5]' in output
    assert re.search(r'Found \d+ footprint files.', output)
    assert re.search(r'Filtered to \d+ footprints for sequences: 44', output)
    assert re.search(r'Found \d+ matching cube files.', output)
    assert 'Generating cutouts in parallel with max threads: 8' in output
    check_output_file(cutout[0], ffi_type, [44])


@pytest.mark.parametrize('ffi_type', ['SPOC', 'TICA'])
def test_cube_cut_from_footprint_sequential(tmpdir, capsys, ffi_type):
    """Test that data cube is cut from FFI file using sequential processing"""
    cutout = cube_cut_from_footprint(coordinates='130 30', 
                                     cutout_size=5,
                                     product=ffi_type,
                                     output_dir=tmpdir,
                                     sequence=44,
                                     threads=1,
                                     verbose=True)
    
    # Assert that relevant message was printed
    captured = capsys.readouterr()
    assert 'Generating cutouts in sequence.' in captured.out
    check_output_file(cutout[0], ffi_type, [44])


def test_cube_cut_from_footprint_multi_sequence(tmpdir):
    """Test that a cube is created for each sequence when multiple are provided"""
    sequences = [1, 13]
    cutouts = cube_cut_from_footprint(coordinates='350 -80', 
                                      cutout_size=5,
                                      product='SPOC',
                                      output_dir=tmpdir,
                                      sequence=sequences)
    
    assert len(cutouts) == 2
    for path in cutouts:
        check_output_file(path, 'SPOC', sequences)


def test_cube_cut_from_footprint_all_sequences(tmpdir):
    """Test that cubes are created for all sequences that intersect the cutout"""
    # Create cutouts for all possible sequences
    coordinates = SkyCoord('350 -80', unit='deg')
    cutout_size = (5, 5)
    product = 'SPOC'
    cutouts = cube_cut_from_footprint(coordinates=coordinates, 
                                      cutout_size=cutout_size,
                                      product=product,
                                      output_dir=tmpdir)
    
    # Crossmatch to get sectors that contain cutout
    all_ffis = _get_s3_ffis('s3://tesscut-ops-footprints/tess_ffi_footprint_cache.json', as_table=True, load_polys=True)
    cone_results = _ra_dec_crossmatch(all_ffis, coordinates, cutout_size, 21)
    seq_list = _create_sequence_list(cone_results, product)
    sequences = [int(seq['sector']) for seq in seq_list]

    assert len(seq_list) == len(cutouts)
    for path in cutouts:
        check_output_file(path, 'SPOC', sequences)


def test_cube_cut_from_footprint_invalid_sequence():
    """Test that InvalidQueryError is raised if sequence does not have cube files"""
    err = 'No FFI cube files were found for sequences: -1'
    with pytest.raises(InvalidQueryError, match=err):
        cube_cut_from_footprint(coordinates='130 30', 
                                cutout_size=5,
                                sequence=-1)


def test_cube_cut_from_footprint_outside_coords():
    """Test that InvalidQueryError is raised if coordinates are not found in sequence"""
    err = 'The given coordinates were not found within the specified sequence(s).'
    with pytest.raises(InvalidQueryError, match=re.escape(err)):
        cube_cut_from_footprint(coordinates='130 30', 
                                cutout_size=5,
                                sequence=1)
