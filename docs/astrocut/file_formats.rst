:orphan:
   
*********************
Astrocut File Formats
*********************

FITS Cutout Files
=================

FITS files output by image cutout classes consist of a PrimaryHDU extension
and one or more ImageHDU extensions, each containing a single cutout.

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

========= ===================================================
Keyword   Value
========= ===================================================
SIMPLE    T (conforms to FITS standard)                     
BITPIX    8 (array data type)                               
NAXIS     0 (number of array dimensions)                    
EXTEND    Number of standard extensions                                                  
ORIGIN    STScI/MAST
DATE      File creation date                             
PROCVER   Software version                      
RA_OBJ    Center coordinate right ascension (deg)                         
DEC_OBJ   Center coordinate declination (deg)                             
CHECKSUM  HDU checksum
DATASUM   Data unit checksum
========= ===================================================

CUTOUT ImageHDU (Subsequent extension(s))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data in each CUTOUT extension is the cutout image. The header includes all of the
keywords from the extension that the cutout image was drawn from, with WCS keywords
updated to match the cutout image. Additionally the keyword ``ORIG_FLE`` has been added,
it contains the name of the file the cutout comes from.



ASDF Cutout Files
==================

Image Cutouts
^^^^^^^^^^^^^

ASDF and FITS image cutout files are output by the `~astrocut.ASDFCutout` class.
The output format (ASDF vs FITS) impacts the structure of the cutout file. Additionally, the ``lite`` parameter controls 
whether the cutout contains only essential data or all data and metadata from the original file.
The ``lite`` parameter has the following effects on the cutout content:

- ``lite=False`` (default): When ``lite=False``, the cutout includes all data and metadata from the original ASDF file, with all
  arrays (and higher-dimensional cubes) that match the spatial dimensions of the science data sliced to
  the cutout shape. The full tree structure and metadata from the original file are preserved.

- ``lite=True``: When ``lite=True``, only the essential cutout data and minimal metadata are included:
  the cutout world coordinate system (WCS) and the original filename. This mode produces significantly
  smaller files and makes for faster processing.

ASDF Format Output
-------------------

When writing to ASDF format, the cutout file structure depends on the value of the ``lite`` parameter.

**Lite ASDF Cutout Structure:**

.. code-block:: none

    asdf_cutout = {
        'history': [...],
        'roman': {
            'meta': {
                'wcs': <sliced gwcs object>,
                'orig_file': <original filename>
            },
            'data': <cutout data array>
        }
    }

**Full ASDF Cutout Structure:**

.. code-block:: none

    asdf_cutout = {
        'asdf_library': [...],
        'history': [...],
        'roman': {
            'meta': {
                'wcs': <sliced gwcs object>,
                'orig_file': <original filename>,
                <other metadata keys preserved from original file>
            },
            'data': <cutout data array>,
            'err': <cutout error array>,
            <other mission data arrays>
        }
    }

In both cases, the ``wcs`` is the original ``gwcs`` object from the input ASDF file that has been
sliced and adjusted to account for the cutout's position and shape.

FITS Format Output
-------------------

When writing to FITS format, the output structure differs from ASDF. The cutout data is stored in an ``ImageHDU`` extension, 
with WCS information encoded in standard FITS headers. With Python 3.11+ and ``stdatamodels>=4.1.0``, the ASDF metadata tree is embedded in the FITS file. 
When ``lite=True``, the embedded ASDF tree contains only the cutout WCS and original filename; when ``lite=False``, the full 
ASDF metadata is embedded.

Note that FITS output files only contain the cutout image data in the ``ImageHDU`` extension, and do not include any additional 
data arrays from the original ASDF file, even when ``lite=False``. This is a key difference from ASDF output.

**FITS Structure:**

.. code-block:: none

    PrimaryHDU (Extension 0)
    ├── ORIGIN: "STScI/MAST"
    ├── DATE: <file creation date>
    ├── PROCVER: <astrocut version>
    ├── RA_OBJ: <center RA in degrees>
    └── DEC_OBJ: <center declination in degrees>

    ImageHDU (Extension 1) - CUTOUT
    ├── Data: <cutout image array>
    ├── WCS keywords: <updated from original WCS>
    └── ORIG_FLE: <name of original ASDF file>

    BinTableHDU (Extension 2, Python 3.11+ and stdatamodels>=4.1.0) - ASDF
    └── Data: <ASDF metadata tree (lite or full) embedded as binary table>

Spectral Subsets
^^^^^^^^^^^^^^^^

ASDF spectral subsets are produced by the `~astrocut.RomanSpectralsubset` class.
The amount of information in each subset file is controlled by the ``lite`` parameter, which determines whether only essential data 
or all data and metadata from the original file are included in the subset. The ``lite`` parameter has the following effects on the subset content:

- ``lite=True`` (default): The subset data only includes the "wl", "flux", and "flux_error" arrays.

- ``lite=False``: The subset includes all data and metadata from the original ASDF file(s), with all arrays that match the 
  dimensions of the ``wl`` array being sliced to the subset shape if a ``wl_range`` was specified.
  The full tree structure and metadata from the original file(s) are preserved.

The subset output structure is also determined by the ``group_by`` parameter in the ``get_asdf_subsets`` and ``write_as_asdf`` methods.
This parameter has three options and controls how the subset data is organized and how many ASDF files are written.


Group by Source ID and Input File
----------------------------------

Setting ``group_by='source_file'`` writes one ASDF file per unique combination of input file and source ID. 
This means that if multiple source IDs are selected from the same input file, each will be written to a separate ASDF file. 
The output filename pattern for this grouping is: ``<input_stem>_subset_<source_id>[_lite].asdf``

**Lite Structure:**

.. code-block:: none

    asdf_subset = {
        'history': [...],
        'roman': {
            'meta': {
                'source_id': <source id>,
                <other metadata keys>
            },
            'data': {
                'wl': <subset wavelength array>,
                'flux': <subset flux array>,
                'flux_error': <subset flux error array>
            }
        }
    }

**Full Structure:**

.. code-block:: none

    asdf_subset = {
        'asdf_library': [...],
        'history': [...],
        'roman': {
            'meta': {
                'source_id': <source id>,
                <other metadata keys>
            },
            'data': {
                'wl': <subset wavelength array>,
                'flux': <subset flux array>,
                'flux_error': <subset flux error array>,
                <other data arrays and values>
            }
        }
    }

Group by Input File
---------------------

Setting ``group_by='file'`` writes one ASDF file per input spectral file, combining selected source IDs from that file into a single ASDF file.
The output filename pattern for this grouping is: ``<input_stem>_subset[_lite].asdf``

**Lite Structure:**

.. code-block:: none

    asdf_subset = {
        'history': [...],
        'roman': {
            'meta': {
                'source_ids': <list of source ids>,
                <other metadata keys>
            },
            'data': {
                <source_id_1>: {
                    'wl': <subset wavelength array>,
                    'flux': <subset flux array>,
                    'flux_error': <subset flux error array>
                },
                <source_id_2>: {
                    'wl': <subset wavelength array>,
                    'flux': <subset flux array>,
                    'flux_error': <subset flux error array>
                }
            }
        }
    }

**Full Structure:**

.. code-block:: none

    asdf_subset = {
        'asdf_library': [...],
        'history': [...],
        'roman': {
            'meta': {
                'source_ids': <list of source ids>,
                <other metadata keys>
            },
            'data': {
                <source_id_1>: {
                    'wl': <subset wavelength array>,
                    'flux': <subset flux array>,
                    'flux_error': <subset flux error array>,
                    <other data arrays and values>
                },
                <source_id_2>: {
                    'wl': <subset wavelength array>,
                    'flux': <subset flux array>,
                    'flux_error': <subset flux error array>,
                    <other data arrays and values>
                }
            }
        }
    }


Combined File for All Sources and Input Files
----------------------------------------------

Setting ``group_by='combined'`` writes one ASDF file containing all requested files and sources. 
The output filename pattern for this grouping is: ``combined_spectral_subset[_lite].asdf``

**Lite Structure:**

.. code-block:: none

    asdf_subset = {
        'history': [...],
        'roman': {
            'meta': {
                <input_file_1>: {
                    'source_ids': <list of source ids>,
                    <other metadata keys>
                },
                <input_file_2>: {
                    'source_ids': <list of source ids>,
                    <other metadata keys>
                }
            },
            'data': {
                <input_file_1>: {
                    <source_id_1>: {
                        'wl': <subset wavelength array>,
                        'flux': <subset flux array>,
                        'flux_error': <subset flux error array>
                    }
                },
                <input_file_2>: {
                    <source_id_2>: {
                        'wl': <subset wavelength array>,
                        'flux': <subset flux array>,
                        'flux_error': <subset flux error array>
                    }
                }
            }
        }
    }

**Full Structure:**

.. code-block:: none

    asdf_subset = {
        'asdf_library_combined': {
            <input_file_1>: {...},
            <input_file_2>: {...}
        },
        'history': [...],
        'history_combined': {
            <input_file_1>: {...},
            <input_file_2>: {...}
        },
        'roman': {
            'meta': {
                <input_file_1>: {
                    'source_ids': <list of source ids>,
                    <other metadata keys>
                },
                <input_file_2>: {
                    'source_ids': <list of source ids>,
                    <other metadata keys>
                }
            },
            'data': {
                <input_file_1>: {
                    <source_id_1>: {
                        'wl': <subset wavelength array>,
                        'flux': <subset flux array>,
                        'flux_error': <subset flux error array>,
                        <other data arrays and values>
                    }
                },
                <input_file_2>: {
                    <source_id_2>: {
                        'wl': <subset wavelength array>,
                        'flux': <subset flux array>,
                        'flux_error': <subset flux error array>,
                        <other data arrays and values>
                    }
                }
            }
        }
    }


Cube Files
==========

See the `TESS Science Data Products Description Document <https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=17>`__
for detailed information on the TESS full-frame image file format.


PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^

The Primary Header of the TESS Mission, SPOC cube FITS file is the same as that from
an individual FFI with the following exceptions:

========= ===================================================
Keyword   Value
========= ===================================================
 ORIGIN   STScI/MAST
 DATE     Date the cube was created
 CAMERA   From the ImageHDU (EXT 1) of an FFI
 CCD      From the ImageHDU (EXT 1) of an FFI
 SECTOR   The TESS observing Sector, passed by the user
 DATE-OBS From the ImageHDU (EXT 1) of the Sector's first FFI
 DATE-END From the ImageHDU (EXT 1) of the Sector's last FFI
 TSTART   From the ImageHDU (EXT 1) of the Sector's first FFI
 TSTOP    From the ImageHDU (EXT 1) of the Sector's last FFI
========= ===================================================


ImageHDU (Extension 1)
^^^^^^^^^^^^^^^^^^^^^^

The ImageHDU extension contains the TESS FFI data cube.
It is 4 dimensional, with two spatial dimensions, time, data and
error flux values. Pixel values are 32 bit floats.
The cube dimensions are ordered in the FITS format as follows:

========= ===================================================
Keyword   Value
========= ===================================================
NAXIS     4 (number of array dimensions)                    
NAXIS1    2 (data value, error value)
NAXIS2    Total number of FFIs
NAXIS3    Length of first array dimension (NAXIS1 from FFIs)
NAXIS4    Length of second array dimension (NAXIS2 from FFIs)
========= ===================================================


BinTableHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^

The BinTableHDU extension contains a table that 
holds all of the image extension header keywords from the individual FFIs. There 
is one column for each keyword plus one additional column called "FFI_FILE" that 
contains FFI filename for each row. Each column name keyword also has an entry in the 
Image extension header, with the value being the keyword value from the FFI header.
This last column allows the FFI Image extension headers to be recreated completely if desired.


Target Pixel Files
==================

The Astrocut target pixel file (TPF) format conforms as closely as possible to the
TESS Mission TPFs. See the `TESS Science Data Products Description Document <https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf#page=23>`__
for detailed information on the TESS Mission TPF format, here it is
described how Astrocut TPFs differ from Mission pipeline TPFs.

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Primary Header of an Astrocut TPF is the same as that from
a Mission TPF with the following exceptions:

========= ====================================================
Keyword   Value
========= ====================================================
ORIGIN    STScI/MAST
CREATOR   astrocut
PROCVER   Astrocut version
SECTOR    Depends on this value having been filled in the cube

 **Mission pipeline header values Astrocut cannot populate**
--------------------------------------------------------------
OBJECT    ""
TCID      0
PXTABLE   0
PMRA      0.0
PMDEC     0.0
PMTOTAL   0.0
TESSMAG   0.0
TEFF      0.0
LOGG      0.0
MH        0.0
RADIUS    0.0
TICVER    0
TICID     None
========= ====================================================

PIXELS BinTableHDU (Extension 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Astrocut PIXELS BinTableHDU comprises the same columns as those included in
the Mission pipeline TPFs, with one addition: an extra column, ``FFI_FILE``, contains
the name of the FFI file that the row's pixels come from.

While all of the columns present in Mission pipeline TPFs are present in cutouts created
from SPOC cubes, they do not all contain data. The columns that are empty in Astrocut TPFs are:

============ ====================================================
Column       Value
============ ====================================================
CADENCENO    0 filled array in cutout shape
RAW_CNTS     -1 filled array in cutout shape
FLUX_BKG     0 filled array in cutout shape
FLUX_BKG_ERR 0 filled array in cutout shape
POS_CORR1    0
POS_CORR2    0
============ ====================================================

The ``TIME`` column is formed by taking the average of the ``TSTART`` and ``TSTOP`` values
from the corresponding FFI for each row. The ``QUALITY`` column is taken from the ``DQUALITY``
image keyword in the individual SPOC FFI files.

Three keywords have also been added to the PIXELS extension header to give additional information
about the cutout world coordinate system (WCS). TESS FFIs are large and therefore are described
by WCS objects that have many non-linear terms. Astrocut creates a new simpler (linear) WCS
object from the matched set of cutout pixel coordinates and sky coordinates (from the FFI WCS).
This linear WCS object will generally work very well, however at larger cutout sizes (100-200
pixels per side and above) the linear WCS fit will start to be noticeably incorrect at the edges
of the cutout. The extra keywords allow the user to determine if the linear WCS is accurate enough
for their purpose, and to retrieve the original WCS with distortion coefficients if it is needed.


+---------+----------------------------------------------------------------+
| Keyword |  Value                                                         |
+=========+================================================================+
| WCS_FFI | | The name of the FFI file used to build the original WCS      |
|         | | from which the cutout and cutout WCS were calculated.        |
+---------+----------------------------------------------------------------+
| WCS_MSEP| | The maximum separation in degrees between the cutout's       |
|         | | linear WCS and the FFI's full WCS.                           |
+---------+----------------------------------------------------------------+
| WCS_SIG | | The error in the cutout's linear WCS, calculated as          |
|         | | ``sqrt((dist(Po_ij, Pl_ij)^2)`` where ``dist(Po_ij, Pl_ij)`` |
|         | | is the angular distance in degrees between the sky position  |
|         | | of of pixel i,j in the original full WCS and the new linear  |
|         | | WCS.                                                         |
+---------+----------------------------------------------------------------+


APERTURE ImageHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The APERTURE ImageHDU extension is similar to that of Mission pipeline TPFs, but contains
slightly different data. For Mission pipeline files, the aperture image gives information about
each pixel, whether it was collected and whether it was used in calculating e.g., the background flux.
Because Astrocut does not do any of the more complex calculations used in the Mission pipeline, each pixel in the
aperture image will either be 1 (pixel was collected and contains data in the cutout) or 0
(pixel is off the edge of the detector and contains no data in the cutout).


Cosmic Ray Binary Table Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This extension is not present in Astrocut TPFs, although it is a part of the Mission pipeline TPFs.


Path Focused Target Pixel Files
===============================

When the `~astrocut.center_on_path` function is used to create cutout TPFs
where the individual image cutouts move along a path in time and space, the TPF format has to be
adjusted accordingly. It still conforms as closely as possible to the TESS Mission pipeline TPF
file format, but differs in several crucial ways. The `~astrocut.center_on_path` function works
on Astrocut TPFs, so that is the baseline file format. Only the differences
between path focused Astrocut TPFs and regular Astrocut TPFs are described here (see `Target Pixel Files`_ for
regular Astrocut TPF format).

PRIMARY PrimaryHDU (Extension 0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional or updated keywords:

========= =======================================================
Keyword   Value
========= =======================================================
DATE      Set the the time the path focused cutout was performed
OBJECT    Moving target object name/identifier, only present if
          set by the user
========= =======================================================

Removed keywords:

========= =======================================================
Keyword   Reason
========= =======================================================
RA_OBJ    Cutout is no longer centered on a sky position
DEC_OBJ   Cutout is no longer centered on a sky position
========= =======================================================


PIXELS BinTableHDU (Extension 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional columns:

============ ========================================================
Column       Value
============ ========================================================
TGT_X        X position of the target in the cutout array at row time
TGT_Y        Y position of the target in the cutout array at row time
TGT_RA       Right ascension (deg) of the target at row time
TGT_DEC      Declination (deg) of the target at row time
============ ========================================================

No world coordinate system (WCS) information is present, since it is no
longer common across all cutout images.


APERTURE ImageHDU (Extension 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The APERTURE extension may or may not be present in a path focussed TPF, to be present
the user must have passed an FFI WCS object into the `~astrocut.center_on_path` function.

The APERTURE ImageHDU extension of path focussed TPFs is very different from other
TESS TPFs. The aperture image, instead of being the size and shape of an individual cutout,
is the size of the full FFI image the cutouts were drawn from. All pixels used in any
individual cutout are marked with 1, while the rest of the pixels are 0, so the entire
trajectory of the cutout path is captured. Additionally the WCS information in the header
is the WCS for the original FFI, including all distortion coefficients. This can be
used in combination with the TGT_RA/DEC and TGT_X/Y columns to trace the path of the
target across the FFI footprint and calculate the WCS object for individual cutout images
if necessary.
