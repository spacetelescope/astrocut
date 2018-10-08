import numpy as np
import astropy.units as u

from astropy.io import fits
from astropy.table import Table,Column
from astropy.coordinates import SkyCoord
from astropy import wcs

from time import time
from copy import deepcopy

import os


def get_cube_wcs(table_header,table_row):
    """
    Takes the header and one entry from the cube table of data and returns 
    a WCS object that encalpsulates the given WCS information.
    """
    
    wcs_header = fits.header.Header()

    for header_key, header_val in table_header.items():
        if not 'TTYPE' in header_key:
            continue

        colNum = int(header_key[5:])-1
        if 'NAXIS' in header_val:
            #print(header_key, header_val)
            wcs_header[header_val] =  int(table_row[colNum])
        else:
            wcs_header[header_val] =  table_row[colNum]

    #for key in wcs_header:
    #   print(key,wcs_header[key])
       
    return wcs.WCS(wcs_header)


def get_cutout_limits(center_coord, cutout_size, cube_wcs):
    """
    Takes the center coordinates, cutout size, and the wcs from
    which the cutout is being taken and returns the x and y pixel limits
    for the cutout.

    cutout_size : array
         [nx,ny] in with ints (pixels) or astropy quantities
    """

    center_pixel = center_coord.to_pixel(cube_wcs)

    lims = np.zeros((2,2),dtype=int)

    for axis, size in enumerate(cutout_size):
        
        if not isinstance(size, u.Quantity): # assume pixels
            dim = size/2
        elif size.unit == u.pixel: # also pixels
            dim = size.value/2
        elif size.unit.physical_type == 'angle':
            pixel_scale = u.Quantity(wcs.utils.proj_plane_pixel_scales(cube_wcs)[axis], cube_wcs.wcs.cunit[axis])
            dim = (size / pixel_scale).decompose()/2

        lims[axis,0] = int(np.round(center_pixel[axis] - dim))
        lims[axis,1] = int(np.round(center_pixel[axis] + dim))
    
    return lims


#def get_cutout_wcs(cutout_lims, cube_wcs):
#    """
#    Get cutout wcs object, with updated NAXIS keywords and crpix. 
#    """

#    cutout_wcs = deepcopy(cube_wcs)
    
#    cutout_wcs.wcs.crpix -= cutout_lims[:,0]
    
#    cutout_wcs._naxis = [cutout_lims[0,1]-cutout_lims[0,0],
#                         cutout_lims[1,1]-cutout_lims[1,0]]

#    return cutout_wcs


def get_cutout_wcs(cutout_coord, cutout_lims, cube_wcs):
    """
    Get cutout wcs header keywords, with updated NAXIS keywords and crpix. 
    Plus transformation back to original location on CCD.

    Returns as dictionary of {<kwd format string>: [value, desc]} pairs.
    TODO: Improve description
    """
    cube_wcs_header = cube_wcs.to_header()

    cutout_wcs_dict = dict()

    ## Cutout array keywords ##
    
    cutout_wcs_dict["WCAX{}"] = [cube_wcs_header['WCSAXES'],"number of WCS axes"]
    # TODO: make sure this is 2 (not making provisions for >2 at this point)

    cutout_wcs_dict["1CTYP{}"] = [cube_wcs_header["CTYPE1"],"right ascension coordinate type"]
    cutout_wcs_dict["2CTYP{}"] = [cube_wcs_header["CTYPE2"],"declination coordinate type"]
    
    # TODO: FILL THESE (pixel vales of COORD on cutout)
    orig_pix = cube_wcs.all_world2pix(cutout_coord.ra.deg, cutout_coord.dec.deg, 0)
    cutout_wcs_dict["1CTYP{}"] = [float(orig_pix[0]) - cutout_lims[0,0],
                                  "[pixel] reference pixel along image axis 1"]
    cutout_wcs_dict["2CTYP{}"] = [float(orig_pix[1]) - cutout_lims[0,0],
                                  "[pixel] reference pixel along image axis 2"]
    
    cutout_wcs_dict["1CRVAL{}"] = [cutout_coord.ra.deg, "[deg] right ascension at reference pixel"]
    cutout_wcs_dict["2CRVAL{}"] = [cutout_coord.dec.deg, "[deg] declination at reference pixel"]
    
    cunits = cube_wcs.wcs.cunit
    cutout_wcs_dict["1CUNI{}"] = [str(cunits[0]), "physical unit in column dimension"]
    cutout_wcs_dict["2CUNI{}"] = [str(cunits[1]), "physical unit in row dimension"]

    px_scales = wcs.utils.proj_plane_pixel_scales(cube_wcs)
    cutout_wcs_dict["1CDLT{}"] = [px_scales[0], "[deg] pixel scale in RA dimension"]
    cutout_wcs_dict["2CDLT{}"] = [px_scales[1], "[deg] pixel scale in DEC dimension"]

    # TODO: THIS IS FILLER, HAVE TO FIGURE OUT HOW TO DO THE TRANSFORMATION FOR REAL
    cutout_wcs_dict["11PC{}"] = [cube_wcs_header["PC1_1"],"linear transformation matrix element cos(th)"]
    cutout_wcs_dict["12PC{}"] = [cube_wcs_header["PC1_2"],"linear transformation matrix element -sin(th)"]
    cutout_wcs_dict["21PC{}"] = [cube_wcs_header["PC2_1"],"linear transformation matrix element sin(th)"]
    cutout_wcs_dict["22PC{}"] = [cube_wcs_header["PC2_2"],"linear transformation matrix element cos(th)"]

    ## Physical keywords ##
    
    cutout_wcs_dict["WCSN{}P"] = ["PHYSICAL", "table column WCS name"]
    cutout_wcs_dict["WCAX{}P"] = [2,"table column physical WCS dimensions"]
    
    cutout_wcs_dict["1CTY{}P"] = ["RAWX","table column physical WCS axis 1 type, CCD col"]
    cutout_wcs_dict["2CTY{}P"] = ["RAWY","table column physical WCS axis 2 type, CCD row"]
    
    cutout_wcs_dict["1CUN{}P"] = ["PIXEL","table column physical WCS axis 1 unit"]
    cutout_wcs_dict["2CUN{}P"] = ["PIXEL","table column physical WCS axis 2 unit"]
    
    cutout_wcs_dict["1CRV{}P"] = [int(cube_wcs_header["CRPIX1"]) - cutout_lims[0,0],
                                  "table column physical WCS ax 1 ref value"]
    cutout_wcs_dict["2CRV{}P"] = [int(cube_wcs_header["CRPIX1"]) - cutout_lims[1,0],
                                  "table column physical WCS ax 2 ref value"]

    # TODO: can we calculate these? or are they fixed?
    cutout_wcs_dict["1CDL{}P"] = [1.0,"table column physical WCS a1 step"]    
    cutout_wcs_dict["2CDL{}P"] = [1.0,"table column physical WCS a2 step"]
    
    cutout_wcs_dict["1CRP{}P"] = [1,"table column physical WCS a1 reference"]
    cutout_wcs_dict["2CRP{}P"] = [1,"table column physical WCS a2 reference"]

    return cutout_wcs_dict
    

def get_cutout(cutout_lims, transposed_cube, verbose=True):
    """
    Making a cutout from an image/uncertainty cube that has been transposed 
    to have time on the longest axis.

    Returns the untransposed image cutout and uncertainty cutout.
    """

    # These limits are not guarenteed to be within the image footprint
    xmin,xmax = cutout_lims[1]
    ymin,ymax = cutout_lims[0]

    # Get the image array limits
    xmax_cube,ymax_cube,_,_ = transposed_cube.shape

    # Adjust limits and figuring out the padding
    padding = np.zeros((3,2),dtype=int)
    if xmin < 0:
        padding[1,0] = -xmin
        xmin = 0
    if ymin < 0:
        padding[2,0] = -ymin
        ymin = 0
    if xmax > xmax_cube:
        padding[1,1] = xmax - xmax_cube
        xmax = xmax_cube
    if ymax > ymax_cube:
        padding[2,1] = ymax - ymax_cube
        ymax = ymax_cube       
        
    # Doing the cutout
    cutout = transposed_cube[xmin:xmax,ymin:ymax,:,:]
    
    img_cutout = cutout[:,:,:,0].transpose((2,0,1))
    uncert_cutout = cutout[:,:,:,1].transpose((2,0,1))
    
    # Making the aperture array
    aperture = np.ones((xmax-xmin, ymax-ymin))

    # Adding padding to the cutouts so that it's the expected size
    if padding.any(): # only do if we need to pad
        img_cutout = np.pad(img_cutout, padding, 'constant', constant_values=np.nan)
        uncert_cutout = np.pad(uncert_cutout, padding, 'constant', constant_values=np.nan)
        aperture = np.pad(aperture, padding[1:], 'constant', constant_values=0)

    if verbose:
        print("Image cutout cube shape: {}".format(img_cutout.shape))
        print("Uncertainty cutout cube shape: {}".format(uncert_cutout.shape))
    
    return img_cutout, uncert_cutout, aperture


def update_primary_header(primary_header, coordinates):

    # Adding cutout specific headers
    primary_header['RA_OBJ'] = (coordinates.ra.deg,'[deg] right ascension')
    primary_header['DEC_OBJ'] = (coordinates.dec.deg,'[deg] declination')
    
    # These are all the things in the TESS pipeline tpfs about the object that we can't fill
    primary_header['OBJECT'] = ("",'string version of target id ')
    primary_header['TCID'] = (0,'unique tess target identifier')
    primary_header['PXTABLE'] = (0,'pixel table id') 
    primary_header['PMRA'] = (0.0,'[mas/yr] RA proper motion') 
    primary_header['PMDEC'] = (0.0,'[mas/yr] Dec proper motion') 
    primary_header['PMTOTAL'] = (0.0,'[mas/yr] total proper motion') 
    primary_header['TESSMAG'] = (0.0,'[mag] TESS magnitude') 
    primary_header['TEFF'] = (0.0,'[K] Effective temperature') 
    primary_header['LOGG'] = (0.0,'[cm/s2] log10 surface gravity') 
    primary_header['MH'] =(0.0,'[log10([M/H])] metallicity') 
    primary_header['RADIUS'] = (0.0,'[solar radii] stellar radius')
    primary_header['TICVER'] = (0,'TICVER')



#def add_column_wcs(header, colnums, wcs_info):
#    """
#    Take WCS info in wcs_info and add it to the header as 
#    """
#    
#    wcs_keywords = {'CTYPE1':'1CTYP{}',
#                    'CTYPE2':'2CTYP{}',
#                    'CRPIX1':'1CRPX{}',
#                    'CRPIX2':'2CRPX{}',
#                    'CRVAL1':'1CRVL{}',
#                    'CRVAL2':'2CRVL{}',
#                    'CUNIT1':'1CUNI{}',
#                    'CUNIT2':'2CUNI{}',
#                    'CDELT1':'1CDLT{}',
#                    'CDELT2':'2CDLT{}',
#                    'PC1_1':'11PC{}',
#                    'PC1_2':'12PC{}',
#                    'PC2_1':'21PC{}',
#                    'PC2_2':'22PC{}'}
#
#
#    for col in colnums:
#        for kw in wcs_keywords:
#            if kw not in wcs_info.keys():
#                continue ## TODO: Something better than this?
#            header[wcs_keywords[kw].format(col)] = wcs_info[kw]


def add_column_wcs(table_header, wcs_dict):


    #cur_col = -1 # holds the current column number
    wcs_col = False # says if column is one that requires wcs info
    
    for kwd in table_header:

        # Adding header descriptions for the table keywords
        if kwd[:-1] == "TTYPE":
            table_header.comments[kwd] = "column name"
            if table_header[kwd] == "CADENCENO":
                table_header.comments[kwd] += ": Row counter, not true cadenceno"
        if kwd[:-1] == "TFORM":
            table_header.comments[kwd] = "column format"
        if kwd[:-1] == "TUNIT":
            table_header.comments[kwd] = "unit"
        if kwd[:-1] == "TDISP":
            table_header.comments[kwd] = "display format"
        if kwd[:-1] == "TDIM":
            table_header.comments[kwd] = "multi-dimensional array spec"
            wcs_col = True # if column holds 2D array need to add wcs info
        if kwd[:-1] == "TNULL":
            table_header.comments[kwd] = "null value"

        # Adding wcs info if necessary
        if (kwd[:-1] == "TTYPE") and wcs_col:
            wcs_col = False # reset
            for wcs_key,(val,com) in wcs_dict.items():
                #print((wcs_key.format(kwd[-1]),val,com))
                table_header.insert(kwd,(wcs_key.format(int(kwd[-1])-1),val,com))

        # Removing keywords we don't want in the table header
            
        
    

            
def apply_header_inherit(hdu_list):
    """Astropy fits functionality does not appy the INHERIT keyword, so we have to do it manually"""
    
    primary_header = hdu_list[0].header

    reserved_kwds = ["COMMENT","SIMPLE","BITPIX", "EXTEND", "NEXTEND"]

    for hdu in hdu_list[1:]:
        if hdu.header.get("INHERIT", False):
            for kwd in primary_header:
                if (kwd not in hdu.header) and (kwd not in reserved_kwds):
                    hdu.header[kwd] = (primary_header[kwd], primary_header.comments[kwd])
            

def build_tpf(cube_fits, img_cube, uncert_cube, cutout_wcs_dict, aperture, coordinates, verbose=True):
    """
    Building the target pixel file.
    """

    # The primary hdu is just the main header, which is the same
    # as the one on the cube file
    primary_hdu = cube_fits[0]
    update_primary_header(primary_hdu.header, coordinates)

    cols = list()

    # Adding the Time relates columns
    cols.append(fits.Column(name='TIME', format='D', unit='BJD - 2457000, days', disp='D14.7',
                            array=(cube_fits[2].columns['TSTART'].array + cube_fits[2].columns['TSTOP'].array)/2))

    cols.append(fits.Column(name='TIMECORR', format='E', unit='d', disp='E14.7',
                            array=cube_fits[2].columns['BARYCORR'].array))

    cols.append(fits.Column(name='CADENCENO', format='J', disp='I10', array=np.arange(1, len(img_cube) + 1)))
    
    # Adding the cutouts
    tform = str(img_cube[0].size) + "E"
    dims = str(img_cube[0].shape)
    empty_arr = np.zeros(img_cube.shape)

    if verbose:
        print("TFORM: {}".format(tform))
        print("DIMS: {}".format(dims))
        print("Array shape: {}".format(empty_arr.shape))
    
    cols.append(fits.Column(name='RAW_CNTS', format=tform.replace('E','J'), unit='count', dim=dims, disp='I8',
                            array=empty_arr-1, null=-1)) 
    cols.append(fits.Column(name='FLUX', format=tform, dim=dims, unit='e-/s', disp='E14.7', array=img_cube))
    cols.append(fits.Column(name='FLUX_ERR', format=tform, dim=dims, unit='e-/s', disp='E14.7', array=uncert_cube)) 
   
    # Adding the background info (zeros b.c we don't have this info)
    cols.append(fits.Column(name='FLUX_BKG', format=tform, dim=dims, unit='e-/s', disp='E14.7',array=empty_arr))
    cols.append(fits.Column(name='FLUX_BKG_ERR', format=tform, dim=dims, unit='e-/s', disp='E14.7',array=empty_arr))

    # Adding the quality flags
    cols.append(fits.Column(name='QUALITY', format='J', disp='B16.16', array=cube_fits[2].columns['DQUALITY'].array))

    # Adding the position correction info (zeros b.c we don't have this info)
    cols.append(fits.Column(name='POS_CORR1', format='E', unit='pixel', disp='E14.7',array=empty_arr[:,0,0]))
    cols.append(fits.Column(name='POS_CORR2', format='E', unit='pixel', disp='E14.7',array=empty_arr[:,0,0]))

    # Adding the FFI_FILE column (not in the pipeline tpfs)
    cols.append(fits.Column(name='FFI_FILE', format='38A', unit='pixel',array=cube_fits[2].columns['FFI_FILE'].array))
        
    # making the table HDU
    table_hdu = fits.BinTableHDU.from_columns(cols)

    # Have to add the comment to the CADENCENO column Header manually
    # (This is needed b/c the CADENCENO is not really the cadence number)
    #table_hdu.header.comments['TTYPE3'] = "Row counter, not true cadence number"
    
    table_hdu.header['EXTNAME'] = 'PIXELS'
    table_hdu.header['INHERIT'] = True
    
    # Adding the wcs keywords to the columns and removing from the header
    #wcs_header = cutout_wcs.to_header()
    add_column_wcs(table_hdu.header, cutout_wcs_dict) 
    #for kword in wcs_header:
    #    table_hdu.header.remove(kword, ignore_missing=True)
    # TODO: what kwds to remove from generic table header??

    # Building the aperture HDU
    aperture_hdu = fits.ImageHDU(data=aperture)
    aperture_hdu.header['EXTNAME'] = 'APERTURE'
    aperture_hdu.header['INHERIT'] = True
    
    cutout_hdu_list = fits.HDUList([primary_hdu,table_hdu, aperture_hdu])

    apply_header_inherit(cutout_hdu_list)

    return cutout_hdu_list



def cube_cut(cube_file, coordinates, cutout_size, target_pixel_file=None, output_path=".", verbose=None):
    """
    Takes a cube file (as created by ~astrocut.make_cube), 
    and makes a cutout stak of the given size around the given coordinates.

    Parameters
    ----------
    cube_file : str
        The cube file containing all the images to be cutout.  
        Must be in the format returned by ~astrocut.make_cube.
    coordinates : str or `astropy.coordinates` object
        The position around which to cutout. It may be specified as a
        string or as the appropriate `astropy.coordinates` object.
    cutout_size : int, array-like, `~astropy.units.Quantity`
        TODO: Is there a default size that makes sense?
        The size of the cutout array. If ``size``
        is a scalar number or a scalar `~astropy.units.Quantity`,
        then a square cutout of ``size`` will be created.  If
        ``size`` has two elements, they should be in ``(ny, nx)``
        order.  Scalar numbers in ``size`` are assumed to be in
        units of pixels. `~astropy.units.Quantity` objects must be in pixel or
        angular units.
    target_pixel_file : str
        Optional. The name for the output target pixel file. 
        If no name is supplied, the file will be named: 
        <cube_file>_<ra>_<dec>_<cutout_size>_astrocut.fits
    output_path : str
        Optional. The path where the output file is saved. 
        The current directory is default.
    verbose : bool
        Optional. If true intermediate information is printed. 

    Returns
    -------
    response: string or None
        If successfull, returns the path to the target pixel file, 
        if unsuccessfull returns None.
    """

    if verbose:
        startTime = time()

    cube = fits.open(cube_file) # TODO: add checking

    # Get the WCS and figure out which pixels are in the cutout
    wcsInd = int(len(cube[2].data)/2) # using the middle file for wcs info
    cubeWcs = get_cube_wcs(cube[2].header, cube[2].data[wcsInd])

    if not isinstance(coordinates, SkyCoord):
        coordinates = SkyCoord.from_name(coordinates) # TODO: more checking here

    if verbose:
        print("Cutout center coordinate:",coordinates.ra.deg,coordinates.dec.deg)


    # making size into an array [ny, nx] TODO: MAKE SURE I AM USING THEM AS NY,NX
    cutout_size = np.atleast_1d(cutout_size)
    if len(cutout_size) == 1:
        cutout_size = np.repeat(cutout_size, 2)

    if len(cutout_size) > 2:
        print("To many dimensions in cutout size, only the first two will be used") # TODO: Make this into a warning
       
    # Get cutout limits
    cutout_lims = get_cutout_limits(coordinates, cutout_size, cubeWcs)

    if verbose:
        print("xmin,xmax:",cutout_lims[1])
        print("ymin,ymax:",cutout_lims[0])

    # Make the cutout
    img_cutout, uncert_cutout, aperture = get_cutout(cutout_lims, cube[1].data)

    # Get cutout wcs info
    cutout_wcs_dict = get_cutout_wcs(coordinates, cutout_lims, cubeWcs)
    
    # Build the TPF
    tpf_object = build_tpf(cube, img_cutout, uncert_cutout, cutout_wcs_dict, aperture, coordinates)

    if verbose:
        writeTime = time()

    if not target_pixel_file:
        # TODO: also strip off excess path from cube file
        _, flename = os.path.split(cube_file)
        target_pixel_file = output_path + "/"
        target_pixel_file += "{}_{}_{}_{}x{}_astrocut.fits".format(flename.rstrip('.fits').rstrip("-cube"),
                                                                  coordinates.ra.value, coordinates.dec.value,
                                                                  cutout_lims[0,1]-cutout_lims[0,0],
                                                                  cutout_lims[1,1]-cutout_lims[1,0])
        
    if verbose:
        print("Target pixel file:",target_pixel_file)
        
    # Write the TPF
    tpf_object.writeto(target_pixel_file, overwrite=True)

    # Close the cube file
    cube.close()

    if verbose:
        print("Write time: {:.2} sec".format(time()-writeTime))
        print("Total time: {:.2} sec".format(time()-startTime))

    return target_pixel_file

