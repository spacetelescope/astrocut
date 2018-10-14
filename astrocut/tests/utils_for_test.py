import os
import numpy as np

from astropy.io import fits
from astropy.table import Table


def add_keywords(hdu, extname, time_increment):
    """
    Add a bunch of required keywords (mostly fake values)
    """
    
    hdu.header['extname'] = 'CAMERA.CCD 1.1 cal'
    hdu.header['camera'] = 1
    hdu.header['ccd'] = 1
    hdu.header['tstart'] = float(time_increment)
    hdu.header['tstop'] = float(time_increment+1)
    hdu.header['date-obs'] = '2019-05-11T00:08:26.816Z'
    hdu.header['date-end'] = '2019-05-11T00:38:26.816Z'
    hdu.header['barycorr'] = 5.0085597E-03
    hdu.header['dquality'] = 0
    
    # WCS keywords just copied from example
    hdu.header['RADESYS'] = 'ICRS    '
    hdu.header['EQUINOX'] =               2000.0
    hdu.header['WCSAXES'] =                    2
    hdu.header['CTYPE1']  = 'RA---TAN-SIP'
    hdu.header['CTYPE2']  = 'DEC--TAN-SIP'
    hdu.header['CRVAL1']  = 250.3497414839765200
    hdu.header['CRVAL2']  =   2.2809255996090630
    hdu.header['CRPIX1']  =               1045.0
    hdu.header['CRPIX2']  =               1001.0
    hdu.header['CD1_1']   =   -0.005564478186178
    hdu.header['CD1_2']   =   -0.001042099258152
    hdu.header['CD2_1']   =    0.001181441465850
    hdu.header['CD2_2']   =   -0.005590816683583
    hdu.header['A_ORDER'] =                    2
    hdu.header['B_ORDER'] =                    2
    hdu.header['A_2_0']   =   2.024511892340E-05
    hdu.header['A_0_2']   =   3.317603337918E-06
    hdu.header['A_1_1']   =  1.73456334971071E-5
    hdu.header['B_2_0']   =   3.331330003472E-06
    hdu.header['B_0_2']   = 2.042474824825892E-5
    hdu.header['B_1_1']   = 1.714767108041439E-5
    hdu.header['AP_ORDER']=                    2
    hdu.header['BP_ORDER']=                    2
    hdu.header['AP_1_0']  = 9.047002963896363E-4
    hdu.header['AP_0_1']  = 6.276607155847164E-4
    hdu.header['AP_2_0']  =  -2.023482905861E-05
    hdu.header['AP_0_2']  =  -3.332285841011E-06
    hdu.header['AP_1_1']  =  -1.731636633824E-05
    hdu.header['BP_1_0']  = 6.279608820532116E-4
    hdu.header['BP_0_1']  = 9.112228860848081E-4
    hdu.header['BP_2_0']  =  -3.343918167224E-06
    hdu.header['BP_0_2']  =  -2.041598249021E-05
    hdu.header['BP_1_1']  =  -1.711876336719E-05
    hdu.header['A_DMAX']  =    44.72893589844534
    hdu.header['B_DMAX']  =    44.62692873032506


def create_test_ffis(img_size, num_images, basename='make_cube-test{}.fits'):
    """
    Create test fits files

    Write negative values for data array and positive values for error array,
    with unique values for all the pixels. The header keywords are just
    populated to make things run but are not tested.
    """

    img = np.arange(img_size*img_size, dtype=np.float32).reshape((img_size,img_size))
    file_list = []
    
    for i in range(num_images):
        
        file_list.append(basename.format(i))

        primary_hdu = fits.PrimaryHDU()
        
        hdu = fits.ImageHDU(-img)
        add_keywords(hdu, 'CAMERA.CCD 1.1 cal', i)
        
        ehdu = fits.ImageHDU(img)
        add_keywords(ehdu, 'CAMERA.CCD 1.1 uncert', i)
        
        hdulist = fits.HDUList([primary_hdu, hdu, ehdu])
        hdulist.writeto(file_list[-1], overwrite=True, checksum=True)
        
        img = img + img_size*img_size

    return file_list