from asdf_cutouts import asdf_cut
from glob import glob
from astropy.coordinates import SkyCoord

# look for the level 2 test files on disk
files = (glob("/grp/roman/TEST_DATA/23Q4_B11/aligntest/*_cal.asdf"))

# grab one for demo-ing
file = files[4]
print('Roman level 2 image file:', file)

# create an Astropy SkyCoord object; provided coordinate is an RA, Dec position within the image, for this set of test images
coordinates = SkyCoord("80.15189743 29.74561219", unit='deg')

cutout = asdf_cut(file, coordinates.ra, coordinates.dec, cutout_size=200, output_file="roman-demo.fits")