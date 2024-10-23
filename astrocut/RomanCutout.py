import asdf

from .ASDFCutout import ASDFCutout

class RomanCutout(ASDFCutout):

    def _read_from_asdf(self):
        # get the 2d image data
        with asdf.open(self.input_file) as af:
            self.data = af['roman']['data']
            self.gwcs = af['roman']['meta']['wcs']