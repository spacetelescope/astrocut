import asdf

from .ASDFCutout import ASDFCutout

class RomanCutout(ASDFCutout):

    def _read_from_asdf(self, input_file):
        # get the 2d image data
        with asdf.open(input_file) as af:
            data = af['roman']['data']
            gwcs = af['roman']['meta']['wcs']

        return (data, gwcs)