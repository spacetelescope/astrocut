#################################################################################
#
# Licensed under a 3-clause BSD style license
#           - see https://github.com/astropy/astropy/blob/master/LICENSE.rst
#
# wcs fitting functionality
#          by Clare Shanahan (shannnnnyyy @github)
#
# Will be added to Astropy (PR: https://github.com/astropy/astropy/pull/7884)
#
# Astropy version is used preferenetially, this is supplied prior to the
# addition of this code to Astropy, and for users using older versions of Astropy
#
#################################################################################

# flake8: noqa

import numpy as np

from astropy import units as u
from astropy.wcs.utils import celestial_frame_to_wcs

def _linear_transformation_fit(params, lon, lat, x, y, w_obj):  # pragma: no cover
    """ Objective function for fitting linear terms."""
    pc = params[0:4]
    crpix = params[4:6]

    w_obj.wcs.pc = ((pc[0],pc[1]),(pc[2],pc[3]))
    w_obj.wcs.crpix = crpix
    lon2, lat2 = w_obj.wcs_pix2world(x,y,1)
    
    resids = np.concatenate((lon-lon2, lat-lat2))
    resids[resids > 180] = 360 - resids[resids > 180]
    resids[resids < -180] = 360	+ resids[resids < -180]

    return resids


def _sip_fit(params, lon, lat, u, v, w_obj, a_order, b_order, a_coeff_names,
             b_coeff_names):  # pragma: no cover
        """ Objective function for fitting SIP."""
        from astropy.modeling.models import SIP # here instead of top to avoid circular import

        crpix = params[0:2]
        cdx = params[2:6].reshape((2,2))
        a_params, b_params = params[6:6+len(a_coeff_names)],params[6+len(a_coeff_names):]
        w_obj.wcs.pc = cdx
        w_obj.wcs.crpix = crpix
        x,y = w_obj.wcs_world2pix(lon,lat,1) #'intermediate world coordinates', x & y
        x,y = np.dot(w_obj.wcs.pc,(x - w_obj.wcs.crpix[0],y - w_obj.wcs.crpix[1]))

        a_params, b_params = params[6:6+len(a_coeff_names)], params[6+len(a_coeff_names):]
        a_coeff, b_coeff = {}, {}
        for i in range(len(a_coeff_names)):
            a_coeff['A_' + a_coeff_names[i]] = a_params[i]
        for i in range(len(b_coeff_names)):
            b_coeff['B_' + b_coeff_names[i]] = b_params[i]

        sip = SIP(crpix=crpix, a_order=a_order, b_order=b_order, a_coeff=a_coeff, \
                  b_coeff=b_coeff)
        fuv, guv = sip(u,v)
        xo, yo = np.dot(cdx, np.array([u+fuv-crpix[0], v+guv-crpix[1]]))
        resids = np.concatenate((x-xo, y-yo))
        return resids


def fit_wcs_from_points(xp, yp, coords, mode, projection='TAN', proj_point=None, order=None, inwcs=None):  # pragma: no cover
    """Given a set of matched x,y pixel positions and celestial coordinates, solves for
    WCS parameters and returns a WCS with these best fit values and other keywords based
    on projection type, frame, and units.
    Along with the matched coordinate pairs, users must provide the projection type
    (e.g. 'TAN'), celestial coordinate pair for the projection point (or 'center' to use
    the center of input sky coordinates), mode ('wcs' to fit only linear terms, or 'all'
    to fit a SIP to the data). If a coordinate pair is passed to 'proj_point', it is
    assumed to be in the same units respectivley as the input (lon, lat) coordinates.
    Additionaly, if mode is set to 'all', the polynomial order must be provided.
    Optionally, an existing ~astropy.wcs.WCS object with some may be passed in. For
    example, this is useful if the user wishes to refit an existing WCS with better
    astrometry, or are using a projection type with non-standard keywords. If any overlap
    between keyword arguments passed to the function and values in the input WCS exist,
    the keyword argument will override, including the CD/PC convention.
    NAXIS1 and NAXIS2 will set as the span of the input points, unless an input WCS is
    provided.
    All output will be in degrees.
    Parameters
    ----------
    xp, yp : float or `numpy.ndarray`
        x & y pixel coordinates.
    coords : `~astropy.coordinates.SkyCoord`
        Skycoord object.
    mode : str
        Whether SIP distortion should be fit (``'all'``), or only the linear
        terms of the core WCS transformation (``'wcs'``).
    inwcs : None or `~astropy.wcs.WCS`
        Optional input WCS object. Populated keyword values will be used. Keyword
        arguments passed to the function will override any in 'inwcs' if both are present.
    projection : None or str
        Three-letter projection code. Defaults to TAN.
    proj_point: `~astropy.coordinates.SkyCoord` or str
        Celestial coordinates of projection point (lat, lon). If 'center', the geometric
        center of input coordinates will be used for the projection. If None, and input
        wcs with CRVAL set must be passed in.
    order : None, int, or tuple of ints
        A order, B order, respectivley, for SIP polynomial to be fit, or single integer
        for both. Must be provided if mode is 'sip'.
    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        WCS object with best fit coefficients given the matched set of X, Y pixel and
        sky positions of sources.
    Notes
    -----
    Please pay careful attention to the logic that applies when both an input WCS with
    keywords populated is passed in. For example, if no 'CUNIT' is set in this input
    WCS, the units of the CRVAL etc. are assumed to be the same as the input Skycoord.
    Additionally, if 'RADESYS' is not set in the input WCS, this will be taken from the
    Skycoord as well. """

    from scipy.optimize import least_squares
    from astropy.wcs import Sip
    import copy

    lon, lat = coords.data.lon.deg, coords.data.lat.deg
    mode = mode.lower()
    if mode not in ("wcs", "all"):
        raise ValueError("mode must be 'wcs' or 'all'")
    if inwcs is not None:
        inwcs = copy.deepcopy(inwcs)
    if projection is None:
        if (inwcs is None) or ('' in inwcs.wcs.ctype):
            raise ValueError("Must provide projection type or input WCS with CTYPE.")
        projection = inwcs.wcs.ctype[0].replace('-SIP','').split('-')[-1]

    # template wcs
    wcs = celestial_frame_to_wcs(frame=coords.frame, projection=projection)

    # Determine CRVAL from input, and
    close = lambda l, p: p[np.where(np.abs(l) == min(np.abs(l)))[0][0]]
    if str(proj_point) == 'center': #use center of input points
        wcs.wcs.crval = ((max(lon)+min(lon))/2.,(max(lat)+min(lat))/2.)
        wcs.wcs.crpix = ((max(xp)+min(xp))/2.,(max(yp)+min(yp))/2.) #initial guess
    elif (proj_point is None) and (inwcs is None):
        raise ValueError("Must give proj_point as argument or as CRVAL in input wcs.")
    elif proj_point is not None: # convert units + rough initial guess for crpix for fit
        lon_u, lat_u = u.Unit(coords.data.lon.unit), u.Unit(coords.data.lat.unit)
        wcs.wcs.crval = (proj_point[0]*lon_u.to(u.deg), proj_point[1]*lat_u.to(u.deg))
        wcs.wcs.crpix = (close(lon-wcs.wcs.crval[0],xp),close(lon-wcs.wcs.crval[1],yp))

    if inwcs is not None:
        if inwcs.wcs.radesys == '': # no frame specified, use Skycoord's frame (warn???)
            inwcs.wcs.radesys = coords.frame.name
        wcs.wcs.radesys = inwcs.wcs.radesys
        coords.transform_to(inwcs.wcs.radesys.lower()) #work in inwcs units
        if proj_point is None: # crval, crpix from wcs. crpix used for initial guess.
            wcs.wcs.crval, wcs.wcs.crpix = inwcs.wcs.crval, inwcs.wcs.crpix
            if wcs.wcs.crpix[0] == wcs.wcs.crpix[1] == 0: # assume 0 wasn't intentional
                wcs.wcs.crpix = (close(lon-wcs.wcs.crval[0],xp), \
                                 close(lon-wcs.wcs.crval[1],yp))
        if inwcs.wcs.has_pc():
            wcs.wcs.pc = inwcs.wcs.pc # work internally with pc
        if inwcs.wcs.has_cd():
            wcs.wcs.pc = inwcs.wcs.cd
        # create dictionaries of both wcs (input and kwarg), merge them favoring kwargs
        wcs_dict = dict(wcs.to_header(relax=False))
        in_wcs_dict = dict(inwcs.to_header(relax=False))
        wcs = WCS({**in_wcs_dict,**wcs_dict})
        wcs.pixel_shape = inwcs.pixel_shape
    else:
        wcs.pixel_shape = (max(xp) - min(xp), max(yp) - min(yp))

    # fit
    p0 = np.concatenate((wcs.wcs.pc.flatten(),wcs.wcs.crpix.flatten()))
    fit = least_squares(_linear_transformation_fit, p0, args=(lon, lat, xp, yp, wcs))

    # put fit values in wcs
    wcs.wcs.crpix = np.array(fit.x[4:6])
    wcs.wcs.pc = np.array(fit.x[0:4].reshape((2,2)))

    if mode == "all":
        wcs.wcs.ctype = [x + '-SIP' for x in wcs.wcs.ctype]
        if (order is None) or (type(order) == float):
            raise ValueError("Must provide integer or tuple (a_order, b_order) for SIP.")
        if type(order) == int:
            a_order = b_order = order
        elif (len(order) == 2):
            if (type(order[0]) != int) or (type(order[1]) != int):
                raise ValueError("Must provide integer or tuple (a_order, b_order) for SIP.")
            a_order, b_order = order
        else:
            raise ValueError("Must provide integer or tuple (a_order, b_order) for SIP.")
        a_coef_names = ['{0}_{1}'.format(i,j) for i in range(a_order+1) \
                        for j in range(a_order+1) if (i+j) < (a_order+1) and (i+j) > 1]
        b_coef_names = ['{0}_{1}'.format(i,j) for i in range(b_order+1) \
                        for j in range(b_order+1) if (i+j) < (b_order + 1) and (i+j) > 1]
        # fit
        p0 = np.concatenate((np.array(wcs.wcs.crpix), wcs.wcs.pc.flatten(), \
                             np.zeros(len(a_coef_names)+len(b_coef_names))))
        fit = least_squares(_sip_fit, p0, args=(lon, lat, xp, yp, wcs, a_order, b_order,
                                                a_coef_names, b_coef_names))
        # put fit values in wcs
        wcs.wcs.pc = fit.x[2:6].reshape((2,2))
        wcs.wcs.crpix = fit.x[0:2]
        coef_fit = (list(fit.x[6:6+len(a_coef_names)]),list(fit.x[6+len(b_coef_names):]))
        a_vals, b_vals = np.zeros((a_order+1,a_order+1)), np.zeros((b_order+1,b_order+1))
        for coef_name in a_coef_names:
            a_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[0].pop(0)
        for coef_name in b_coef_names:
            b_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[1].pop(0)
        wcs.sip = Sip(a_vals, b_vals, a_vals * -1., b_vals * -1., wcs.wcs.crpix)

    if (inwcs is not None): # maintain cd matrix if it was in input wcs
        if inwcs.wcs.has_cd():
            wcs.wcs.cd = wcs.wcs.pc
            wcs.wcs.__delattr__('pc')

    return wcs
