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
from astropy.coordinates import Angle, SkyCoord, UnitSphericalRepresentation


def offset_by(lon, lat, posang, distance):
    """
    Point with the given offset from the given point.
    Parameters
    ----------
    lon, lat, posang, distance : `~astropy.coordinates.Angle`, `~astropy.units.Quantity` or float
        Longitude and latitude of the starting point,
        position angle and distance to the final point.
        Quantities should be in angular units; floats in radians.
        Polar points at lat= +/-90 are treated as limit of +/-(90-epsilon) and same lon.
    Returns
    -------
    lon, lat : `~astropy.coordinates.Angle`
        The position of the final point.  If any of the angles are arrays,
        these will contain arrays following the appropriate `numpy` broadcasting rules.
        0 <= lon < 2pi.
    Notes
    -----
    """

    # Calculations are done using the spherical trigonometry sine and cosine rules
    # of the triangle A at North Pole,   B at starting point,   C at final point
    # with angles     A (change in lon), B (posang),            C (not used, but negative reciprocal posang)
    # with sides      a (distance),      b (final co-latitude), c (starting colatitude)
    # B, a, c are knowns; A and b are unknowns
    # https://en.wikipedia.org/wiki/Spherical_trigonometry

    cos_a = np.cos(distance)
    sin_a = np.sin(distance)
    cos_c = np.sin(lat)
    sin_c = np.cos(lat)
    cos_B = np.cos(posang)
    sin_B = np.sin(posang)

    # cosine rule: Know two sides: a,c and included angle: B; get unknown side b
    cos_b = cos_c * cos_a + sin_c * sin_a * cos_B
    # sin_b = np.sqrt(1 - cos_b**2)
    # sine rule and cosine rule for A (using both lets arctan2 pick quadrant).
    # multiplying both sin_A and cos_A by x=sin_b * sin_c prevents /0 errors
    # at poles.  Correct for the x=0 multiplication a few lines down.
    # sin_A/sin_a == sin_B/sin_b    # Sine rule
    xsin_A = sin_a * sin_B * sin_c
    # cos_a == cos_b * cos_c + sin_b * sin_c * cos_A  # cosine rule
    xcos_A = cos_a - cos_b * cos_c

    A = Angle(np.arctan2(xsin_A, xcos_A), u.radian)
    # Treat the poles as if they are infinitesimally far from pole but at given lon
    small_sin_c = sin_c < 1e-12
    if small_sin_c.any():
        # For south pole (cos_c = -1), A = posang; for North pole, A=180 deg - posang
        A_pole = (90*u.deg + cos_c*(90*u.deg-Angle(posang, u.radian))).to(u.rad)
        if A.shape:
            # broadcast to ensure the shape is like that of A, which is also
            # affected by the (possible) shapes of lat, posang, and distance.
            small_sin_c = np.broadcast_to(small_sin_c, A.shape)
            A[small_sin_c] = A_pole[small_sin_c]
        else:
            A = A_pole

    outlon = (Angle(lon, u.radian) + A).wrap_at(360.0*u.deg).to(u.deg)
    outlat = Angle(np.arcsin(cos_b), u.radian).to(u.deg)

    return outlon, outlat

def directional_offset_by(sky_coord, position_angle, separation):
        """
        Computes coordinates at the given offset from this coordinate.

        Parameters
        ----------
        position_angle : `~astropy.coordinates.Angle`
            position_angle of offset
        separation : `~astropy.coordinates.Angle`
            offset angular separation

        Returns
        -------
        newpoints : `~astropy.coordinates.SkyCoord`
            The coordinates for the location that corresponds to offsetting by
            the given `position_angle` and `separation`.

        Notes
        -----
        Returned SkyCoord frame retains only the frame attributes that are for
        the resulting frame type.  (e.g. if the input frame is
        `~astropy.coordinates.ICRS`, an ``equinox`` value will be retained, but
        an ``obstime`` will not.)

        For a more complete set of transform offsets, use `~astropy.wcs.WCS`.
        `~astropy.coordinates.SkyCoord.skyoffset_frame()` can also be used to
        create a spherical frame with (lat=0, lon=0) at a reference point,
        approximating an xy cartesian system for small offsets. This method
        is distinct in that it is accurate on the sphere.

        See Also
        --------
        position_angle : inverse operation for the ``position_angle`` component
        separation : inverse operation for the ``separation`` component

        """

        slat = sky_coord.represent_as(UnitSphericalRepresentation).lat
        slon = sky_coord.represent_as(UnitSphericalRepresentation).lon

        newlon, newlat = offset_by(lon=slon, lat=slat,
                                   posang=position_angle, distance=separation)

        return SkyCoord(newlon, newlat, frame=sky_coord.frame)

    

def _linear_wcs_fit(params, lon, lat, x, y, w_obj):  # pragma: no cover
    """
    Objective function for fitting linear terms.

    Parameters
    ----------
    params : array
        6 element array. First 4 elements are PC matrix, last 2 are CRPIX.
    lon, lat: array
        Sky coordinates.
    x, y: array
        Pixel coordinates
    w_obj: `~astropy.wcs.WCS`
        WCS object
        """
    cd = params[0:4]
    crpix = params[4:6]

    w_obj.wcs.cd = ((cd[0], cd[1]), (cd[2], cd[3]))
    w_obj.wcs.crpix = crpix
    lon2, lat2 = w_obj.wcs_pix2world(x, y, 0)

    resids = np.concatenate((lon-lon2, lat-lat2))
    resids[resids > 180] = 360 - resids[resids > 180]
    resids[resids < -180] = 360	+ resids[resids < -180]

    return resids


def _sip_fit(params, lon, lat, u, v, w_obj, order, coeff_names):  # pragma: no cover

    """ Objective function for fitting SIP.
     Parameters
    -----------
    params : array
        Fittable parameters. First 4 elements are PC matrix, last 2 are CRPIX.
    lon, lat: array
        Sky coordinates.
    u, v: array
        Pixel coordinates
    w_obj: `~astropy.wcs.WCS`
        WCS object
    """

    from astropy.modeling.models import SIP, InverseSIP   # here to avoid circular import

    # unpack params
    crpix = params[0:2]
    cdx = params[2:6].reshape((2, 2))
    a_params = params[6:6+len(coeff_names)]
    b_params = params[6+len(coeff_names):]

    # assign to wcs, used for transfomations in this function
    w_obj.wcs.cd = cdx
    w_obj.wcs.crpix = crpix

    a_coeff, b_coeff = {}, {}
    for i in range(len(coeff_names)):
        a_coeff['A_' + coeff_names[i]] = a_params[i]
        b_coeff['B_' + coeff_names[i]] = b_params[i]

    sip = SIP(crpix=crpix, a_order=order, b_order=order,
              a_coeff=a_coeff, b_coeff=b_coeff)
    fuv, guv = sip(u, v)

    xo, yo = np.dot(cdx, np.array([u+fuv-crpix[0], v+guv-crpix[1]]))

    # use all pix2world in case `projection` contains distortion table
    x, y = w_obj.all_world2pix(lon, lat, 0)
    x, y = np.dot(w_obj.wcs.cd, (x-w_obj.wcs.crpix[0], y-w_obj.wcs.crpix[1]))

    resids = np.concatenate((x-xo, y-yo))
    # to avoid bad restuls if near 360 -> 0 degree crossover
    resids[resids > 180] = 360 - resids[resids > 180]
    resids[resids < -180] = 360 + resids[resids < -180]

    return resids



def fit_wcs_from_points(xy, world_coords, proj_point='center',
                        projection='TAN', sip_degree=None):  # pragma: no cover
    """
    Given two matching sets of coordinates on detector and sky,
    compute the WCS.

    Fits a WCS object to matched set of input detector and sky coordinates.
    Optionally, a SIP can be fit to account for geometric
    distortion. Returns an `~astropy.wcs.WCS` object with the best fit
    parameters for mapping between input pixel and sky coordinates.

    The projection type (default 'TAN') can passed in as a string, one of
    the valid three-letter projection codes - or as a WCS object with
    projection keywords already set. Note that if an input WCS has any
    non-polynomial distortion, this will be applied and reflected in the
    fit terms and coefficients. Passing in a WCS object in this way essentially
    allows it to be refit based on the matched input coordinates and projection
    point, but take care when using this option as non-projection related
    keywords in the input might cause unexpected behavior.

    Notes
    ------
    - The fiducial point for the spherical projection can be set to 'center'
      to use the mean position of input sky coordinates, or as an
      `~astropy.coordinates.SkyCoord` object.
    - Units in all output WCS objects will always be in degrees.
    - If the coordinate frame differs between `~astropy.coordinates.SkyCoord`
      objects passed in for ``world_coords`` and ``proj_point``, the frame for
      ``world_coords``  will override as the frame for the output WCS.
    - If a WCS object is passed in to ``projection`` the CD/PC matrix will
      be used as an initial guess for the fit. If this is known to be
      significantly off and may throw off the fit, set to the identity matrix
      (for example, by doing wcs.wcs.pc = [(1., 0.,), (0., 1.)])

    Parameters
    ----------
    xy : tuple of two `numpy.ndarray`
        x & y pixel coordinates.
    world_coords : `~astropy.coordinates.SkyCoord`
        Skycoord object with world coordinates.
    proj_point : 'center' or ~astropy.coordinates.SkyCoord`
        Defaults to 'center', in which the geometric center of input world
        coordinates will be used as the projection point. To specify an exact
        point for the projection, a Skycoord object with a coordinate pair can
        be passed in. For consistency, the units and frame of these coordinates
        will be transformed to match ``world_coords`` if they don't.
    projection : str or `~astropy.wcs.WCS`
        Three letter projection code, of any of standard projections defined
        in the FITS WCS standard. Optionally, a WCS object with projection
        keywords set may be passed in.
    sip_degree : None or int
        If set to a non-zero integer value, will fit SIP of degree
        ``sip_degree`` to model geometric distortion. Defaults to None, meaning
        no distortion corrections will be fit.

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        The best-fit WCS to the points given.
    """

    from astropy.coordinates import SkyCoord # here to avoid circular import
    import astropy.units as u
    from astropy.wcs import Sip
    from scipy.optimize import least_squares

    xp, yp = xy
    try:
        lon, lat = world_coords.data.lon.deg, world_coords.data.lat.deg
    except AttributeError:
        unit_sph =  world_coords.unit_spherical
        lon, lat = unit_sph.lon.deg, unit_sph.lat.deg

    # verify input
    if (proj_point != 'center') and (type(proj_point) != type(world_coords)):
        raise ValueError("proj_point must be set to 'center', or an" +
                         "`~astropy.coordinates.SkyCoord` object with " +
                         "a pair of points.")
    if proj_point != 'center':
        assert proj_point.size == 1

    proj_codes = [
        'AZP', 'SZP', 'TAN', 'STG', 'SIN', 'ARC', 'ZEA', 'AIR', 'CYP',
        'CEA', 'CAR', 'MER', 'SFL', 'PAR', 'MOL', 'AIT', 'COP', 'COE',
        'COD', 'COO', 'BON', 'PCO', 'TSC', 'CSC', 'QSC', 'HPX', 'XPH'
    ]
    if type(projection) == str:
        if projection not in proj_codes:
            raise ValueError("Must specify valid projection code from list of "
                             + "supported types: ", ', '.join(proj_codes))
        # empty wcs to fill in with fit values
        wcs = celestial_frame_to_wcs(frame=world_coords.frame,
                                     projection=projection)
    else: #if projection is not string, should be wcs object. use as template.
        wcs = copy.deepcopy(projection)
        wcs.cdelt = (1., 1.) # make sure cdelt is 1
        wcs.sip = None

    # Change PC to CD, since cdelt will be set to 1
    if wcs.wcs.has_pc():
        wcs.wcs.cd = wcs.wcs.pc
        wcs.wcs.__delattr__('pc')

    if (type(sip_degree) != type(None)) and (type(sip_degree) != int):
        raise ValueError("sip_degree must be None, or integer.")

    # set pixel_shape to span of input points
    wcs.pixel_shape = (xp.max()+1-xp.min(), yp.max()+1-yp.min())

    # determine CRVAL from input
    close = lambda l, p: p[np.argmin(np.abs(l))]
    if str(proj_point) == 'center':  # use center of input points
        sc1 = SkyCoord(lon.min()*u.deg, lat.max()*u.deg)
        sc2 = SkyCoord(lon.max()*u.deg, lat.min()*u.deg)
        pa = sc1.position_angle(sc2)
        sep = sc1.separation(sc2)
        midpoint_sc = directional_offset_by(sc1, pa, sep/2)
        wcs.wcs.crval = ((midpoint_sc.data.lon.deg, midpoint_sc.data.lat.deg))
        wcs.wcs.crpix = ((xp.max()+xp.min())/2., (yp.max()+yp.min())/2.)
    elif proj_point is not None:  # convert units, initial guess for crpix
        proj_point.transform_to(world_coords)
        wcs.wcs.crval = (proj_point.data.lon.deg, proj_point.data.lat.deg)
        wcs.wcs.crpix = (close(lon-wcs.wcs.crval[0], xp),
                         close(lon-wcs.wcs.crval[1], yp))

    # fit linear terms, assign to wcs
    # use (1, 0, 0, 1) as initial guess, in case input wcs was passed in
    # and cd terms are way off.
    p0 = np.concatenate([wcs.wcs.cd.flatten(), wcs.wcs.crpix.flatten()])
    
    xpmin, xpmax, ypmin, ypmax = xp.min(), xp.max(), yp.min(), yp.max()
    if xpmin==xpmax: xpmin, xpmax = xpmin-0.5, xpmax+0.5
    if ypmin==ypmax: ypmin, ypmax = ypmin-0.5, ypmax+0.5
    
    fit = least_squares(_linear_wcs_fit, p0,
                        args=(lon, lat, xp, yp, wcs),
                        bounds=[[-np.inf,-np.inf,-np.inf,-np.inf, xpmin, ypmin],
                                [ np.inf, np.inf, np.inf, np.inf, xpmax, ypmax]])
    wcs.wcs.crpix = np.array(fit.x[4:6])
    wcs.wcs.cd = np.array(fit.x[0:4].reshape((2, 2)))

    # fit SIP, if specified. Only fit forward coefficients
    if sip_degree:
        degree = sip_degree
        if '-SIP' not in wcs.wcs.ctype[0]:
            wcs.wcs.ctype = [x + '-SIP' for x in wcs.wcs.ctype]

        coef_names = ['{0}_{1}'.format(i, j) for i in range(degree+1)
                      for j in range(degree+1) if (i+j) < (degree+1) and
                      (i+j) > 1]
        p0 = np.concatenate((np.array(wcs.wcs.crpix), wcs.wcs.cd.flatten(),
                             np.zeros(2*len(coef_names))))

        fit = least_squares(_sip_fit, p0,
                            args=(lon, lat, xp, yp, wcs, degree, coef_names))
        coef_fit = (list(fit.x[6:6+len(coef_names)]),
                    list(fit.x[6+len(coef_names):]))

        # put fit values in wcs
        wcs.wcs.cd = fit.x[2:6].reshape((2, 2))
        wcs.wcs.crpix = fit.x[0:2]

        a_vals = np.zeros((degree+1, degree+1))
        b_vals = np.zeros((degree+1, degree+1))

        for coef_name in coef_names:
            a_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[0].pop(0)
            b_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[1].pop(0)

        wcs.sip = Sip(a_vals, b_vals, np.zeros((degree+1, degree+1)),
                      np.zeros((degree+1, degree+1)), wcs.wcs.crpix)

    return wcs
