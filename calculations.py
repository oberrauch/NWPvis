"""

This module contains functions to calculate derived variables based on the
variables originally contained in the loaded dataset.

Author(s): Alzbeta Medvedova, Moritz Oberrauch

References:
 .. [Bolton 1980] Bolton, D., 1980, The computation of equivalent potential
    temperature. Mon. Wea. Rev., 108, 1046-1053

 .. [Hobbs 2006]: Hobbs, P. V., and J. M. Wallace, 2006, Atmospheric Science:
    An Introductory Survey. 2nd ed. Academic Press, 504 pp.

 .. [Stull 2011]: Stull, R., 2011, Meteorology for Scientists & Engineers,
    3rd Edition. Univ. of British Columbia,  938 pp.,  ISBN 978-0-88865-178-5


"""
# build ins

#
import numpy as np
import pandas as pd

# local imports
from constants import g, t_0, p0, Rd, Rvap, c_p, c_l, rcp, gamma_d


# In the input files, we have the following variables:
# t, q, u, v
# w [Pa/s]
# vo: relative vorticity
# cc: cloud fraction
# specific rain water/snow water, cloud liquid water/cloud ice water content
# pressure, geopotential, geopotential height

# we need:
# theta_es (saturation equivalent potential temperature)
# vorticity (relative? probably...)
# N_m^2: brunt-vaisala frequency I think? (?)
# total cloud water
# hydrometeors, suspended and precipitating
# vertical velocity: have in Pa/s, want in m/s? Need Pa/m for conversion
# CIN, CAPE
# horizontal convergence

# calculation pipeline:
# es: saturation vapor pressure
# rh: relative humidity. needs es
# theta: needs p, t
# T_lcl: needs t, rh
# theta_e: needs t, p, q, T_lcl
# vertical velocity in m/s: needs t, p, rh, es


# %% DRY THERMODYNAMICS

def theta_from_t_p(data):
    """Potential temperature

    Compute potential temperature theta at all hybrid levels

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature (data.t) in Kelvin and
        pressure (data.pressure) in Pascal

    Returns
    -------
    theta : xr.DataArray
        potential temperature in Kelvin (for plotting)
    """
    theta = data.t * (p0 / data.pressure) ** rcp
    return theta


def N_dry_from_p_theta(data):
    # TODO
    # Calculate Brunt Vaisala frequency... need T, p or something?
    # What do I need here? MetPY has sigma... What's sigma?
    # sigma = -RT/p * d(ln theta)/dp... huh?
    # I want N_m

    # dry: N = sqrt(g/theta * d(theta)/dz)
    return


def windspeed(data):
    """Wind speed

    Calculate total scalar wind speed from components in x-, y- anz z-direction

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing longitudinal and latitudinal wind components in
        [m/s] (data.u and data.v, respectively), and vertical velocity with
        respect to pressure in [Pa/s] (data.)

    Returns
    -------
    wspd : xr.DataArray
        Scalar wind speed [m/s]
    """
    if 'w_ms' not in data.keys():
        # Compute vertical wind speed with respect to height if not in data
        w = w_from_omega(data)
    else:
        #
        w = data.w_ms

    wspd = np.sqrt(data.u ** 2 + data.v ** 2 + w ** 2)

    return wspd


# %% MOIST THERMODYNAMICS

def es_from_t(data):
    """Saturation vapor pressure

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature (data.t) in Kelvin

    Returns
    -------
    es : xr.DataArray
        Saturation vapor pressure [Pa]

    Notes
    -----
    Calculate saturation vapor pressure :math:`e_s` based on [Bolton 2006]_
    .. math:: e_s(T) = 6.112\exp(\frac{17.67 T}{T + 243.5})
    using temperature :math:``T in Kelvin rather than degree Celsius.

    """

    es = 611.2 * np.exp(17.67 * (data.t - t_0) / (data.t - 29.65))
    return es


def rh_from_t_q_p(data):
    """Relative humidity

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature (data.t), mixing ratio (data.q) and
        pressure (data.pressure)

    Returns
    -------
    rh : xr.DataArray
        relative humidity at model levels [%]

    Notes
    -----

    Calculate relative humidity :math:`RH` in percent [%] at all model levels
    as the ratio between mixing ratio :math:`q` and saturation mixing ratio
    :math:`q_s` following [Hobbs 2006]_ Eq. 3.64 (p. 82)

    .. math:: RH = 100\frac{q}{q_s}

    The saturation mixing ratio is computed as follows [Hobbs 2006], Eq. 3.63

    .. math:: RH = 0.622\frac{e_s}{p - e_s},

    with :math:`e_s` as saturation water vapor pressure and :math:`p` as
    pressure.

    """
    # Calculate saturation vapor pressure from temperature
    es = es_from_t(data)

    # Calculate saturation mixing ratio
    qs = 0.622 * (es / (data.pressure - es))

    # Calculate relative humidity in percent
    rh = 100 * data.q / qs
    return rh


def w_from_omega(data):
    """Vertical wind speed with respect to height

    Convert vertical wind speed with respect to pressure [Pa/s] into that with
    respect to height [m/s] assuming hydrostatic balance on a synoptic scale,
    following [Hobbs 2006]_ Eq. 7.33 (adapted from the MetPy package).

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature [K] (data.t), pressure [Pa]
        (data.pressure), and mixing ratio (data.q)

    Returns
    -------
    w_ms : xr.DataArray
        Vertical wind speed [m/s], up corresponds to the positive z-direction

    Notes
    -----
    Assuming hydrostatic balance

    .. math:: \frac{\mathrm{d}p}{\mathrm{d}z} = -\rho{}g

    there is an approximate linear relationship between vertical wind speed
    with respect to pressure :math:`\omega` [Pa/s] and that with respect to
    height :math:`\omega` [m/s] as described in [Hobbs 2006]_ Eq. 7.33

    .. math:: \omega \simeq -\rho{}gw

    The density :math:``\rho can be seen as the sum of partial densities for
    dry air :math:`\rho_d` and water vapor :math:`\rho_v`. Following
    [Hobbs 2006]_ (p. 67), the density can be computed as

    .. math:: \rho = \rho_d + \rho_v = \frac{p-e}{R_d T} + \frac{e}{R_v T}

    Hereby, :math:`p` represents pressure, :math:`e` the partial pressure of
    water vapor, :math:`R_d` and :math:`R_v` the gas constants for dry air and
    water vapor, respectively.

    """

    if 'rh' not in data.keys():
        # Calculate relative humidity if not in the data
        rh = rh_from_t_q_p(data)
    else:
        rh = data['rh']

    # Calculate partial pressure of water vapor
    e = rh * es_from_t(data) * 1e-2

    # Compute air density accounting for water vapor
    rho_dry = (data.pressure - e) / (Rd * data.t)
    rho_water = e / (Rvap * data.t)
    rho = rho_dry + rho_water

    # Convert vertical wind speed from pressure to height coordinates
    w_ms = -rho * g * data.w
    return w_ms


def T_lcl_from_T_rh(data):
    """Temperature at lifting condensation level (LCL)

    Absolute temperature at the lifting condensation level according to
    Bolton [1980], Eq. 22. Needs temperature [K] and relative humidity [%].

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    """

    # Calculate relative humidity if not in data
    # TODO: ensure consistency
    if 'rh' not in data.keys():
        data['rh'] = rh_from_t_q_p(data)

    denominator = (1 / (data.t - 55)) + (np.log(data.rh / 100) / 2840)
    T_lcl = 1 / denominator + 55
    return T_lcl


def theta_e_from_t_p_q_Tlcl(data):
    """
    Equivalent potential temperature according to Bolton [1980], Eq. 43.
    Function of absolute temperature T [K], pressure p [Pa], mixing
    ratio q [kg/kg] (q is called "r" in Bolton and has unist [g/kg], we have
    [kg/kg]), and absolute temperature at the lifting condensation
    level T_lcl [K].

    Parameters
    ----------
    data : xr.Dataset
        dataset

    Returns
    -------
    theta_e : xr.DataArray
        Equivalent potential temperature

    """

    # Get T_lcl
    T_lcl = T_lcl_from_T_rh(data)

    # Define exponents
    exp_1 = rcp * (1 - 0.28 * data.q)
    exp_2 = (3.376 / T_lcl - 0.00254) * 1e3 * data.q * (1 + 0.81 * data.q)

    # Get theta_e
    theta_e = data.t * (p0 / data.pressure) ** exp_1 * np.exp(exp_2)

    return theta_e


def theta_es_from_t_p_q(data):
    """
    Saturated quivalent potential temperature from Bolton [1980], Eq. 43.
    Function of absolute temperature T [K], pressure p [Pa], mixing
    ratio q [kg/kg] (q is called "r" in Bolton and has unist [g/kg], we have
    [kg/kg]). Since we assume saturation, simply replace T_lcl by temperature

    Parameters
    ----------
    data : xr.Dataset
        dataset

    Returns
    -------
    theta_es : xr.DataArray
        Saturation equivalent potential temperature

    """

    # Define exponents
    exp_1 = rcp * (1 - 0.28 * data.q)
    exp_2 = (3.376 / data.t - 0.00254) * 1e3 * data.q * (1 + 0.81 * data.q)

    # Get theta_e
    theta_es = data.t * (p0 / data.pressure) ** exp_1 * np.exp(exp_2)

    return theta_es


def N_moist_squared(data):
    """
    Moist Brunt-Vaisala frequency. Based on Kirshbaum [2004] eq. 6, or
    Schreiner [2011], eq. 3.1. TODO REF
    Derivatives with numpy: central differences in the interior of the array,
    forward/backward differences at the boundary points

    Parameters
    ----------
    data : xr.Dataset
        dataset

    Returns
    -------
    N_m_squared : xr.DataArray
        Moist Brunt-Vaisala frequency (squared)

    """

    # CHECK: correct to add liquid/cloud water and ice to w to get total water?
    # Total water mixing ratio calculation:
    # first get mixing ratios of all components (common denominator),
    # then sum them: r_tot = m_water_total / m_dry_air

    r_vap = data.q / (1 - data.q)  # water vapor / dry air
    r_w_cloud = data.clwc / (1 - data.clwc)  # cloud liquid water / dry air
    r_i_cloud = data.ciwc / (1 - data.ciwc)  # cloud ice water / dry air
    r_rain = data.crwc / (1 - data.crwc)  # rain / dry air
    r_snow = data.cswc / (1 - data.cswc)  # snow / dry air
    r_tot = r_vap + r_w_cloud + r_i_cloud + r_rain + r_snow  # total water

    # Moist adiabatic lapse rate, after Stull [2011], eq. 4.37b
    a = 8711  # [K]
    b = 1.35e7  # [K^2]

    # TODO: calculating gamma, use total water as well or just vapor?
    gamma_m = gamma_d * (1 + (a * data.q / data.t)) / (
            1 + (b * data.q / (data.t ** 2)))

    # numpy can't derive on a non-uniform meshgrid: get df and dz separately
    df = np.gradient((c_p + c_l * r_tot) * np.log(data.theta_e), axis=0)
    dz = np.gradient(data.geopotential_height, axis=0)
    dq = np.gradient(r_tot, axis=0)

    term_1 = gamma_m * (df / dz)
    term_2 = (c_l * gamma_m * np.log(data.t) + g) * (dq / dz)

    frac = 1 / (1 + r_tot)
    N_m_squared = frac * (term_1 - term_2)

    return N_m_squared


# %% Rotation of wind coordinates

def bearing(lon0, lat0, lon1, lat1):
    """
    Calculate the bearing angle (measured clockwise from the north direction)
    in RADIANS. Used for re-calculating the wind direction in the transect
    plane and in/out of page for the diagonal cross-sections.

    The diagonal cross-sections does NOT have a constant bearing - the first
    point gives just an approximation.

    Parameters
    ----------
    lon0, lat0, lon1, lat1 : float
        Location of the initial and final points of the cross-section

    Returns
    -------
    bearing : float
        bearing angle

    """

    # [lon0, lat0, lon1, lat1] = [5.5, 46.0, 17.3, 52.0]
    [lon0, lat0, lon1, lat1] = np.deg2rad([lon0, lat0, lon1, lat1])
    dLon = lon1 - lon0

    y = np.sin(dLon) * np.cos(lat1)
    x = np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(
        dLon)

    # arctan2 chooses quadrant correctly: resulting angle will lie between
    #   -pi/2 and pi/2 since dLon > 0 by definition of the cross-sections
    #   this angle is based on the unit circle, i.e. w.r.t. east
    #   and corresponds to bearing of 0 to 180 deg w.r.t. north
    bearing = np.arctan2(y, x)

    # however, we want to keep sign of v for the view from the south:
    # limit bearing from -90 to 90 deg (-pi/2 to pi/2) around north
    if bearing > np.pi / 2:
        bearing = bearing - np.pi

    return bearing


def angle(lon0, lat0, lon1, lat1):
    """
    Calculate the angle (measured clockwise from the north direction)
    in RADIANS. Used for re-calculating the wind direction in the transect
    plane and in/out of page for the diagonal cross-sections.

    Although this is technically trigonometry on a sphere, we neglect that:
    the data is projected on a rectangular grid - the cross-sections are not
    really "straight lines" anyway.

    Parameters
    ----------
    lon0, lat0, lon1, lat1 : float
        Location of the initial and final points of the cross-section

    Returns
    -------
    bearing : float
        bearing angle

    """

    # [lon0, lat0, lon1, lat1] = [5.5, 46.0, 17.3, 52.0]  # for trial purposes
    dLon = lon1 - lon0
    dLat = lat1 - lat0

    # arctan2 chooses quadrant correctly: resulting angle will lie between
    #   -pi/2 and pi/2 since dLon > 0 by definition of the cross-sections
    #   this angle is based on the unit circle, i.e. w.r.t. east
    #   and corresponds to an angle of 0 to 180 deg w.r.t. north
    angle = np.arctan2(dLat, dLon)  # equivalent to "atan(y/x)"

    # however, we want to keep sign of v for the view from the south:
    # limit angle from -90 to 90 deg (-pi/2 to pi/2) around north
    if angle > np.pi / 2:
        angle = angle - np.pi

    # angle = np.rad2deg(angle)  # uncomment if we want degrees, not radians
    return angle


def diag_wind(u, v, angle):
    """
    Calculates the transect/perpendicular wind components for diagonal
    cross-sections, based on the angle

    TODO check if this works correctly!

    Parameters
    ----------
    u, v : xr.DataArray
        original u-wind and v-wind components component
    angle : float
        angle in radians

    Returns
    -------
    out_of_page_wind, transect_plane_wind : xr.DataArray
        new wind components. This works when the diagonal cross-sectiones are
        viewed "from the south", i.e. with longitude increasing left to right
        of the figure

    """

    transect_plane_wind = -v * np.cos(angle) + u * np.sin(angle)
    out_of_page_wind = v * np.sin(angle) + u * np.cos(angle)

    return out_of_page_wind, transect_plane_wind


# %% Function to call on a given input dataset to add all derived variables

def calculate_all_vars(ds):
    """
    A function to add all the derived variables to the original input dataset

    Parameters
    ----------
    ds : xr.Dataaset
        Input dataset: contains t, q, u, v, w, vo, crwc, cswc, clwc, ciwc, cc,
        geopotential, geopotential height, pressure...

    Returns
    -------
    ds : xr.Dataset
        Output dataset: contains all the input variables plus all the derived
        quantities

    """
    # add "initial time" attribute to calculate time differences later
    ds.attrs['init_time'] = pd.to_datetime(ds.time[0].values)

    # add all calculated variables
    # TODO check calculations with MetPy / atmos packages?
    ds['rh'] = rh_from_t_q_p(ds)  # Needs temperature still in [K]
    ds['theta'] = theta_from_t_p(ds)  # Potential temperature [K]
    ds['theta_e'] = theta_e_from_t_p_q_Tlcl(ds)  # Equivalent pot. temp. [K]
    ds['theta_es'] = theta_es_from_t_p_q(ds)  # Satur. equiv. pot. temp. [K]
    ds['w_ms'] = w_from_omega(ds)  # Vertical velocity [m/s]
    ds['wspd'] = windspeed(ds)  # Total scalar wind speed [m/s]
    ds['N_m'] = N_moist_squared(ds)  # Moist Brunt-Vaisala frequency [1/s^2]
    return ds
