"""Calculations of physical atmospheric properties

This module contains functions to calculate derived variables based on the
variables originally contained in the loaded dataset.
The (default ECMWF) input dataset `data` contains the following variables on
all (vertical) model levels and all points in time:
- air temperature `data.t` in Kelvin
- specific humidity `data.q` in Kilogram per Kilogram
- eastward and northward component of the wind `data.u` and `data.v` in Meter
    per Second
- Lagrangian tendency of air pressure, i.e. vertical velocity with respect to
    pressure `data.w` in Pascal per Second
- relative vorticity `data.vo` in Hertz (i.e., per Seconds)
- fraction of cloud cover `data.cc` from 0 to 1
- specific cloud liquid and ice water content, `data.clwc` and `data.ciwc`,
    respectively, in Kilogram per Kilogram
- specific rain and snow water content, `data.crwc` and `data.cswc`,
    respectively, in Kilogram per Kilogram
- ozone mass mixing ratio `data.o3` in Kilogram per Kilogram
The logarithmic surface pressure `data.lnsp` has nor vertical coordinate but
spans only longitudes, latitudes and time. The surface geopotential `data.z`
(corresponding to the topography) does not change with time.
The vertical coordinates (pressure, geopotential and geopotential height) are
calculated by the `vertical_coordinates` module.


TODO: continue here
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



Author(s): Alzbeta Medvedova, Moritz Oberrauch

See Also
--------
vertical_coordinates : module calculating pressure and geopotential (height) on
    full and half model levels.

References
----------
 .. [Bolton 1980] Bolton, D., 1980, The computation of equivalent potential
    temperature. Mon. Wea. Rev., 108, 1046-1053

 .. [Hobbs 2006]: Hobbs, P. V., and J. M. Wallace, 2006, Atmospheric Science:
    An Introductory Survey. 2nd ed. Academic Press, 504 pp.

 .. [Stull 2011]: Stull, R., 2011, Meteorology for Scientists & Engineers,
    3rd Edition. Univ. of British Columbia,  938 pp.,  ISBN 978-0-88865-178-5


"""
# build ins

# external libraries
import numpy as np
import pandas as pd

# local imports -
import constants as const


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
    theta = data.t * (const.PRESSURE_0 / data.pressure) ** (
            const.R_DRY / const.SPEC_CP_DRY)
    return theta


def N_dry_from_p_theta(data):
    # TODO: implement if needed
    # Calculate Brunt Vaisala frequency... need T, p or something?
    # What do I need here? MetPY has sigma... What's sigma?
    # sigma = -RT/p * d(ln theta)/dp... huh?
    # I want N_m

    # dry: N = sqrt(g/theta * d(theta)/dz)
    raise NotImplementedError
    theta = theta_from_t_p(data)
    theta_vertical_gradient = 1
    brunt_vaisala_freq = np.sqrt(const.G / theta * theta_vertical_gradient)
    return brunt_vaisala_freq


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

def virtual_temperature(data):
    """Virtual temperature

    Calculate approximated virtual temperature following Eq. 3.16 [Hobbs 2006]_
    from temperature and mixing ratio.

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature [K] (data.t) and mixing ratio [kg/kg]
        (data.q)

    Returns
    -------
    float:
        approximated virtual temperature

    Notes
    -----
    TODO: check if variables are consistent with rest

    Virtual temperature accounts for water vapor when using the equation of
    state for ideal gases with the gas constant for dry air :math:`R_d`.
    Following [Hobbs 2006]_ it is defined as

    .. math:: T_v = \frac{T}{1-\frac{e}{p}(1-\varepsilon)},

    whereby :math:`T` is the air temperature, :math:`e` the partial pressure of
    water vapor, :math:`p` the air pressure and :math:`\varepsilon = R_d/R_v`
    the ratio between gas constant of dry air and water vapor.

    The partial pressure of water vapor can be expressed as
    .. math:: e = \frac{w}{w+\varepsilon}p.
    Substituting this in the equation above, and neglecting all terms with
    :math:`w^2` (or even higher orders) we get

    .. math:: T_v \simeq T (1+(1/\varepsilon-1)w).


    """

    t_v = data['t'] * (1 + (const.R_WATER / const.R_DRY - 1.0) * data['q'])
    return t_v


def es_from_t(data):
    """Saturation pressure of water vapor

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

    es = 611.2 * np.exp(17.67 * (data.t - const.TEMP_0) / (data.t - 29.65))
    return es


def rh_from_t_q_p(data):
    """Relative humidity

    Calculate relative humidity [%] from temperature [K], mixing ratio [kg/kg]
    and pressure [Pa].

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

    Calculate relative humidity :math:`RH` in percent [%] as the ratio between
    mixing ratio :math:`q` and saturation mixing ratio :math:`q_s` following
    [Hobbs 2006]_ Eq. 3.64 (p. 82)

    .. math:: RH = 100\frac{q}{q_s}

    The saturation mixing ratio is computed as follows [Hobbs 2006]_, Eq. 3.63

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
    rho_dry = (data.pressure - e) / (const.R_DRY * data.t)
    rho_water = e / (const.R_WATER * data.t)
    rho = rho_dry + rho_water

    # Convert vertical wind speed from pressure to height coordinates
    w_ms = data.w / (-rho * const.G)
    return w_ms


def T_lcl_from_T_rh(data):
    """Temperature at lifting condensation level (LCL)

    Calculate the absolute temperature at the lifting condensation level
    following [Bolton 1980]_ Eq. 22.

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature [K] (data.t), and either relative
        humidity [%] (data.rh) or pressure [Pa] (data.pressure) and mixing
        ratio [kg/kg] (data.q)in oder to compute the relative humidity

    Returns
    -------
    T_lcl : xr.DataArray
        absolute temperature [K] at the LCL

    Notes
    -----

    The function follows Eq. 22 from [Bolton 1960]_ which reads

    .. math:: T_{LCL} = \frac{1}{\frac{1}{T - 55} - \frac{\ln(RH/100)}{2840}}
        + 55.

    """

    if 'rh' not in data.keys():
        # Calculate relative humidity if not in data
        rh = rh_from_t_q_p(data)
    else:
        rh = data['rh']

    # calculate temperature at lifting condensation level
    denominator = (1 / (data.t - 55)) + (np.log(rh * 1e-2) / 2840)
    T_lcl = 1 / denominator + 55
    return T_lcl


def theta_e_from_t_p_q_Tlcl(data):
    """Equivalent potential temperature

    Calculates equivalent potential temperature following Eq. 43 in
    [Bolton 1980]_.

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature [K] (data.t), relative humidity [%]
        (data.rh), pressure [Pa] (data.pressure) and mixing ratio [kg/kg]
        (data.q)

    Returns
    -------
    theta_e : xr.DataArray
        Equivalent potential temperature

    Notes
    -----
    Equation 43 in the conclusion of [Bolton 1980]_ reads

    .. math:: \theta_e = T\left(\frac{1000~mathrm{hPa}}{p}\right)
        ^{\frac{R_d}{c_p}(1-0.28q)}
        \cdot \exp\left[\left(\fraq{3.376}{T_\text{LCL}} - 0.00254\right)
            q(1+0.81q)\right].

    Thereby, :math:`T` represents the absolute temperature [K], :math:`p` the
    pressure [Pa], :math:`q` the mixing ratio [kg/kg] and :math:`T_\text{LCL}`
    the the absolute temperature at the lifting condensation level [K].

    """

    # Calculate temperature at LCL
    T_lcl = T_lcl_from_T_rh(data)

    # Specify exponents for following equation
    exp1 = const.R_DRY / const.SPEC_CP_DRY * (1 - 0.28 * data.q)
    exp2 = (3.376 / T_lcl - 0.00254) * data.q * 1e3 * (1 + 0.81 * data.q)
    # Calculate equivalent potential temperature
    theta_e = data.t * (const.PRESSURE_0 / data.pressure) ** exp1 * np.exp(
        exp2)
    return theta_e


def theta_es_from_t_p_q(data):
    """Saturation equivalent potential temperature

    Calculates saturation equivalent potential temperature following Eq. 43 in
    [Bolton 1980]_, by using temperature as temperature at the lifting
    condensation level.

    Parameters
    ----------
    data : xr.Dataset
        dataset containing temperature [K] (data.t), relative humidity [%]
        (data.rh), pressure [Pa] (data.pressure) and mixing ratio [kg/kg]
        (data.q)

    Returns
    -------
    theta_e : xr.DataArray
        Saturation equivalent potential temperature

    Notes
    -----
    Equation 43 in the conclusion of [Bolton 1980]_ reads

    .. math:: \theta_e = T\left(\frac{1000~mathrm{hPa}}{p}\right)
        ^{\frac{R_d}{c_p}(1-0.28q)}
        \cdot \exp\left[\left(\fraq{3.376}{T_\text{LCL}} - 0.00254\right)
            q(1+0.81q)\right].

    Thereby, :math:`T` represents the absolute temperature [K], :math:`p` the
    pressure [Pa], :math:`q` the mixing ratio [kg/kg] and :math:`T_\text{LCL}`
    the the absolute temperature at the lifting condensation level [K].

    """

    # Specify exponents for following equation
    exp1 = const.R_DRY / const.SPEC_CP_DRY * (1 - 0.28 * data.q)
    exp2 = (3.376 / data.t - 0.00254) * data.q * 1e3 * (1 + 0.81 * data.q)
    # Calculate equivalent potential temperature
    theta_e = data.t * (const.PRESSURE_0 / data.pressure) ** exp1 * np.exp(
        exp2)
    return theta_e


def N_moist_squared(data):
    """Moist Brunt-Väisälä frequency

    TODO: not look at yet

    Calculates moist Brunt-Vaisala frequency [rad/s]

    Based on Kirshbaum [2004] eq. 6, or
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
    gamma_m = const.GAMMA_DRY * (1 + (a * data.q / data.t)) / (
            1 + (b * data.q / (data.t ** 2)))

    # numpy can't derive on a non-uniform meshgrid: get df and dz separately
    df = np.gradient(
        (const.SPEC_CP_DRY + const.SPEC_CP_WATER * r_tot) * np.log(
            data.theta_e), axis=0)
    dz = np.gradient(data.geopotential_height, axis=0)
    dq = np.gradient(r_tot, axis=0)

    term_1 = gamma_m * (df / dz)
    term_2 = (const.SPEC_CP_WATER * gamma_m * np.log(data.t) + const.G) * (
            dq / dz)

    frac = 1 / (1 + r_tot)
    N_m_squared = frac * (term_1 - term_2)

    return N_m_squared


# %% Rotation of wind coordinates

def bearing(lon0, lat0, lon1, lat1):
    """Bearing angle

    TODO: not look at yet

    Calculate the bearing angle [rad] between two positions given by coordinate
    points. The bearing angle is measured clockwise from the north direction.
    Used for re-calculating the wind direction in the transect plane and in/out
    of page for the diagonal cross-sections.

    The diagonal cross-sections does NOT have a constant bearing - the first
    point gives just an approximation. TODO: ask Betka

    Parameters
    ----------
    lon0 : float
        Longitudinal position of start point in degrees west
    lat0 : float
        Latitudinal position of start point in degrees north
    lon1 : float
        Longitudinal position of end point in degrees west
    lat1 : float
        Latitudinal position of end point in degrees north
        

    Returns
    -------
    bearing : float
        bearing angle [rad] between start point (lon0, lat0) and end point
        (lon1, lat1)

    """

    # Convert degrees into radians
    [lon0, lat0, lon1, lat1] = np.deg2rad([lon0, lat0, lon1, lat1])
    # Compute longitudinal distance
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
    TODO: not look at yet

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
    TODO: not look at yet

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
    TODO: not look at yet

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
    # ds['N_m'] = N_moist_squared(ds)  # Moist Brunt-Vaisala frequency [1/s^2]
    return ds
