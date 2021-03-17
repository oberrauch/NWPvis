"""Vertical coordinates

This script calculates pressure and geopotential height on full (and half)
model levels of a xr.Dataset.

TODO: finish/update this docstring.

Based on the script of Johannes Horak and later by Deborah Morgenstern.

Inputs:
ds: xr.Dataset containing:
  1. temperature and humidity on model levels
     (optionally also other variables)
  2. logarithm of surface pressure
  3. surface geopotefinal
ecmwf_ab_coeffs.pkl (optional):
  pickle file containing level definitions a and b - can also be downloaded
  in this script if not yet available

Output:
ds: xr.Dataset, which now contains also pressure fields, geopotential and
  geopotential height on all model levels

Author(s): Alzbeta Medvedova, Moritz Oberrauch

References:
 .. [ECMWF]: European Centre for Medium-Range Weather Forecasts, 2015, "IFS
    DOCUMENTATION – Cy41r1 Operational implementation 12 May 2015, PART III:
    DYNAMICS AND NUMERICAL PROCEDURES", Shinfield Park, Reading, RG2 9AX,
    England, URL: https://www.ecmwf.int/node/9210, DOI: 10.21957/a3hkssbw

"""

import pandas as pd
import xarray as xr
import numpy as np
import requests
import os

# local dependencies
import constants as const
import calculations


# %% MATH FROM THE ECMWF DOCUMENTATION
#
# t, q, tv: temperature, relative humidity, virtual temperature
# tv = t * [1 + (R_vap/R_dry - 1) * q]      (no Eq. number)
# code can be vectorized - no dependency on neighboring values
#
#
# p, p_S: pressure, surface pressure
# a, b: coefficients on half levels, constants from ECMWF
# p[k+1/2] = a[k+1/2] + b[k+1/2] * p_s      (Eq. 2.11)
# p[k] = (p[k+1/2] - p[k+1/2])/2            (no Eq. number)
# code can be vectorized - no dependency on neighboring values
#
#
# dp: pressure differential
# dp[k] = p[k+1/2] - p[k-1/2]               (Eq. 2.13)
# code can be vectorized - no dependency on neighboring values
#
# alpha: function of pressure, pressure gradient
# alpha[1] = ln(2)
# alpha[k] = 1 - p[k-1/2]/dp[k] * ln(p[k+1/2]/p[k-1/2])  (for k > 1, Eq. 2.23)
# code can be vectorized - no dependency on neighboring values
#
#
# psi: geopotential
# psi[k+1/2] = psi[sfc] + (sum_{j=k+1}^NLEV R_dry* tv[j] *
#                           ln(p[j+1/2]/p[j-1/2]))       (Eq. 2.21)
# psi[k] = psi[k+1/2] + alpha[k] * R_dry * tv[k]
# code CANNOT be vectorized - psi has to be calculated iteratively
#  - start at the ground where k = 137, repeat up to low k's
#  - at this point, we will have all tv and p calculated


# FUNCTIONS FOR INPUT DATA

def get_model_level_definition(dir_path=None):
    """Definition of model levels

    TODO: decide if not just keeping this file and removing download routine

    Function loading the coefficients a, b needed for calculating pressure
    at half levels. If the file containing the coefficients does not exist yet,
    it will be downloaded.

    Parameters
    ----------
    dir_path : str
        check it the pickle file containing the coefficients exists
        at this path - if not, download it from ECMWF and save
        as pickle (.pkl) to this path

    Returns
    -------
    xr.DataArray

    """

    # specify path and filename
    if not dir_path:
        # use default path to data directory if not provided
        package_dir, _ = os.path.split(__file__)
        dir_path = os.path.join(package_dir, './data/')
    file_name = "ecmwf_ab_coeffs.nc"
    path = os.path.join(dir_path, file_name)

    if not os.path.isfile(path):
        # download file if necessary
        url = "https://www.ecmwf.int/en/forecasts/" \
              "documentation-and-support/137-model-levels"
        # convert HTML table into a pandas DataFrame
        html = requests.get(url).content
        level_coef = pd.read_html(html, index_col=0)[0]
        # select only constants A and B
        level_coef = level_coef[['a [Pa]', 'b']]
        # rename columns and index and convert into xarray DataArray
        level_coef.index.name = 'level'
        level_coef.rename(columns={'a [Pa]': 'A', 'b': 'B'}, inplace=True)
        level_coef = level_coef.to_xarray()
        # add meta data
        level_coef.A.attrs['units'] = 'Pa'
        level_coef.A.attrs['long_name'] = 'Coefficient A from the ECMWF ' \
                                          'L137 model level definitions ' \
                                          'to compute half-level pressures'
        level_coef.B.attrs['units'] = '-'
        level_coef.B.attrs['long_name'] = 'Coefficient B from the ECMWF ' \
                                          'L137 model level definitions ' \
                                          'to compute half-level pressures'
        # store to file
        level_coef.to_netcdf(path)
    else:
        # read from file
        level_coef = xr.load_dataset(path)

    return level_coef


def get_pressure_and_alpha(ds):
    """Pressure calculations

    Calculate pressure, alpha on full model levels based on Eqns. 2.11, 2.23
    from the [ECMWF]_ documentation.

    Parameters
    ----------
    ds : xr.Dataset
        dataset containing the logarithmic surface pressure (data.lnsp)

    Returns
    -------
    pressure : xr.DataArray
        Pressure [Pa] on full model levels
    alpha: xr.DataArray
        Defined at full levels, needed for geopotential calculation
    pressure_ratio: xr.DataArray
        Ratio on pressures on half-levels, needed for calculating geopotential

    Notes
    -----

    The vertical model layers are defined by the pressure at the interfaces
    between them. The pressure :math:`p_{k+1/2}` at those so called
    'half-levels' is given by the following equation (see [ECMWF]_ Eq. 2.11):

    .. math:: p_{k+1/2} = A_{k+1/2} + B_{k+1/2} \cdot p_s.

    Thereby, :math:`A_{k+1/2}` and :math:`B_{k+1/2}` are constants, :math:`p_s`
    represents the surface pressure field.

    The pressure associated with each model level (i.e., at the middle of the
    layer) is defined as the average between the lower and upper half-level.

    .. math::  p_k = \frac{p_{k-1/2} + p_{k+1/2}}{2}

    For the calculation of the geopotential (Eq. 2.22) two more variables are
    needed. The pressure ratio between the surrounding half levels
    :math:`\frac{p_{k+1/2}}{p_{k-1/2}}` and the :math:`\alpha_k` coefficient

    .. math:: \alpha_k = 1 - \frac{p_{k-1/2}}{\Delta{}p_{k}}
        \ln\left(\frac{p_{k+1/2}}{p_{k-1/2}}\right)

    for :math:`k>1` and :math:`\alpha_1 = \ln{}2`.

    """
    # Get coefficients to compute pressure at half levels
    level_coef = get_model_level_definition()
    # Get sfc pressure from logarithmic surface pressure
    sfc_pressure = np.exp(ds['lnsp'])

    # Compute pressure at all half levels
    p_half_levels = level_coef.A + level_coef.B * sfc_pressure
    p_plus = p_half_levels.drop_sel(level=0)
    p_minus = p_plus.shift(level=1)
    # Compute average between upper and lower half level
    pressure = 0.5 * (p_plus + p_minus)

    # Get pressure ratio on full levels for use in alpha and Eq. 2.21
    pressure_ratio = (p_plus / p_minus)

    # Calculate alpha from Eq. 2.23, needed to compute the full-level values of
    # geopotential as in Eq. 2.22
    delta_p = p_half_levels.diff(dim='level')
    alpha = 1 - (p_minus / delta_p
                 * np.log(pressure_ratio))
    # set alpha at level 1 to ln(2)
    alpha.loc[dict(level=1)] = np.log(2)

    return pressure, alpha, pressure_ratio


def get_geopotential(ds, dir_path=None):
    """Geopotential calculations

    Calculates pressure, geopotential, and geopotential height at all full
    model levels - to be used as vertical coordinates.

    Coefficients a, b needed for conversion are either loaded from a pickle
    file or downloaded in the process

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the following variables for the dimensions time,
        height (level) and longitude/latitude:
        - logarithmic surface pressure (data.lnsp)
        - temperature (data.t)
        - mixing ratio (data.q)
        - surface geopotential (data.z)

    dir_path : str, optional
        path to data directory where file with model levels is stored, defaults
        to './data/'

    Returns
    -------
    ds : xr.Dataset
        Same dataset, now containing also pressure, geopotential and
        geopotential height at all vertical levels
    """

    # Compute needed variables, add to dataset if needed for later
    temp_virtual = calculations.virtual_temperature(ds)
    ds['pressure'], alpha, pressure_ratio = get_pressure_and_alpha(ds)
    # reverse virtual temperature and pressure ratio along level coordinates
    temp_virtual = temp_virtual.reindex(level=temp_virtual.level[::-1])
    pressure_ratio = pressure_ratio.reindex(level=pressure_ratio.level[::-1])

    # compute geopotential at half and full model levels
    geopot_half_level = ds.z + const.R_DRY * (
                 temp_virtual * np.log(pressure_ratio)).cumsum(dim='level')
    ds['geopotential'] = geopot_half_level + alpha * const.R_DRY * temp_virtual

    # add geopotential height to dataset
    ds['geopotential_height'] = ds['geopotential'] / const.G

    # Add metadata
    ds.geopotential.attrs['units'] = 'm**2 s**-2'
    ds.geopotential.attrs['long_name'] = 'Geopotential on full model levels'
    ds.geopotential_height.attrs['units'] = 'm'
    ds.geopotential_height.attrs['long_name'] = \
        'Geopotential height on full model levels'
    ds.pressure.attrs['units'] = 'Pa'
    ds.pressure.attrs['long_name'] = 'Pressure on full model levels'

    return ds
