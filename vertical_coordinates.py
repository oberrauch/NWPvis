"""Vertical coordinates

This module calculates pressure, geopotential and geopotential height on full
and half model levels.




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

Author(s): Alzbeta Medvedova, Moritz Oberrauch (Based on the script of Johannes
Horak and Deborah Morgenstern)

References
----------
 .. [ECMWF]: European Centre for Medium-Range Weather Forecasts, 2015, "IFS
    DOCUMENTATION – Cy41r1 Operational implementation 12 May 2015, PART III:
    DYNAMICS AND NUMERICAL PROCEDURES", Shinfield Park, Reading, RG2 9AX,
    England, URL: https://www.ecmwf.int/node/9210, DOI: 10.21957/a3hkssbw

"""

# built ins
import os

# external libraries
import pandas as pd
import xarray as xr
import numpy as np
import requests

# local dependencies
import constants as const
import calculations


def get_model_level_definition(dir_path=None):
    """Definition of model levels

    TODO: decide if not just keeping this file and removing download routine

    Function loading the coefficients a, b needed for calculating pressure
    at half levels. If the file containing `ecmwf_ab_coeffs.nc` the
    coefficients does not exist yet, it will be downloaded and stored.
    The directory where the file should be (or will be saved) can be provided,
    default is ./data/

    Parameters
    ----------
    dir_path : str, optional
        path to directory where the file should be (or will be saved),
        defaults to ./data/

    Returns
    -------
    xr.DataArray

    """

    # specify path and filename
    if not dir_path:
        # use default path to data directory if not provided
        package_dir, _ = os.path.split(__file__)
        dir_path = os.path.join(package_dir, './data/')
    file_name = 'ecmwf_ab_coeffs.nc'
    path = os.path.join(dir_path, file_name)

    if not os.path.isfile(path):
        # download file if necessary
        url = 'https://www.ecmwf.int/en/forecasts/' \
              'documentation-and-support/137-model-levels'
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


def get_pressure_and_alpha(data, dir_path=None, inplace=True):
    """Pressure calculations

    Calculate pressure, alpha on full model levels based on Eqns. 2.11, 2.23
    from the [ECMWF]_ documentation.

    Parameters
    ----------
    data : xr.Dataset
        dataset containing the logarithmic surface pressure (data.lnsp)

    dir_path : str, optional
        path to directory where the file should be (or will be saved),
        defaults to ./data/

    inplace : bool, optional, default=True
        If True, the calculated pressure (but not alpha and pressure ratios) is
        added inplace to the given dataset, otherwise not.

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
    level_coef = get_model_level_definition(dir_path)
    # Get sfc pressure from logarithmic surface pressure
    sfc_pressure = np.exp(data['lnsp'])

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

    # add to dataset
    if inplace:
        data['pressure'] = pressure

    return pressure, alpha, pressure_ratio


def get_geopotential(data, dir_path=None, inplace=True):
    """Geopotential calculations

    Calculates pressure, geopotential, and geopotential height at all full
    model levels following the [ECMWF]_ documentation (Eq. 2.22). Coefficients
    :math:`a`, :math:`b` needed for conversion are either loaded from a *.nc
    file or downloaded in the process.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the following variables for the dimensions time,
        height (level) and longitude/latitude:
        - logarithmic surface pressure (data.lnsp)
        - temperature (data.t)
        - mixing ratio (data.q)
        - surface geopotential (data.z)

    dir_path : str, optional
        path to directory where the file should be (or will be saved),
        defaults to ./data/

    inplace : bool, optional, default=True
        If True, the calculated geopotential and geopotential height are added
        inplace to the given dataset, otherwise not.

    Returns
    -------
    data : xr.Dataset
        Dataset containing pressure, geopotential and geopotential height
        at all vertical levels in addition to the other variables


    Notes
    -----
    Equiation 2.22 of the [EMCWF]_ documentation describes the computation of
    the full level geopotential :math:`\phi_k` as a function of the half level
    geopotential :math:`\phi_{k+1/2}`, the coefficient :math:`\alpha_k` (see
    `get_pressure_and_alpha`), the gas constant for dry air
    :math:`R_\text{dry}` and the virutal tempertaure :math:`T_{v,k}`
    (see the `calculations` module) as

    .. math:: \phi_k = \phi_{k+1/2} + \alpha_k R_\text{dry}T_{v,k}

    See Also
    --------
    get_pressure_and_alpha : TODO
    calculations : TODO


    """

    # Compute needed variables, add pressure dataset if inplace
    temp_virtual = calculations.virtual_temperature(data)
    pressure, alpha, pressure_ratio = get_pressure_and_alpha(data,
                                                             dir_path,
                                                             inplace)
    # reverse virtual temperature and pressure ratio along level coordinates
    # in order to use the cumulative sum function on the dataset
    temp_virtual = temp_virtual.reindex(level=temp_virtual.level[::-1])
    pressure_ratio = pressure_ratio.reindex(level=pressure_ratio.level[::-1])

    # compute geopotential at half levels
    geopot_half_level = data.z + const.R_DRY * (
            temp_virtual * np.log(pressure_ratio)).cumsum(dim='level')
    # reverse level coordinates again for consistency
    geopot_half_level = geopot_half_level.reindex(
        level=geopot_half_level.level[::-1]).transpose()
    temp_virtual = temp_virtual.reindex(level=temp_virtual.level[::-1])

    if inplace:
        # create new pointer to the given dataset
        data_new = data
    else:
        # create a deep copy of the given dataset and add pressure
        data_new = data.copy(deep=True)
        data_new['pressure'] = pressure

    # compute geopotential at full model levels and add to dataset
    data_new['geopotential'] = (geopot_half_level
                                + alpha * const.R_DRY * temp_virtual)

    # add geopotential height to dataset
    data_new['geopotential_height'] = data_new['geopotential'] / const.G

    # Add metadata
    data_new.geopotential.attrs['units'] = 'm**2 s**-2'
    data_new.geopotential.attrs['long_name'] = 'Geopotential on full model ' \
                                               'levels'
    data_new.geopotential_height.attrs['units'] = 'm'
    data_new.geopotential_height.attrs['long_name'] = 'Geopotential height' \
                                                      'on full model levels'
    data_new.pressure.attrs['units'] = 'Pa'
    data_new.pressure.attrs['long_name'] = 'Pressure on full model levels'

    return data_new
