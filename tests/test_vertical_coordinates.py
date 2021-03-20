"""TODO: add docstring"""
# import standard libraries
import os

# import
import numpy as np
import pandas as pd
import xarray as xr
from unittest import TestCase

# local imports
from data_import import get_input_data
from vertical_coordinates import get_model_level_definition, \
    get_pressure_and_alpha, get_geopotential
from constants import G, R_DRY
from calculations import virtual_temperature


class TestVerticalCoordinates(TestCase):

    def setUp(self):
        """Load test data."""

        # specify path to files
        dir_path = '../data/'
        model_level_path = os.path.join(dir_path, 'ML.nc')
        sfc_lnsp_path = os.path.join(dir_path, 'SFC_LNSP.nc')
        sfc_geopot_path = os.path.join(dir_path, 'TOPO.nc')

        # load all model data and combine into one dataset
        self.ds = get_input_data(path_model_level=model_level_path,
                                 path_lnsp=sfc_lnsp_path,
                                 path_sfc_geopotential=sfc_geopot_path)

        # Get definitions of model levels
        self.level_coef = get_model_level_definition()
        # Get pressure and terms depending on pressure
        p = get_pressure_and_alpha(self.ds)
        self.pressure, self.alpha, self.pressure_ratio = p

    def tearDown(self):
        """Close dataset."""
        self.ds.close()

    def test_get_model_level_definition(self):
        """Test"""
        # get test file
        path = '../data/test_data/test_ecmwf_ab_coeffs.nc'
        test_level_coef = xr.load_dataset(path)

        # remove normal file from test data directory if present
        dir_path = '../data/test_data/'
        path = os.path.join(dir_path, 'ecmwf_ab_coeffs.nc')
        if os.path.isfile(path):
            os.remove(path)

        # test downloading file
        level_coef = get_model_level_definition(dir_path)
        xr.testing.assert_allclose(level_coef, test_level_coef)

        # test reading file
        level_coef = get_model_level_definition(dir_path)
        xr.testing.assert_allclose(level_coef, test_level_coef)

        # remove normal file from test data directory if present
        dir_path = '../data/test_data/'
        path = os.path.join(dir_path, 'ecmwf_ab_coeffs.nc')
        if os.path.isfile(path):
            os.remove(path)

    def test_get_pressure_and_alpha(self):
        """Test computation of pressure and pressure related terms."""
        # get test dataset and test coefficients
        path = '../data/test_data/test_ds.nc'
        test_ds = xr.load_dataset(path)
        path = '../data/test_data/test_ecmwf_ab_coeffs.nc'
        test_level_coef = xr.load_dataset(path)

        # compute pressure, pressure ratio and alpha from test data
        sfc_pressure = np.exp(test_ds['lnsp'])
        p_half_levels = test_level_coef.A + test_level_coef.B * sfc_pressure
        p_plus = p_half_levels.drop_sel(level=0)
        p_minus = p_plus.shift(level=1)
        test_pressure = 0.5 * (p_plus + p_minus)
        test_pressure_ratio = (p_plus / p_minus)
        delta_p = p_half_levels.diff(dim='level')
        test_alpha = 1 - (p_minus / delta_p
                          * np.log(test_pressure_ratio))
        test_alpha.loc[dict(level=1)] = np.log(2)

        # test against computed values
        np.testing.assert_allclose(self.pressure, test_pressure)
        np.testing.assert_allclose(self.pressure_ratio, test_pressure_ratio)
        np.testing.assert_allclose(self.alpha, test_alpha)

    def test_get_geopotential(self):
        """Test the calculation of geopotential."""

        # get test dataset and test coefficients
        path = '../data/test_data/test_ds.nc'
        test_ds = xr.load_dataset(path)

        # compute geopotential following the ECMWF's equations
        temp_virtual = virtual_temperature(test_ds)
        pressure, alpha, pressure_ratio = get_pressure_and_alpha(test_ds)
        temp_virtual = temp_virtual.reindex(level=temp_virtual.level[::-1])
        pressure_ratio = pressure_ratio.reindex(
            level=pressure_ratio.level[::-1])
        geopot_half_level = test_ds.z + R_DRY * (
                temp_virtual * np.log(pressure_ratio)).cumsum(dim='level')
        geopot_half_level = geopot_half_level.reindex(
            level=geopot_half_level.level[::-1])
        test_geopotential = geopot_half_level + alpha * R_DRY * temp_virtual
        test_geopotential_height = test_geopotential / G

        # compute geopotential using the module's function
        ds = get_geopotential(self.ds)

        # compare
        xr.testing.assert_equal(ds['geopotential'], test_geopotential)
        xr.testing.assert_equal(ds['geopotential_height'],
                                test_geopotential_height)
