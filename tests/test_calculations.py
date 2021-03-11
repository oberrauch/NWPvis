"""TODO: add docstring"""
# import standard libraries
import os

# import
import numpy as np
import pandas as pd
import xarray as xr
import itertools
from unittest import TestCase

# local imports
from load_cut_nc_files import get_input_data
import constants
import calculations


class TestCalculations(TestCase):

    def setUp(self):
        """Load test data."""

        # specify path to files
        dir_path = '../data/'
        model_level_path = os.path.join(dir_path, 'ML.nc')
        sfc_lnsp_path = os.path.join(dir_path, 'SFC_LNSP.nc')
        sfc_geopot_path = os.path.join(dir_path, 'TOPO.nc')

        # load all model data and combine into one dataset
        self.ds = get_input_data(filename_model_level=model_level_path,
                                 filename_sfc_lnsp=sfc_lnsp_path,
                                 filename_sfc_geopotential=sfc_geopot_path)

    def tearDown(self):
        """Close dataset."""
        self.ds.close()

    def test_theta_from_t_p(self):
        """Test calculation of potential temperature."""

        # create dummy data containing temperature [K] and pressure [Pa]
        temp = 15. + constants.t_0
        p = 1025.0e2
        # combine into dataset
        ds = xr.Dataset(data_vars={'t': xr.DataArray(temp),
                                   'pressure': xr.DataArray(p)})
        # compute potential temperature
        theta = calculations.theta_from_t_p(ds)

        # test against analytic solution
        theta_anal = temp * (constants.p0 / p) ** (
                constants.Rd / constants.c_p)
        np.testing.assert_allclose(theta, theta_anal)

    def test_N_dry_from_p_theta(self):
        # TODO
        raise NotImplementedError

    def test_windspeed(self):
        """Test calculation of total wind speed from wind components."""

        # create dummy data
        u = np.array([i[0] for i in itertools.product([0, 1, -1], repeat=3)])
        v = np.array([i[1] for i in itertools.product([0, 1, -1], repeat=3)])
        w = np.array([i[2] for i in itertools.product([0, 1, -1], repeat=3)])
        # combine into dataset
        ds = xr.Dataset(data_vars={'u': xr.DataArray(u),
                                   'v': xr.DataArray(v),
                                   'w_ms': xr.DataArray(w)})
        # compute windspeed
        wspd = calculations.windspeed(ds)

        # test against analytic solution
        wspd_anal = np.sqrt((u ** 2 + v ** 2 + w ** 2))
        np.testing.assert_allclose(wspd, wspd_anal)

        # test without vertical wind speed
        # TODO: test computation of vertical wind speed ?!
        # TODO: add omega to dataset
        ds = xr.Dataset(data_vars={'u': xr.DataArray(u),
                                   'v': xr.DataArray(v)})

    def test_es_from_t(self):
        """Test calculations of saturation water vapor pressure."""
        # create dummy data containing temperature [K]
        temp = 15. + constants.t_0
        # combine into dataset
        ds = xr.Dataset(data_vars={'t': xr.DataArray(temp)})
        # compute saturation water vapor pressure
        es = calculations.es_from_t(ds)

        # test against analytic solution
        es_anal = 611.2 * np.exp(17.67 * (temp - constants.t_0)
                                 / (temp - 29.65))
        np.testing.assert_allclose(es, es_anal)
        # TODO: test with demo file, compare results to fixed values ?!
        pass

    def test_rh_from_t_q_p(self):
        """Test calculation of relative humidity

        The values and results are taken from Exercise 3.8 in [Hobbs 2006]_.
        """
        # create dummy data containing temperature [K], pressure [Pa] and
        # mixing ratio
        temp = 18. + constants.t_0
        p = 1000e2
        q = 6e-3
        # combine into dataset
        ds = xr.Dataset(data_vars={'t': xr.DataArray(temp),
                                   'pressure': xr.DataArray(p),
                                   'q': xr.DataArray(q)})
        # compute temperature
        rh = calculations.rh_from_t_q_p(ds)

        # test against analytic solution
        es = 611.2 * np.exp(17.67 * (temp - constants.t_0)
                            / (temp - 29.65))
        qs = 0.622 * (es / (p - es))
        rh_anal = 100 * q / qs
        np.testing.assert_allclose(rh, rh_anal)
        # test against solution from Hobbs (2006)
        np.testing.assert_allclose(rh, 46, atol=0.5, rtol=0.1)

    def test_w_from_omega(self):
        """Test conversion of vertical wind speed with respect to pressure
        into that with respect to height."""
        raise NotImplementedError

    def test_T_lcl_from_T_rh(self):

        pass
