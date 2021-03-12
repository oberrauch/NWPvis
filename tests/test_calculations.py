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
        # TODO test against known values
        pass

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

        # TODO test against known values
        pass

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

        # TODO test against known values
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
        # create dummy data containing temperature [K], pressure [Pa], vertical
        # wind speed in pressure coordinates [Pa/s] and relative humidity [%]
        temp = 18. + constants.t_0
        p = 1000e2
        rh = 50.
        omega = 100e2 / (24 * 3600)  # 100hPa/day
        # combine into dataset
        ds = xr.Dataset(data_vars={'t': xr.DataArray(temp),
                                   'pressure': xr.DataArray(p),
                                   'rh': xr.DataArray(rh),
                                   'w': xr.DataArray(omega)})

        # convert vertical wind speed into height coordinate
        w_ms = calculations.w_from_omega(ds)

        # test against analytical solution
        es = 611.2 * np.exp(17.67 * (temp - constants.t_0) / (temp - 29.65))
        e = rh * es * 1e-2
        rho = (p - e) / (constants.Rd * temp) + e / (constants.Rvap * temp)
        w_ms_anal = omega / (-rho * constants.g)
        np.testing.assert_allclose(w_ms, w_ms_anal)
        # test against rule of thumb for lower troposphere 100 hPa/day = 1 cm/s
        np.testing.assert_allclose(w_ms, -0.01, atol=0.001, rtol=0.1)

    def test_T_lcl_from_T_rh(self):
        """Test calculation of the temperature at the lifting condensation
        level (LCL)."""
        # create dummy data containing temperature [K], relative humidity [%]
        temp = 18. + constants.t_0
        rh = 50.
        # combine into dataset
        ds = xr.Dataset(data_vars={'t': xr.DataArray(temp),
                                   'rh': xr.DataArray(rh)})
        # calculate temperature at LCL
        temp_lcl = calculations.T_lcl_from_T_rh(ds)

        # test against anayltical solution
        temp_lcl_anal = 1 / (1 / (temp - 55) + np.log(rh * 1e-2) / 2840) + 55
        np.testing.assert_allclose(temp_lcl, temp_lcl_anal)

        # TODO: test against know values
        pass
