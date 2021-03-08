import os

import numpy as np
import pandas as pd
import xarray as xr
import itertools
from unittest import TestCase

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
        """Test the calculation of potential temperature (theta)."""

        # create dummy data
        temp = 15. + constants.t_0
        p = 1025.0e2
        # combine into dataset
        ds = xr.Dataset(data_vars={'t': xr.DataArray(temp),
                                   'pressure': xr.DataArray(p)})
        theta = calculations.theta_from_t_p(ds)
        # test against analytic solution
        theta_anal = temp * (constants.p0 / p) ** (
                constants.Rd / constants.c_p)
        np.testing.assert_allclose(theta, theta_anal)

        # test with demo file, compare results to fixed values ?!
        pass

    def test_windspeed(self):
        """..."""
        # create dummy data
        u = np.array([i[0] for i in itertools.product([0, 1, -1], repeat=3)])
        v = np.array([i[1] for i in itertools.product([0, 1, -1], repeat=3)])
        w = np.array([i[2] for i in itertools.product([0, 1, -1], repeat=3)])
        # combine into dataset
        ds = xr.Dataset(data_vars={'u': xr.DataArray(u),
                                   'v': xr.DataArray(v),
                                   'w_ms': xr.DataArray(w)})
        wspd = calculations.windspeed(ds)
        # test against analytic solution
        wspd_anal = np.sqrt((u ** 2 + v ** 2 + w ** 2))
        np.testing.assert_allclose(wspd, wspd_anal)

        # test with demo file, compare results to fixed values ?!
        pass
