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
from data_import import get_input_data, slice_lat, slice_lon, slice_diag


class TestCalculations(TestCase):

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

        # define arbitrary coordinates, as dictionary
        self.ibk_coords = {'lon': 11.4041, 'lat': 47.2692}

    def tearDown(self):
        """Close dataset."""
        self.ds.close()

    def test_get_input_data(self):
        """Test data file imports, only the three file method."""
        # test loaded dataset against know dataset
        ds_test = xr.load_dataset('../data/test_data/test_ds.nc')
        assert self.ds.equals(ds_test)

        # close datasets
        ds_test.close()

    def test_slice_lat(self):
        """Test data slicing along fixed latitudes."""
        # slice data along fixed latitudes
        lats = np.array([self.ibk_coords['lat']])
        lat_ds = slice_lat(self.ds, lats)

        # test if given latitude was selected
        print(lat_ds.latitude, lats)
        np.testing.assert_allclose(lat_ds.latitude, lats, atol=0.05)
        # TODO: test against test dataset, look at attributes first

    def test_slice_lon(self):
        """Test data slicing along fixed longitudes."""
        # slice data along fixed longitude
        lons = np.array([self.ibk_coords['lon']])
        lon_ds = slice_lon(self.ds, lons)

        # test if given latitude was selected
        np.testing.assert_allclose(lon_ds.longitude, lons, atol=0.05)
        # TODO: test against test dataset, look at attributes first

    def test_slice_diag(self):
        """Test diagonal data slicing."""
        raise NotImplementedError

