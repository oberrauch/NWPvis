import os

from unittest import TestCase

from load_cut_nc_files import get_input_data


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
        pass
