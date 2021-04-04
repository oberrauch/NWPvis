"""Import data

This module has two functions:
1. Load all the needed *.nc files, combine them into one dataset.
   This makes it easier to work with the data later.
   Inputs: ML_DATA, SFC_LNSP, GEOPOTENTIAL_DATA
                   paths to netCDF4 files containing:
                   1. temperature and humidity on
                   model levels (contains other variables
                   as well)
                   2. logarithm of surface pressure
                   3. surface geopotential
2. Select slices of data to be visualized:
   - along constant latitude
   - along constant longitude
   - along a defined diagonal
   When slicing, some metadata is added to the created slices

Author(s): Alzbeta Medvedova, Moritz Oberrauch



"""

# external libraries
import xarray as xr
import numpy as np

# local imports
from calculations import angle, diag_wind


def get_input_data(path_sfc_geopotential=None,
                   path_lnsp=None,
                   path_model_level=None,
                   path_all_data=None):
    """

    Loading the given *.nc files needed for the visualization of vertical
    cross-sections and combining them into one dataset.

    The data file(s) must at least contain the following variables at all model
    levels:
    - surface geopotential (z)
    - logarithm of surface pressure (lnsp)
    - temperature (t)
    - humidity/mixing ratio (q)

    This data can either be contained in one file (all_data) or three separate
    files (sfc_geopotential, lnsp, model_level).

    Parameters
    ----------
    path_model_level : str, optional
        Path to the *.nc file containing data on model levels.
    path_lnsp : str, optional
        Path to the *.nc file containing log of sfc pressure.
    path_sfc_geopotential : str, optional
        Path to the *.nc file containing surface geopotential.
    path_all_data : str, optional
        Path to the *.nc file containing all data, overrides other paths.
        This is basically an artifact from the original version of the code
        (J. Horak + D. Morgenstern) and will be removed if the data is always
        going to be provided in three files

    Raises
    ------
    ValueError
        If files are not provided correctly, i.e., either one file with all
        needed data or three separate files

    Returns
    -------
    data : xr.Dataset
        combined dataset containing all variables needed for
        further calculations
    """

    if path_all_data is not None:
        # if one file with all data is provided, load this file
        data = xr.load_dataset(path_all_data)

    elif any(x is None for x in [path_model_level,
                                 path_lnsp,
                                 path_sfc_geopotential]):
        # raise an error if not all necessary files are provided
        raise ValueError("Provide path to all necessary data files.")

    else:
        # load all three files
        model_level = xr.load_dataset(path_model_level)
        lnsp = xr.load_dataset(path_lnsp)
        sfc_geopotential = xr.load_dataset(path_sfc_geopotential)
        # the model topography does not change, hence only one (the first) time
        # step is necessary
        z = sfc_geopotential.isel(time=0)
        # combine them into one dataset
        data = xr.merge([model_level, lnsp, z], join="exact")

    return data


def slice_lat(ds, lats, tolerance=0.05):
    """Slice along latitudes.

    Selects data from the dataset along given lines of constant latitude and
    add properties for plots. The selection allows is done using the nearest
    neighbor lookup, with the given tolerance.

    Additional parameters needed later for plotting, like the mesh grid, the
    axes labels, etc., are computed and/or defined here and added to the
    dataset as attributes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced
    lats : array-like
        List of latitudes as float
    tolerance : float, optional, default=0.05
        Maximum distance between specified latitudes and valid values

    Returns
    -------
    ds : xr.Dataset
        Subset of input data with added plotting attributes

    """
    # subset dataset
    ds_lat = ds.sel(latitude=lats,
                    method='nearest',
                    tolerance=tolerance).copy(deep=True)

    # x-mesh by stacking the longitudes for each level above each other
    ds_lat.attrs['x_mesh'] = np.tile(ds.longitude, (len(ds.level), 1))
    # use longitude values as x-ticks and labels
    ds_lat.attrs['x_axis'] = ds.longitude
    ds_lat.attrs['x_ticklabels'] = ds.longitude
    # define x-axis label and title
    ds_lat.attrs['xlab'] = 'Longitude [°E]'
    ds_lat.attrs['title'] = 'Cross-section: {} along {:.1f}°N\n'

    # specify whether the slice is along longitudes/latitudes or diagonal
    ds_lat.attrs['cross_section_style'] = 'straight'

    # specify transect and perpendicular wind
    ds_lat['transect_wind'] = ds_lat.u
    # S/N wind - minus sign to make southerly (out of page) positive:
    # TODO: does this hold true for southern hemisphere?!
    ds_lat['perp_wind'] = -1 * ds_lat.v

    return ds_lat


def slice_lon(ds, lons, tolerance=0.05):
    """Slice along longitudes

    Selects data from the dataset along given lines of constant longitudes and
    add properties for plots. The selection allows is done using the nearest
    neighbor lookup, with the given tolerance.

    Additional parameters needed later for plotting, like the mesh grid, the
    axes labels, etc., are computed and/or defined here and added to the
    dataset as attributes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced
    lons : array-like
        List of latitudes as float
    tolerance : float, optional, default=0.05
        Maximum distance between specified longitudes and valid values

    Returns
    -------
    ds : xr.Dataset
        Subset of input data with added plotting attributes

    """
    # subset dataset
    ds_lon = ds.sel(longitude=lons,
                    method='nearest',
                    tolerance=tolerance).copy(deep=True)

    # x-mesh by stacking the latitudes for each level above each other
    ds_lon.attrs['x_mesh'] = np.tile(ds.latitude, (len(ds.level), 1))
    # use latitude values as x-ticks and labels
    ds_lon.attrs['x_axis'] = ds.latitude
    ds_lon.attrs['x_ticklabels'] = ds.latitude
    # define x-axis label and title
    ds_lon.attrs['xlab'] = 'Latitude [°N]'
    ds_lon.attrs['title'] = 'Cross-section: {} along {:.1f}°E\n'

    # specify whether the slice is along longitudes/latitudes or diagonal
    ds_lon.attrs['cross_section_style'] = 'straight'

    # specify transect and perpendicular wind
    ds_lon['transect_wind'] = ds_lon.v
    ds_lon['perp_wind'] = ds_lon.u

    return ds_lon


def slice_diag(ds, lon0, lat0, lon1, lat1):
    """
    TODO: not look at yet

    Selects a slice of data from the dataset along a diagonal defined by two
    points - their longitude and latitude

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced
    lon0, lat0, lon1, lat1 : float
        lat/lon coordinates of the first (0) and last (1) point of the
        diagonal cross-section

    Returns
    -------
    ds : xr.Dataset
        a diagonal cross-section with a new dimension in the data

    """
    num = 100  # number of horizontal points for the diagonal cross-section

    # Make sure that longitude in the cross-sections always increases
    #   (if it doesn't, exchange starting points)
    # This is necessary for correct re-mapping of winds and plotting, and
    #   it also limits cross-section bearing angle from 0 to 180 deg
    if lon1 < lon0:
        [lon1, lon0] = [lon0, lon1]
        [lat1, lat0] = [lat0, lat1]

    # Define cross-section locations: arrays need the same number of points!
    lat_np = np.linspace(lat0, lat1, num)
    lon_np = np.linspace(lon0, lon1, num)

    # Get an array lat/lon pairs to be used as plot labels
    latlon_pairs = np.column_stack((lat_np, lon_np))
    xlabels = list(map(tuple, np.round(latlon_pairs, 2)))

    # Get cross-section locations as xr.DataArrays: used for interpolation
    lat_xr = xr.DataArray(lat_np, dims='diag')
    lon_xr = xr.DataArray(lon_np, dims='diag')

    # Interpolate along the defined line
    ds_diag = ds.interp(latitude=lat_xr, longitude=lon_xr).copy()

    # Index for plotting
    idx = np.linspace(0, num, num)

    # add metadata: x-axis properties and title
    ds_diag.attrs['x_axis'] = idx
    ds_diag.attrs['x_ticklabels'] = xlabels
    ds_diag.attrs['x_mesh'] = np.tile(idx, (len(ds.level), 1))
    ds_diag.attrs['xlab'] = '(Latitude [°N], Longitude [°E])'
    ds_diag.attrs['title'] = 'View from south: diagonal cross-section of {}\n'

    # used for filling the title text later
    ds_diag.attrs['cross_section_style'] = 'diagonal'

    # determine which wind is transect or perpendicular
    diag_angle = angle(lon0, lat0, lon1, lat1)

    tw, pw = diag_wind(ds_diag.u, ds_diag.v, diag_angle)
    ds_diag['transect_wind'] = tw
    ds_diag['perp_wind'] = pw

    return ds_diag
