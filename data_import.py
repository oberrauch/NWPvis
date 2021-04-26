"""Import data

TODO: finish module docstring

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

import numpy as np
import pandas as pd
import xarray as xr
from itertools import repeat
from geopy import distance as gdistance
from geographiclib import geodesic


def _lat_lon_to_distance(ds):
    """Distance between longitude/latitude coordinates.

    Computes and returns geodesic distance in kilometers between the
    longitude/latitude coordinates of the given dataset. Dataset can be a slice
    along longitudes, latitudes or diagonally, i.e., latitudes and longitudes
    must be either be of same length or constant. Hence this should/can be used
    only after slicing the data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced

    Returns
    -------
    float array
        Array containing the cumulative geodesic distance along the coordinate
        slice in kilometers

    """
    # create array of lat/lon pairs
    lats = ds.latitude.values
    lons = ds.longitude.values
    if lats.size == lons.size:
        points = np.columnstack((lats, lons))
    elif lats.size == 1:
        points = np.columnstack((repeat(lats), lons))
    elif lons.size == 1:
        points = np.columnstack((lats, repeat(lons)))
    else:
        raise ValueError(
            "Latitudes and longitudes must be either be of same length "
            "or constant")

    # iteratively compute geodesic distance between subsequent points
    # Note: distance along the great circle is alternatively possible
    distance_m = list()
    distance_m.append(0)
    for p0, p1 in zip(points[:-1], points[1:]):
        distance_m.append(gdistance.geodesic(p0, p1).m)
    # compute cumulative distance
    distance_m = np.cumsum(distance_m)

    return distance_m


def get_input_data(path_sfc_geopotential=None,
                   path_lnsp=None,
                   path_model_level=None,
                   path_all_data=None):
    """Read data from file.

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
        # add/remove/change attributes
        data.z.attrs['long_name'] = 'Surface geopotential'
        data.z.attrs['standard_name'] = 'sfc_geopotential'
        data.lnsp.attrs['units'] = 'ln(Pa)'
        data.lnsp.attrs['standard_name'] = 'ln_sfc_p'

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

    # compute distance between coordinate slices and add to dataset
    distance_m = _lat_lon_to_distance(ds_lat)
    ds_lat.attrs['distance_m'] = distance_m

    # use distance along slice as x axis
    ds_lat.attrs['x_axis'] = distance_m * 1e-3
    # x-mesh by stacking the distances for each level above each other
    ds_lat.attrs['x_mesh'] = np.tile(distance_m * 1e-3, (len(ds.level), 1))
    # specify x-axis label and title
    ds_lat.attrs['xlab'] = 'Distance [km]'
    ds_lat.attrs['title'] = 'Cross-section: {} along {:.1f}°N\n'

    # specify whether the slice is along longitudes/latitudes or diagonal
    ds_lat.attrs['cross_section_style'] = 'straight'

    # specify parallel and normal wind
    ds_lat['parallel_wind'] = ds_lat.u
    # S/N wind - minus sign to make southerly (out of page) positive:
    # TODO: does this hold true for southern hemisphere?!
    # TODO: right hand coordinate system
    ds_lat['normal_wind'] = -1 * ds_lat.v

    return ds_lat


def slice_lon(ds, lons, tolerance=0.05):
    """Slice along longitudes.

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

    # compute distance between coordinate slices and add to dataset
    distance_m = _lat_lon_to_distance(ds_lon)
    ds_lon.attrs['distance_m'] = distance_m

    # use distance along slice as x axis
    ds_lon.attrs['x_axis'] = distance_m * 1e-3
    # x-mesh by stacking the distances for each level above each other
    ds_lon.attrs['x_mesh'] = np.tile(distance_m * 1e-3, (len(ds.level), 1))
    # specify x-axis label and title
    ds_lon.attrs['xlab'] = 'Distance [km]'
    ds_lon.attrs['title'] = 'Cross-section: {} along {:.1f}°E\n'

    # specify whether the slice is along longitudes/latitudes or diagonal
    ds_lon.attrs['cross_section_style'] = 'straight'

    # specify parallel and normal wind
    # TODO: right hand coordinate system
    ds_lon['parallel_wind'] = ds_lon.v
    ds_lon['normal_wind'] = ds_lon.u

    return ds_lon


def slice_diag(ds, lat1, lon1, lat2, lon2, res_km=None):
    """
    TODO: docstring
    TODO: rename to slice, since it can be along constant lon/lats as well

    Selects a slice of data from the dataset along a diagonal defined by two
    points - their longitude and latitude

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be sliced
    lat1, lon1, lat2, lon2 : float
        lat/lon coordinates of the first (1) and last (2) point of the
        diagonal cross-section
    res_km : float, optional
        Grid spacing along the new cross section in kilometers. Default is to
        match the original dataset

    Returns
    -------
    ds : xr.Dataset
        a diagonal cross-section with a new dimension in the data

    """

    if not res_km:
        # compute resolution in kilometers based on the resolution of the given
        # dataset and the northern/southern-most latitude of the slice
        sign_digits = 2
        dlon = abs(min(ds.longitude.diff(dim='longitude').values))
        dlon = round(dlon, sign_digits - (int(np.floor(np.log10(dlon))) - 1))
        dlat = abs(min(ds.latitude.diff(dim='latitude').values))
        dlat = round(dlat, sign_digits - (int(np.floor(np.log10(dlat))) - 1))

        max_lat = np.deg2rad(max(ds.latitude.values))
        radius_earth = 6371
        res_km = 2 * np.pi * radius_earth * min(dlat, dlon) / 360 * \
                 np.cos(max_lat)

    # compute the geodesic line (and distance) on the WGS84 ellipsoid
    line = geodesic.Geodesic.WGS84.InverseLine(lat1=lat1, lon1=lon1,
                                               lat2=lat2, lon2=lon2)
    total_distance_m = line.s13

    # define distance along slice as index
    distance_m = np.arange(0, total_distance_m, res_km * 1e3)
    distance_m = np.append(distance_m, total_distance_m)
    # compute coordinate points and azimuth angle along the geodesic line with
    # with the given distance from each other, convert into Dataset
    x_axis = pd.DataFrame([[line.Position(d)['lat2'], line.Position(d)['lon2'],
                            line.Position(d)['azi2']] for d in distance_m],
                          index=distance_m / 1e3,
                          columns=['lat', 'lon', 'azi'])
    x_axis.index.name = 'diag'
    x_axis = x_axis.to_xarray()

    # Interpolate along the defined line. Passing DataArrays as the new
    # coordinate, makes the interpolation use their dimension for broadcasting
    ds_diag = ds.interp(latitude=x_axis.lat, longitude=x_axis.lon)

    # compute distance between coordinate slices and add to dataset
    ds_diag['distance_m'] = x_axis.diag

    # add metadata: x-axis properties and title
    ds_diag.attrs['x_axis'] = distance_m * 1e-3
    ds_diag.attrs['x_mesh'] = np.tile(distance_m * 1e-3, (len(ds.level), 1))
    ds_diag.attrs['xlab'] = 'Distance [km]'
    ds_diag.attrs['title'] = 'View from south: diagonal cross-section of {}\n'

    # used for filling the title text later
    ds_diag.attrs['cross_section_style'] = 'diagonal'

    # Compute parallel and normal wind for diagonal slice by rotating the axis.
    # Hence the azimuth angle must be converted into rotation angle
    rot_angle = np.deg2rad(90) - x_axis.azi
    ds_diag['parallel_wind'] = (ds_diag.u * np.cos(rot_angle) +
                                ds_diag.v * np.sin(rot_angle))
    ds_diag['normal_wind'] = (-ds_diag.u * np.sin(rot_angle) +
                              ds_diag.v * np.cos(rot_angle))

    return ds_diag
