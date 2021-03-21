#!/usr/bin/env python
# coding: utf-8

# Template of full workflow
# 
# STEPS:
# 1. Combine input files into one dataset (`data_import`)
# 2. Add geopotential and pressure to the dataset (`vertical_coordinates`)
# 3. Add all calculated variables (`calculations`)
# 4. Select time and slices/cross-sections (`data_import`)
# 5. Make desired plots for the selected cross-section of data (`plotting`)

# build ins
import os
import sys

# external libraries
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# local libraries
# sys.path.append('/Users/oberrauch/work/profiles/NWPvis')
import data_import
import vertical_coordinates
import calculations
import plotting

# define paths to input data
# dir_path = '../NWPvis/data/'
dir_path = '../data/'
ml_path = os.path.join(dir_path, 'ML.nc')  # model level data
lnsp_path = os.path.join(dir_path, 'LNSP.nc')  # log of surface pressure data
geopot_path = os.path.join(dir_path, 'TOPO.nc')  # surface geopotential data

# load files and combine into onde dataset
ds = data_import.get_input_data(path_model_level=ml_path,
                                path_lnsp=lnsp_path,
                                path_sfc_geopotential=geopot_path)

# define and compute slices along constant latitudes and longitudes
lats = [46.0, 47.3, 50.0]
lons = [5.5, 11.4, 13.3]

ds_slices = dict()
ds_slices['lats'] = data_import.slice_lat(ds, lats)
ds_slices['lons'] = data_import.slice_lon(ds, lons)

# compute vertical coordinates
for i in ds_slices:
    ds_slices[i] = vertical_coordinates.get_geopotential(ds_slices[i])
    ds_slices[i] = calculations.calculate_all_vars(ds_slices[i])


# select a single latitude and time step
data_out = ds_slices['lats'].sel(latitude=47.3, method='nearest').isel(time=2)

# plot temperature
fig_t, ax_t = plotting.Temperature_plot(data_out).make_figure()
plt.show()
