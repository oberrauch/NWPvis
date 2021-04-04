#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#
# Author : Alzbeta Medvedova
#
# This script takes netCDF files as input and produces plots of vertical
# cross-sections of multiple atmospheric variables relevant for weather
# forecasting.
#
# STEPS:
# 1. Combine input files into one dataset (load_cut_nc_files)
# 2. Add geopotential and pressure to the dataset (geopotential_calculation)
# 3. Add all calculated variables (calculations)
# 4. Select time and slices/cross-sections (load_cut_nc_files)
# 5. Make desired plots for the selected cross-section of data (plotting)
#
#
# TODO LIST:
# - re-write the geopotential calculation code to work with selected slices
#   instead of whole 4D fields: for speed-up of the code
# - write tests to see if everything works, run those when changes are made
# - possibly rewrite this script to take also slice definition / time as a
#   command line argument, not only file names? If not, pre-define which time
#   steps we want to plot. It takes too long to calculate the data for diagonal
#   cross-sections for all the time steps otherwise.
# - write checks if input files are correct and contain the expected variables
# - change the name of saved files: include info on which slice it is
#
##############################################################################


import os
import sys

# import: local dependencies
from data_import import get_input_data
from vertical_coordinates import get_geopotential
from calculations import calculate_all_vars
from data_import import slice_lat, slice_lon, slice_diag

# definitions of plotting functions/classes
from plotting import WindProfilePlot, TemperatureProfilePlot, RhProfilePlot, \
                     StabilityProfilePlot, PrecipitationProfilePlot
from plotting import plot_topography


# bool option to save all the produced figures (now used for )
savefigs = False

# %% Load all data, get geopotential

# parse path to files - command line arguments
ML_DATA = sys.argv[1]               # model level data
SFC_LNSP = sys.argv[2]              # log of surface pressure data
GEOPOTENTIAL_DATA = sys.argv[3]     # surface geopotential data

path_figs = sys.argv[4]             # path to save figures

# load all model data, combine into one dataset for convenience
ds = get_input_data(path_model_level=ML_DATA,
                    path_lnsp=SFC_LNSP,
                    path_sfc_geopotential=GEOPOTENTIAL_DATA)

# Calculate geopotential and pressure: do this BEFORE choosing slices!
ds = get_geopotential(os.getcwd(), ds)


# %% Choose data slices: add all variables to them

# slices along chosen latitudes/longitudes
#   THESE ARE EXAMPLE VALUES: to be determined which cross-sections are going
#   to be "standardized" for the weather briefing
ds_lat = slice_lat(ds, [46.0, 47.3, 48.0, 50.0])
ds_lon = slice_lon(ds, [5.5, 6.6, 7.6, 11.4, 12.4, 13.3])

# Add calculated variables to the data set
ds_lat = calculate_all_vars(ds_lat)
ds_lon = calculate_all_vars(ds_lon)

# %% ProfilePlot top view of where the cross-sections are now
# this is useful for visualizing where the cross-sections are
# values of the plotted lines must be chosen directly in the function

plot_topo = False
if plot_topo:
    fig, ax = plot_topography(ds)


# %% Trial cross-section plotting: EXAMPLE CASES
# uncomment whatever line(s) to get different examples

# # nice example for an example of water clouds
# data_out = ds_lon.sel(longitude=12.4, method='nearest').isel(time=38)

# # nice example for winds/isentropes
# data_out = ds_lon.sel(longitude=12.4, method='nearest').isel(time=26)

# # max ciwc of the dataset in this slice
# data_out = ds_lon.sel(longitude=5.5, method='nearest').isel(time=38)

# # max crwc, cswc of the dataset in this slice, also ok for ice clouds
# data_out = ds_lon.sel(longitude=6.6, method='nearest').isel(time=39)

# nice for wind
# data_out = ds_lon.sel(longitude=5.5, method='nearest').isel(time=36)
# data_out = ds_lat.sel(latitude=47.3, method='nearest').isel(time=36)

# also nice for water clouds
data_out = ds_lat.sel(latitude=47.3, method='nearest').isel(time=2)

# # diagonal cross-section
# data_out = ds_diag_1.isel(time=26)

# # plotting along diagonals takes time because of interpolation:
# # select time step BEFORE calling slice_diag
# ds_diag = slice_diag(ds.isel(time=36), 5.5, 47.3, 17.5, 47.3)
# data_out = calculate_all_vars(ds_diag)

# ds_diag = slice_diag(ds.isel(time=36), 5.5, 46.0, 17.5, 55.0)
# data_out_2 = calculate_all_vars(ds_diag)
# WindProfilePlot(data_out_2).make_figure()


fig_t, ax_t = TemperatureProfilePlot(data_out).make_figure()
fig_wind, ax_wind = WindProfilePlot(data_out).make_figure()
fig_rh, ax_rh = RhProfilePlot(data_out).make_figure()
# fig_prcp, ax_prcp = PrecipitationProfilePlot(data_out).make_figure()
# fig_Nm, axNm = StabilityProfilePlot(data_out, meshgrid, axis).make_figure()

if savefigs:
    time = str(data_out.time.dt.strftime("%Y%m%d_%H").values)
    fig_t.savefig(path_figs+time+'_T.png',
                  dpi=200, bbox_inches='tight', format='png')
    fig_wind.savefig(path_figs+time+'_wspd.png',
                     dpi=200, bbox_inches='tight', format='png')
    fig_rh.savefig(path_figs+time+'_RH.png',
                   dpi=200, bbox_inches='tight', format='png')
