"""Plots of vertical transects

Author : Alzbeta Medvedova, Moritz Oberrauch

This script contains functions and classes necessary to create the final
product of this project: the plots of vertical cross-sections.

TODO s:
- make more or less dynamic: colors, contour steps, ... as parameters?!

"""

# external libraries
import numpy as np
import pandas as pd
import xarray as xr

# plotting libraries
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import colors, colorbar, cm
import cmocean.cm as cmo

# local dependencies
import constants


def plot_topography(ds):
    """ TODO: why are these values fixed?!?!?
    I see, for showcase purposes...
    This function must be made more flexible or deleted entirely.

    Parameters
    ----------
    ds

    """
    # initiate figure
    fig, ax = plt.subplots(figsize=[12, 8])
    # change axis projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # plot topography (surface geopotential divided by g): shows dataset extent
    topo = ds.z / constants.G
    topo.plot(ax=ax, cmap='terrain', vmin=0)
    # add borders + coastline for easier orientation
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)

    # add lines of constant latitude
    ax.hlines(y=[46, 47.3, 48.6],
              xmin=ds.longitude.min(),
              xmax=ds.longitude.max(),
              colors='r')
    # add lines of constant longitude (position of IBK)
    ax.vlines(x=[11.4],
              ymin=ds.latitude.min() - 2.5,
              ymax=ds.latitude.max(),
              colors='r')
    # add diagonal lines, calculated to cross through IBK
    ax.plot([5.5, 17.3], [42.6, 52.0], color='r')
    ax.plot([5.5, 17.3], [52.0, 42.6], color='r')


class ProfilePlot:
    """Parent class for all plot of vertical cross sections.

    An ProfilePlot object is instanced with an xarray Dataset, containing the
    data to be plotted. It has methods to plot the following parameters:
    - contour lines of potential temperature
    - contour lines of equivalent potential temperature
    - quiver field of parallel wind (i.e., along the cross section)
    - contour lines of normal wind (i.e., into/out of the cross section)
    - altitude line of 0 degC temperature (as contours, hence can be multiple)
    Additionally, it has a method which takes care of all other plot elements
    (i.e., adding a title, labels, etc.) to finalize the figure.

    Attributes
    ----------
    data : xr.Dataset
        Dateset containing all the variables that can/will be plotted
    grid : numpy.ndarray
        The mesh grid used for plotting the transect, using geopotential height
        as vertical coordinate and longitude/latitude as horizontal
        coordinates. The grid is computed during the slicing, see data_import
        module for details.
    lat : numpy.ndarray
        Latitude(s) spanning the given dataset
    lon : numpy.ndarray
        Longitudes(s) spanning the given dataset
    topo : np.ndarray
        Surface topography in meters, computes from the surface geopotential
    title : string
        Figure title
    fig : matplotlib.pyplot.figure
        The figure instance
    ax : matplotlib.pyplot.axes
        The axes instance

    Methods
    -------
    __init__(data, figsize=(12, 8))
        The constructor copies the given dataset needed coordinates before
        instancing a figure (and axes) with the given figure size.
    finish_figure_settings()
        Finalizes the figure by adding topography, labels, title, and caption.
    plot_theta_contours(color='k')
        Plots isentropes (contour lines of potential temperature)
    plot_theta_contours(color='k')
        Plots contour lines of equivalent potential temperature
    plot_parallel_wind_quivers(nx=30, nz=35)
        Plots quiver field of parallel wind with nx horizontal grid points and
        nz vertical grid points.
    plot_plot_normal_wind_contour(color='k')
        Plots contour lines of normal (in/out of page) wind
    plot_zero_plot_zero_degree_line(color='w')
        Plots contour along the 0 degC altitude line(s)

    See Also
    --------
    data_import: TODO

    """

    def __init__(self, data, figsize=(12, 8)):
        """Each plot must be initialized with a xarray Dataset, containing the
        data to be plotted. The figure size can be supplied/changed as well.

        Parameters
        ----------
        data: xr.Dataset
            Dataset containing the data to be plotted
        figsize: int tuple, optional, default = (12, 8)
            Size of the new figure
        """
        # store data as instance attribute
        self.data = data
        # compute grid
        self.grid = np.tile(data.distance_km, (len(data.level), 1))

        # store coordinates of latitude, longitude and topography as attributes
        self.lat = self.data.latitude.values
        self.lon = self.data.longitude.values
        self.topo = self.data.z.values * 1e-3 / constants.G

        # define empty attributes which will be set later
        self.title = ''

        # initiate figure
        self.fig = plt.figure(figsize=figsize)
        # self.ax = self.fig.add_axes((0.05, 0.07, 0.88, 0.88))
        self.ax = self.fig.add_axes((0.05, 0.07, 0.93, 0.88))

    def finish_figure_settings(self, vertical_limit=12):
        """Finalize the figure after plotting selected variables by:

        - Plotting the topography to the vertical cross-section
        - Adding title, axes labels, grid-ticks and grid-tick labels
        - Setting axes limits
        - Adding additional information as text, like time since model run
        """
        # plot topography
        self.ax.fill_between(self.data.distance_km, self.topo, 0, color='k')

        # axes labels
        self.ax.set_xlabel('Distance [km]', fontsize=14)
        self.ax.set_ylabel('Altitude [km]', fontsize=14)
        # set limit y-axis to 0 and 12 km
        self.ax.set_ylim(0, vertical_limit)

        # add start and end coordinates to the x-tick labels
        self.ax.text(0.01, 0.015,
                     '{:.2f}°N {:.2f}°E'.format(self.lat.min(),
                                                self.lon.min()),
                     transform=self.fig.transFigure, ha='left', fontsize=14)
        self.ax.text(0.99, 0.015,
                     '{:.2f}°N {:.2f}°E'.format(self.lat.max(),
                                                self.lon.max()),
                     transform=self.fig.transFigure, ha='right', fontsize=14)

        # figure title: add new empty line for text with dates + time
        # TODO: set title for each sub class
        self.ax.set_title(self.title, fontsize=16)

        # Compute different timestamps and time differences
        # Initial time of model run
        it = pd.to_datetime(self.data.init_time)
        # Current time of the given dataset (figure time)
        ft = pd.to_datetime(self.data.time.values)
        # Compute time difference between run time and figure time in hours
        dt = int((ft - it).seconds / 3600)

        # Format timestamps for plot
        # Initial time and date of model run + time difference on the left
        # e.g., 00 UTC Run: 2000JAN01 +12
        txt_left = (str(it.hour).zfill(2) + ' UTC Run: ' +
                    str(it.year) + it.month_name()[0:3].upper() + str(it.day) +
                    '  +' + str(dt))
        # Date and time of the given dataset on the right
        # e.g., Sat 2000JAN01 12 UTC
        txt_right = (ft.day_name()[0:3] + ' ' +
                     str(ft.year) + ft.month_name()[0:3].upper() + str(
                    ft.day) +
                     ' ' + str(ft.hour).zfill(2) + ' UTC')

        # add timestamps to the plot
        self.ax.text(0.0, 1.01, txt_left, ha='left',
                     fontsize=14, transform=self.ax.transAxes)
        self.ax.text(1.0, 1.01, txt_right, ha='right',
                     fontsize=14, transform=self.ax.transAxes)

    def plot_contours(self, variable, colors='k', ls='-', lw=1., levels=20,
                      label_format=None, label_fs=12, label_levels=None,
                      **kwargs):
        """Wrapper of matplotlib.pyplot.contour.

        Parameters
        ----------
        variable : str
            Name of the variable to be plotted
        colors : str or array-like, optional, default='k'
            Color(s) of the contour lines, see matplotlib.pyplot.contour
        ls : str or array-like, optional, default='-'
            Line style(s) of the contour lines, see matplotlib.pyplot.contour
        lw : float or array-like, optional, default=1.
            Line width(s) of the contour lines, see matplotlib.pyplot.contour
        levels : int or array-like, optional, default=20
            Determines the number of levels (if int) or specific levels (if
            array-like) of the contour lines, see matplotlib.pyplot.contour
        label_format : matplotlib.tickerFormatter or str or callable or dict,
            optional, default=None
            Formatter for the level labels, see
            matplotlib.contour.ContourLabeler.clabel for details. If none is
            give, no labels are plotted.
        label_fs : float, optional, default=12
            Font size of the level labels
        label_levels : int or array-like, optional, default=20
            Levels that should be labeled: if array-like, must be a subset of
            levels; if int n, every n-th level is labeled.
        kwargs :
            Key word arguments for the matplotlib.pyplot.contour and the
            matplotlib.contour.ContourLabeler.clabel function.


        """
        # plot contours of potential temperature (isentropes)
        isentrope = self.ax.contour(self.grid,
                                    self.data.geopotential_height * 1e-3,
                                    self.data[variable],
                                    levels=levels,
                                    colors=colors,
                                    linewidths=lw,
                                    linestyles=ls, **kwargs)
        # add contour labels with no decimal points
        if label_format:
            if isinstance(label_levels, int):
                label_levels = levels[::label_levels]
            isentrope.clabel(fmt=label_format, fontsize=label_fs,
                             levels=label_levels)

    def plot_theta_contours(self, colors='k', ls='-', lw=0.7, contour_step=4):
        """Plot contour lines of potential temperature in degC.
        TODO: unit?!

        Parameters
        ----------
        colors: string, optional, default='k'
            color for the contour lines, default is black
        """
        # the contour levels are defined as multiples of 4 K (default) and
        # range from -200 degC to +200 degC
        levels = np.arange(-200, 200 + contour_step, contour_step)

        # plot contours of potential temperature (isentropes)
        isentrope = self.ax.contour(self.grid,
                                    self.data.geopotential_height * 1e-3,
                                    self.data.theta - constants.TEMP_0,
                                    levels=levels,
                                    colors=colors,
                                    linewidths=lw,
                                    linestyles=ls)
        # add contour labels with no decimal points
        isentrope.clabel(fmt='%1.0f', fontsize=12)

    def plot_theta_e_contours(self, colors='k', ls='-', lw=0.7, contour_step=4,
                              labels_spacing=2):
        """Plot contour lines of equivalent potential temperature in degC.
        TODO: unit?!

        Parameters
        ----------
        colors: string, optional, default='k'
            color for the contour lines, default is black
        """
        # the contour levels are defined as multiples of 4 K (or degC) and
        # range from -100 degC to +100 degC
        levels = np.arange(-100, 100 + contour_step, contour_step)

        # plot contours of potential temperature (isentropes)
        isentrope = self.ax.contour(self.grid,
                                    self.data.geopotential_height * 1e-3,
                                    self.data.theta_e - constants.TEMP_0,
                                    levels=levels,
                                    colors=colors,
                                    linewidths=lw,
                                    linestyles=ls)
        # add contour labels with no decimal points
        isentrope.clabel(fmt='%1.0f', levels=levels[::labels_spacing],
                         fontsize=12)

    def plot_parallel_wind_quivers(self, nx=30, nz=35):
        """Quiver field of parallel wind

        Plot quivers of normal wind field, it is possible to determine how
        much quivers in grid- and z- direction should be plotted.

        Parameters
        ----------
        nx: int, optional, default=30
            Number of quivers along the grid-direction
        nz: int, optional, default=35
            Number of quivers along the z-direction

        """
        # Plotting quivers at every grid point is too messy. Hence, a quiver is
        # plotted only on every p-th grid point until the given number of
        # quivers in grid and z direction (nx, ny) are full
        nz_all, nx_all = self.grid.shape
        px = int(round(nx_all / nx))
        pz = int(round(nz_all / nz))

        # plot quiver field
        wind_quiver = self.ax.quiver(self.grid[::pz, ::px],
                                     self.data.geopotential_height[::pz,
                                     ::px] * 1e-3,
                                     self.data.parallel_wind[::pz, ::px],
                                     self.data.w_ms[::pz, ::px],
                                     # TODO: check quivers, all horizontal and not angled
                                     color='black',
                                     width=0.002,
                                     headlength=5)
        # add one arrow next to the legend as scale
        self.ax.quiverkey(wind_quiver, 0.965, 0.965, 20,
                          label='20 m/s', labelpos='N',
                          labelsep=0.06,
                          coordinates='figure')

    def plot_normal_wind_contour(self, color='k'):
        """Plot contours of normal wind field in steps of 5 m/s.

        Parameters
        ----------
        color: string, optional, default='k'
            color for the contour lines, default is black
        """
        # the contour levels are defined as multiples of 5 m/s and cover the
        # full range between minimum and maximum wind speed
        contour_step = 5
        max_wind = float(abs(self.data.normal_wind).max().values)
        max_wind = contour_step * round(max_wind / contour_step)
        levels = np.arange(-max_wind, max_wind + contour_step, contour_step)
        # plot contour lines
        wind_contour = self.ax.contour(self.grid,
                                       self.data.geopotential_height * 1e-3,
                                       self.data.normal_wind,
                                       levels=levels,
                                       linewidths=1,
                                       colors=color)
        # change negative values (out of page wind) to dashed lines
        wind_contour.monochrome = True
        # add contour labels with no decimal points
        wind_contour.clabel(fmt='%1.0f', fontsize=12)

    def plot_zero_degree_line(self, color='w'):
        """Plot 0°C altitude line.

        Parameters
        ----------
        color: string, optional, default='w'
            color for the contour lines, default is white
        """
        zero_temp_line = self.ax.contour(self.grid,
                                         self.data.geopotential_height * 1e-3,
                                         self.data.t - constants.TEMP_0,
                                         levels=[0.0],
                                         colors=color,
                                         linewidths=2)


class WindProfilePlot(ProfilePlot):
    """Figure of wind speed and direction in vertical transect

    Instancing a WindProfilePlot with a given dataset instantly creates a
    figure, with total (3D) wind speed as colored contour background, parallel
    and normal (2D) as quiver plot and contour lines, respectively, and
    isentropes. Labels, title, ... are handled by the parent method
    `finish_figure_setting()` before adding a figure caption.

    Attributes
    ----------
    varname : string
        Class attribute containing the name of the variable "total wind speed".
        Used for legends, title and other.
    units : string
        Class attribute containing the corresponding unit of the class variable
        "[m/s]". Used for legends, title and other.


    Methods
    -------
    __init__(data, figsize=(12, 8)
        The constructor instances a ProfilePlot figure, adds the total wind
        speed, parallel and normal wind, and isentropes. Corresponding
        labels, title, legend(s) and caption(s) are added.

    See Also
    --------
    ProfilePlot : parent class instancing figure and axes and hosting the
        methods for adding wind and potential temperature contours and dealing
        with labels and titles.

    """

    # Specific class attributed used for labels and title
    varname = 'total wind speed'
    units = '[m/s]'

    def __init__(self, data, figsize=(12, 8)):
        """Wind plot constructor

        Instancing a WindProfilePlot with a given dataset instantly creates a
        figure, with total (3D) wind speed as colored contour background,
        parallel and normal (2D) as quiver plot and contour lines,
        respectively, and isentropes. Labels, title, etc are handled by the
        parent method `finish_figure_setting()` before adding a figure caption.

        Parameters
        ----------
        data: xr.Dataset
            Dataset containing the data to be plotted
        figsize: int tuple, optional, default = (12, 8)
            Size of the new figure

        """
        # initialize ProfilePlot parent class
        super(WindProfilePlot, self).__init__(data, figsize)

        # define levels of wind speed contour plot
        # TODO: the following values could be given as parameters
        cmap = cmo.haline_r
        wind_min = 0
        wind_max = 60
        wind_step = 2
        levels = np.arange(wind_min, wind_max + wind_step, wind_step)
        # Plot total wind speed as background
        bcg = self.ax.contourf(self.grid,
                               self.data.geopotential_height * 1e-3,
                               self.data.wspd,
                               levels=levels,
                               cmap=cmap,
                               extend='max',
                               alpha=0.9,
                               antialiased=True)
        # add colorbar
        cax, _ = colorbar.make_axes(self.ax, location='right', fraction=0.035,
                                    shrink=1.0, aspect=30, pad=0.015)
        self.fig.colorbar(bcg, cax=cax)

        # plot contour lines of potential temperature
        self.plot_theta_contours(colors='w')
        # plot quiver field of parallel wind
        self.plot_parallel_wind_quivers()
        # plot contour lines of normal wind
        self.plot_normal_wind_contour()
        # finish figure layout settings and labels
        self.finish_figure_settings()

        # add figure caption below
        # self.fig.tight_layout()
        # figtext = 'ECMWF forecast: wind speed [m/s]: vectors (transect ' \
        #           'plane) and black contours (full lines out of page, \n' \
        #           'dashed into the page); shading (scalar wind speed)], ' \
        #           'potential temperature [C, white contours]'
        # self.fig.text(0.5, -0.1, figtext, transform=self.ax.transAxes,
        #               ha='center', va='top', fontsize=12, wrap=True)


class TemperatureProfilePlot(ProfilePlot):
    """Figure of air temperature in vertical transect

    Instancing a TemperatureProfilePlot with a given dataset instantly creates
    a figure, with air temperature as colored contour background, parallel and
    normal (2D) as quiver plot and contour lines, respectively,
    isentropes and the zero degree altitude line. Labels, title, etc. are
    handled by the parent method `finish_figure_setting()` before adding a
    figure caption.

    Attributes
    ----------
    varname : string
        Class attribute containing the name of the variable "temperature". Used
        for legends, title and other.
    units : string
        Class attribute containing the corresponding unit of the class variable
        "[°C]". Used for legends, title and other.


    Methods
    -------
    __init__(data, figsize=(12, 8)
        The constructor instances a ProfilePlot figure, adds the total wind
        speed, parallel and normal wind, and isentropes. Corresponding
        labels, title, legend(s) and caption(s) are added.

    See Also
    --------
    ProfilePlot : parent class instancing figure and axes and hosting the
        methods for adding wind and potential temperature contours and dealing
        with labels and titles.

    """

    # Class attributes: specific for wind plot
    varname = 'temperature'
    units = '[°C]'

    def __init__(self, data):
        """Temperature plot constructor

        Instancing a TemperatureProfilePlot with a given dataset instantly
        creates a figure, with air temperature as colored contour background,
        parallel and normal (2D) as quiver plot and contour lines,
        respectively, isentropes and the zero degree altitude line. Labels,
        title, etc. are handled by the parent method `finish_figure_setting()`
        before adding a figure caption.

        Parameters
        ----------
        data: xr.Dataset
            Dataset containing the data to be plotted
        figsize: int tuple, optional, default = (12, 8)
            Size of the new figure

        """
        # initialize ProfilePlot parent class
        super(TemperatureProfilePlot, self).__init__(data)

        # define levels of temperature contour plot
        cmap = cmo.thermal
        temp_min = -50
        temp_max = 40
        temp_step = 2
        levels = np.arange(temp_min, temp_max + temp_step, temp_step)

        # Plot potential temperature in degC as background
        bcg = self.ax.contourf(self.grid,
                               self.data.geopotential_height * 1e-3,
                               self.data.t - constants.TEMP_0,
                               levels=levels,
                               cmap=cmap,
                               extend='both',
                               alpha=0.9,
                               antialiased=True)
        # add colorbar
        cax, _ = colorbar.make_axes(self.ax, location='right', fraction=0.035,
                                    shrink=1.0, aspect=30, pad=0.015)
        # c_norm = colors.BoundaryNorm(levels, cmap.N, extend='both')
        self.fig.colorbar(bcg, cax=cax)

        # plot contour lines of potential temperature
        self.plot_theta_contours(colors='w')
        # plot quiver field of parallel wind
        self.plot_parallel_wind_quivers()
        # plot contour lines of normal wind
        self.plot_normal_wind_contour()
        # add zero degree temperature line
        self.plot_zero_degree_line(color='blue')

        # finish figure layout settings and labels
        self.finish_figure_settings()
        # add figure caption below
        # self.fig.tight_layout()
        # ax_loc = self.fig.axes[0].get_position()
        # figtext = 'ECMWF forecast: temperature [°C, shading], 0°C line ' \
        #           '(blue), wind [m/s]: vectors (transect plane), black \n' \
        #           'contours (full lines out of page, dashed into the ' \
        #           'page)], potential temperature [C, white contours]'
        # self.fig.text(ax_loc.xmin, 0.00, figtext,
        #               ha='left', va='top', fontsize=12, wrap=True)


class EquivalentPotentialTemperatureProfilePlot(ProfilePlot):
    """Figure of air temperature in vertical transect

    Instancing a TemperatureProfilePlot with a given dataset instantly creates
    a figure, with air temperature as colored contour background, parallel and
    normal (2D) as quiver plot and contour lines, respectively,
    isentropes and the zero degree altitude line. Labels, title, etc. are
    handled by the parent method `finish_figure_setting()` before adding a
    figure caption.

    Attributes
    ----------
    varname : string
        Class attribute containing the name of the variable "temperature". Used
        for legends, title and other.
    units : string
        Class attribute containing the corresponding unit of the class variable
        "[°C]". Used for legends, title and other.


    Methods
    -------
    __init__(data, figsize=(12, 8)
        The constructor instances a ProfilePlot figure, adds the total wind
        speed, parallel and normal wind, and isentropes. Corresponding
        labels, title, legend(s) and caption(s) are added.

    See Also
    --------
    ProfilePlot : parent class instancing figure and axes and hosting the
        methods for adding wind and potential temperature contours and dealing
        with labels and titles.

    """

    # Class attributes: specific for wind plot
    varname = 'temperature'
    units = '[°C]'

    def __init__(self, data):
        """Temperature plot constructor

        Instancing a TemperatureProfilePlot with a given dataset instantly
        creates a figure, with air temperature as colored contour background,
        parallel and normal (2D) as quiver plot and contour lines,
        respectively, isentropes and the zero degree altitude line. Labels,
        title, etc. are handled by the parent method `finish_figure_setting()`
        before adding a figure caption.

        Parameters
        ----------
        data: xr.Dataset
            Dataset containing the data to be plotted
        figsize: int tuple, optional, default = (12, 8)
            Size of the new figure

        """
        # initialize ProfilePlot parent class
        super(EquivalentPotentialTemperatureProfilePlot, self).__init__(data)

        # define levels of temperature contour plot
        cmap = cmo.balance
        temp_min = -28
        temp_max = 92
        temp_step = 2
        levels = np.arange(temp_min, temp_max + temp_step, temp_step)

        # Plot potential temperature in degC as background
        bcg = self.ax.contourf(self.grid,
                               self.data.geopotential_height * 1e-3,
                               self.data.theta_e - constants.TEMP_0,
                               levels=levels,
                               cmap=cmap,
                               extend='both',
                               alpha=0.9,
                               antialiased=True)
        # add colorbar
        cax, _ = colorbar.make_axes(self.ax, location='right', fraction=0.035,
                                    shrink=1.0, aspect=30, pad=0.015)
        # c_norm = colors.BoundaryNorm(levels, cmap.N, extend='both')
        self.fig.colorbar(bcg, cax=cax)
        #
        # rrr = self.ax.contourf(self.grid,
        #                        self.data.geopotential_height * 1e-3,
        #                        self.data.rh,
        #                        cmap=cmo.gray_r,
        #                        levels=np.arange(0,100,10),
        #                        extend='max',
        #                        alpha=0.5)
        # self.fig.colorbar(rrr, cax=cax)

        # plot contour lines of potential temperature
        self.plot_theta_e_contours(colors='k', ls='--', contour_step=2)
        # # plot quiver field of parallel wind
        # self.plot_parallel_wind_quivers()
        # # plot contour lines of normal wind
        # self.plot_normal_wind_contour()
        # # add zero degree temperature line
        # self.plot_zero_degree_line(color='blue')

        # finish figure layout settings and labels
        self.finish_figure_settings()
        # add figure caption below
        # self.fig.tight_layout()
        # ax_loc = self.fig.axes[0].get_position()
        # figtext = 'ECMWF forecast: temperature [°C, shading], 0°C line ' \
        #           '(blue), wind [m/s]: vectors (transect plane), black \n' \
        #           'contours (full lines out of page, dashed into the ' \
        #           'page)], potential temperature [C, white contours]'
        # self.fig.text(ax_loc.xmin, 0.00, figtext,
        #               ha='left', va='top', fontsize=12, wrap=True)


class RhProfilePlot(ProfilePlot):
    """Figure of (relative) humidity in vertical transect

    Instancing a RhProfilePlot with a given dataset instantly creates a figure,
    with relative humidity as colored contour background, parallel and
    normal (2D) as quiver plot and contour lines, respectively,
    and isentropes (equivalent potential temperature). Labels, title, etc. are
    handled by the parent method `finish_figure_setting()` before adding a
    figure caption.

    Attributes
    ----------
    varname : string
        Class attribute containing the name of the variable
        "relative humidity". Used for legends, title and other.
    units : string
        Class attribute containing the corresponding unit of the class variable
        "[%]". Used for legends, title and other.


    Methods
    -------
    __init__(data, figsize=(12, 8)
        The constructor instances a ProfilePlot figure, adds the total wind
        speed, parallel and normal wind, and isentropes. Corresponding
        labels, title, legend(s) and caption(s) are added.

    See Also
    --------
    ProfilePlot : parent class instancing figure and axes and hosting the
        methods for adding wind and potential temperature contours and dealing
        with labels and titles.

    """

    # Class attributes: specific for wind plot
    varname = 'relative humidity'
    units = '[%]'

    def __init__(self, data, figsize=(12, 8)):
        """Humidity plot constructor

        Instancing a RhProfilePlot with a given dataset instantly creates a
        figure, with relative humidity as colored contour background, parallel
        and normal (2D) as quiver plot and contour lines, respectively,
        and isentropes (equivalent potential temperature). Labels, title, etc.
        are handled by the parent method `finish_figure_setting()` before
        adding a figure caption.

        Parameters
        ----------
        data: xr.Dataset
            Dataset containing the data to be plotted
        figsize: int tuple, optional, default = (12, 8)
            Size of the new figure
        """
        # initialize ProfilePlot parent class
        super(RhProfilePlot, self).__init__(data, figsize)

        # the contour levels are defined as in 10% steps from 0%grid to 100%
        contour_step = 10
        levels = np.arange(0, 100, step=contour_step)

        # Background specific for the relative humidity figure
        bcg = self.ax.contourf(self.grid,
                               self.data.geopotential_height * 1e-3,
                               self.data.rh,
                               levels=levels,
                               cmap='Greens',
                               extend='max',
                               alpha=0.8,
                               antialiased=True)
        # add colorbar
        cax, _ = colorbar.make_axes(self.ax, location='right', fraction=0.035,
                                    shrink=1.0, aspect=30, pad=0.015)
        self.fig.colorbar(bcg, cax=cax)

        # plot contour lines of equivalent potential temperature
        self.plot_theta_e_contours()
        # # plot quiver field of parallel wind
        # self.plot_parallel_wind_quivers()
        # # plot contour lines of normal wind
        # self.plot_normal_wind_contour()

        # finish figure layout settings and labels
        self.finish_figure_settings()
        # add figure caption below
        # self.fig.tight_layout()
        # ax_loc = self.fig.axes[0].get_position()
        # figtext = 'ECMWF forecast: relative humidity (shading), wind speed ' \
        #           '[m/s]: vectors (transect plane), black contours \n(full ' \
        #           'lines out of page, dashed into the page), equivalent ' \
        #           'potential temperature [C, black contours]'
        # self.fig.text(ax_loc.xmin, 0.00, figtext,
        #               ha='left', va='top', fontsize=12, wrap=True)


class StabilityProfilePlot(ProfilePlot):
    """ Inherits methods and attributes from the 'ProfilePlot' class """
    # TODO: This still doesn't work I'm pretty sure!
    # Some mistake in the calculation probably? CHECK!

    # Class attributes: specific for stability plot
    varname = 'moist Brunt-Väisälä frequency'
    units = '[$s^{-2}$]'

    def __init__(self, data):
        """TODO: finish docstring"""
        # initialize ProfilePlot parent class
        super(StabilityProfilePlot, self).__init__(data)

        # Background specific for the relative humidity figure
        bcg = self.ax.contourf(self.grid,
                               self.data.geopotential_height * 1e-3,
                               self.data.N_m,
                               levels=np.arange(-4e-4, 4e-4, 4e-5),
                               cmap='RdGy',
                               extend='both',
                               alpha=0.8,
                               antialiased=True)

        cbar = self.fig.colorbar(bcg)
        cbar.ax.set_ylabel('$N_m^2$' + ' ' + self.units, fontsize=14)

        # plot contour lines of equivalent potential temperature
        self.plot_theta_e_contours('black')
        # plot quiver field of parallel wind
        self.plot_parallel_wind_quivers()
        # plot contour lines of normal wind
        self.plot_normal_wind_contour()

        # finish figure layout settings and labels
        self.finish_figure_settings()
        # TODO: add figure explanation/caption


class VorticityProfilePlot(ProfilePlot):
    """ Dataset has relative vorticity [s^-1]"""
    # TODO: not implemented
    pass


class PrecipitationProfilePlot(ProfilePlot):
    """ Dataset has suspended/precipitating water/ice"""
    # TODO: not implemented... or at least not finalized as I'd like it to be.
    # Add figtext.
    # Move cbars closer to each other to make more space for the plot itself

    # Class attributes: specific for stability plot
    varname = 'suspended and precipitating water'
    units = '[$kg~kg^{-1}$]'

    def plot_suspended_water_ice(self):
        '''
        Plots specific cloud liquid / ice water content as filled contours
        '''
        water = self.ax.contourf(self.grid,
                                 self.data.geopotential_height * 1e-3,
                                 self.data.clwc,
                                 levels=np.array([0.05, 0.1, 0.2, 0.5]) * 1e-3,
                                 cmap='Greys',
                                 extend='max',
                                 alpha=0.8,
                                 antialiased=True)

        ice = self.ax.contourf(self.grid,
                               self.data.geopotential_height * 1e-3,
                               self.data.ciwc,
                               levels=np.array([0.05, 0.1, 0.2, 0.5]) * 1e-3,
                               cmap='Blues',
                               extend='max',
                               alpha=0.8,
                               antialiased=True)

        # TODO 
        # to make more space for the figure itself
        cbar_w = self.fig.colorbar(water)
        cbar_i = self.fig.colorbar(ice).set_ticks([])
        cbar_w.ax.set_ylabel(self.varname.capitalize() + ' ' + self.units,
                             fontsize=14)
        return

    # Specify which other variables should be overlaid on the plot
    def make_figure(self):
        # plot background and its colorbar
        self.plot_suspended_water_ice()
        # add theta contours
        self.plot_theta_e_contours('black')
        # add zero temperature line
        self.plot_zero_degree_line(color='blue')
        # finish figure layout settings and labels
        self.finish_figure_settings()
        return self.fig, self.ax
