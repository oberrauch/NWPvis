""" Physical constants

This module contains physical constant necessary for calculations of quantities
not contained in the original dataset.

Author(s): Alzbeta Medvedova, Moritz Oberrauch

References:

    American Meteorological Society (AMS), 2018: Glossary of Meteorology:
    geopotential height, https://glossary.ametsoc.org/wiki/Geopotential_height

    Markowski P. and Richardson Y., 2016: Mesoscale Meteorology in Midlatitudes
    Wiley-Blackwell, Royal Meteorological Society, ISBN: 978-0-470-74213-6

    Stull, R., 2011: Meteorology for Scientists & Engineers, 3rd Edition
    Univ. of British Columbia.  938 pages.  ISBN 978-0-88865-178-5

"""

# Gravitational acceleration, global average at sea level (AMS 2018)
g = 9.80665  # [m/s^2]

# Gas constants for the equation of state (Markowski and Richardson, 2016)
Rd = 287.04  # [J/kg/K] gas constant, dry air
Rvap = 461.51  # [J/kg/K] gas constant, water vapor

# Specific heat capacities
# TODO: REF and exact value for specific heat of liquid water
c_p = 1005.7  # [J/kg/K] specific heat of dry air at constant pressure
c_l = 4190  # [J/kg/K] specific heat of liquid water at constant pressure
rcp = Rd / c_p  # [-] Rd/c_p

# reference values
t_0 = 273.15  # [K] reference temperature (0 degC)
p0 = 1e5  # [Pa] reference pressure (100 hPa)

# Compute dry adiabatic lapse rate following Stull (2011)
gamma_d = g / c_p  # [K/m] dry adiabatic lapse rate
