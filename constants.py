""" Physical constants

This module contains physical constant necessary for calculations of quantities
not contained in the original dataset.

Author(s): Alzbeta Medvedova, Moritz Oberrauch

References:
 .. [AMS 2018] American Meteorological Society AMS, 2018: Glossary of
    Meteorology: geopotential height,
    https://glossary.ametsoc.org/wiki/Geopotential_height

 .. [Markowski 2016] Markowski P. and Richardson Y., 2016: Mesoscale
    Meteorology in Midlatitudes Wiley-Blackwell, Royal Meteorological Society,
    ISBN 978-0-470-74213-6

 .. [Stull 2011] Stull, R., 2011: Meteorology for Scientists & Engineers, 3rd
    Edition, Univ. of British Columbia, 938 pages, ISBN 978-0-88865-178-5

"""

# Gravitational acceleration, global average at sea level [AMS 2018]_
G = 9.80665  # [m/s^2]

# Gas constants for the equation of state [Markowski 2016]_
R_DRY = 287.04  # [J/kg/K] gas constant, dry air
R_WATER = 461.51  # [J/kg/K] gas constant, water vapor

# Specific isobaric heat capacities
# TODO: REF and exact values (currently from https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities)
SPEC_CP_DRY = 1003.5  # [J/kg/K] dry air, at 0 degC and sea level
SPEC_CP_WATER = 4181.3  # [J/kg/K] liquid water at 25 degC

# reference values
TEMP_0 = 273.15  # [K] reference temperature (0 degC)
PRESSURE_0 = 1e5  # [Pa] reference pressure (100 hPa)

# Compute dry adiabatic lapse rate following [Stull 2011]_
GAMMA_DRY = G / SPEC_CP_DRY  # [K/m] dry adiabatic lapse rate

