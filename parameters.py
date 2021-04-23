# %%
#!/usr/bin/env python
#
# Copyright (c) 2021, Kaustubh Hakim
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
#
# CHILI 1.0 
#
# parameters.py (module to store global parameters and constants)
#  
# If you are using this code, please cite the following publication
# Hakim et al. (2021) Lithologic Controls on Silicate Weathering Regimes of
# Temperate Planets. The Planetary Science Journal 2. doi:10.3847/PSJ/abe1b8

# %%
# Import packages
import numpy as np
from matplotlib.cm import get_cmap

# %%
# Global parameters
Teq0    = 254                           # K         Earth equilibrium blackbody temperature
T0      = 298.15                        # K         Ambient temperature
T278    = 278                           # K         Present day global mean surface temperature
T288    = 288                           # K         Present day global mean surface temperature
T348    = 348                           # K         50 C
xCO2g0  = 280e-6                        #           Pre-industrial CO2 volume mixing ratio
xCO2g1  = 1                             #           Maximum CO2 volume mixing ratio
P0      = 1                             # bar       Total pressure, 1 bar
P1      = 0.01                          # bar       Total pressure, minimum
P2      = 1000                          # bar       Total pressure, maximum
Pd0     = 200                           # bar       Assumed deep-sea pressure
S0      = 1360                          # W/m2      Present day solar flux
qc      = 0.3                           # m/a       Global average continental runoff
qs      = 0.05                          # m/a       Global average hydrothermal fluid flow rate
alpha0  = 0.3                           #           Planetary albedo
Lc      = 1                             # m         Flow path length (continents)
Ls      = 100                           # m         Flow path length (seafloor)
t_soilc = 1e5                           # yr        Age of soils (continents)
t_soils = 5e7                           # yr        Age of soils (seafloor)
Fcstar  = 7.5                           # Tmol/yr   Foley (2015)
Fcmini  = 4                             # Tmol/yr   Kris-Tott & Catling (2017)
Fcmaxi  = 23                            # Tmol/yr   Holland (1978)
Fclowl  = Fcstar - Fcmini               #           lower error on Fcstar
Fcuppl  = Fcmaxi - Fcstar               #           upper error on Fcstar
Fsstar  = 0.45                          # Tmol/yr   Foley (2015)
Fsmini  = 0.225                         # Tmol/yr   Kris-Tott & Catling (2017)
Fsmaxi  = 3.3                           # Tmol/yr   Sleep & Zahnle (2001)
Fslowl  = Fsstar - Fsmini               #           lower error on Fsstar
Fsuppl  = Fsmaxi - Fsstar               #           upper error on Fsstar

# %%
# Define a range for control parameters
T       = np.linspace(273.15,372.15)    # K         Temp range for generic T-based calculations
xCO2    = np.logspace(-8,0)             #           Volume mixing ratio or activity of CO2(g) 
ppm     = xCO2 * 10**6                  # ppm       PCO2 with ppm units
P       = np.logspace(-2,3)             # bar       Ptot (0.01-1000 bar)
Pfull   = np.logspace(-2,3.05)          # bar       Ptot (0.01-1100 bar)
P1bar   = np.logspace(0,3)              # bar       Ptot (1-1000 bar)
pHfull  = np.linspace(0,14)             #           Full pH range
tclim   = np.linspace(0.02,3.98)        # Ga        time Gyears ago, time(present day) = 0 Ga

C_eq    = np.logspace(-6,1)             # mol/dm3   Solute concentration
keff    = np.logspace(-8,2)             # mol/m2/yr Effective kinetic rate coefficient
t_soil  = np.logspace(0,8)              # yr        Age of soils
t_soill = np.linspace(0,1,num=2)        # yr        Age of soils
t_soil_ = np.array([0,1e5,1e6,5e7,1e8]) # yr        Age of soils
sp_area = np.logspace(1,3)              # m2/kg     SPecific reactive surface area of rock
rho     = np.linspace(2000,3000)        # kg/m3     Density of rock
L       = np.logspace(-1,3)             # m         Flowpath length
L_      = np.array([1,15,100])          # m         Flowpath length
phi     = np.linspace(0.05,0.5)         #           Porosity
mol_mass= np.logspace(-1,0)             # kg/mol    Mean molar mass of rock
X_r     = np.linspace(0.1,1)            # X_r       Fraction of reactive minerals in a rock

q       = np.logspace(-5,3,num=200)     # m/yr      Runoff or Fluid flow rate 
HCO3full= np.logspace(-12,6,num=200)    # mol/dm3   Bicarbonate ion concentration 
Dwfull  = np.logspace(-12,6,num=200)    # m/yr      Damkoehler coefficient

# %%
col1 = get_cmap('Dark2').colors  # type: matplotlib.colors.ListedColormap
col2 = get_cmap('Set1').colors
col3 = get_cmap('Set3').colors
colors = col1 + col2 + col3

col_basa = colors[0]
col_gabb = colors[1]
col_peri = colors[2]
col_gran = colors[3]
col_anor = colors[4]
col_fors = colors[5]
col_woll = 'k'
col_enst = colors[6]
col_phlo = colors[7]
col_faya = colors[8]
col_ferr = colors[9]
col_musc = colors[10]
col_anni = colors[11]
col_albi = colors[12]
col_grun = colors[17]
col_kfel = colors[14]
col_anth = colors[15]
col_quar = colors[16]
col_bica = colors[22]
col_carb = colors[26]
col_wate = colors[19]
col_co2a = colors[20]
col_kaol = colors[21]
col_cont = colors[24]
col_walk = colors[14]
col_folc = colors[12]
col_kric = colors[15]
col_seaf = colors[27]
col_slee = colors[21]
col_brad = colors[21]
col_fols = colors[11]
col_kris = colors[10]
lwi_mine = 'solid'
lwo_mine = 'dotted'
lwi_rock = 'dashed'
lwo_rock = 'dashdot'
wid_mine = 3
wid_rock = 4
wid_walk = 2
wid_pres = 4
