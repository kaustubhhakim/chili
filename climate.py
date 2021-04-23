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
# climate.py (module to calculate T as a function of PCO2)
#  
# If you are using this code, please cite the following publication
# Hakim et al. (2021) Lithologic Controls on Silicate Weathering Regimes of
# Temperate Planets. The Planetary Science Journal 2. doi:10.3847/PSJ/abe1b8

# %%
# Import packages
import numpy as np
from scipy.optimize import newton
from scipy.interpolate import interp1d
from astropy.constants import sigma_sb
from astropy.constants import R

# %%
def PH2O(Temp):
    '''Calculate saturation vapor pressure of water at a given temperature.
    
    Parameters
    ----------
    Temp : float
       Temperature (K)
    
    Returns
    -------
    PH2O : float
        Saturation vapor pressure of water (bar)
        
    '''
    
    m_w    = 18e-3 # kg/m3
    L_w    = 2469e3 # J / kg
    T_sat0 = 273 # K
    P_sat0 = 610e-5 # bar
    PH2O   = P_sat0 * np.exp( - m_w * L_w / R.value * (1/Temp - 1/T_sat0) )
    return PH2O

# %%
# Climate model from Walker et al. (1981)
def T_WHAK(P_CO2, S=1368, albedo=0.3):
    '''Calculate T as a function of P_CO2, S, albedo
    
    Parameters
    ----------
    P_CO2 : float
       Partial pressure of CO2(g) (bar)
    S : float
        Stellar flux (W/m2)    
    albedo : float
        Planetary albedo
    
    Returns
    -------
    Temp : float
        Surface temperature (K)
        
    '''
    
    P_CO20 = 280e-6 # bar
    Teq0   = 253 # K
    Teq    = (S * (1 - albedo) / (4 * sigma_sb.value) )**(1/4)
    Temp   = 285 + 2 * (Teq - Teq0)  + 4.6 * (P_CO2/P_CO20)**0.346 - 4.6
    return Temp

# %%
# Climate model from Kadoya & Tajika (2019) with planetary albedo as free param
def T_KATA(P_CO2, S=1368, albedo=0.3):
    '''Calculate T as a function of P_CO2, S, albedo
    
    NOTE: albedo (planetary albedo) is a free parameter
    OLR = S_avg gives a polynomial equation in T and P_CO2
    OLR formulation from Eqs. (1-7) of Kadoya & Tajika (2019)
    
    Parameters
    ----------
    P_CO2 : float
       Partial pressure of CO2(g) (bar)
    S : float
        Stellar flux (W/m2)
    albedo : float
        Planetary albedo
    
    Returns
    -------
    Temp : float
        Surface temperature (K)
        
    '''
    
    S_avg   = S / 4 * (1 - albedo)
    
    I0      = - 3.1 # W/m2, Kadoya & Tajika (2019)
    if P_CO2 < 1:
        p = 0.2 * np.log10(P_CO2) # P_CO2 < 1 bar, Kadoya & Tajika (2019)
        B = np.array([[ 87.8373, -311.289, -504.408, -422.929, -134.611],
                      [ 54.9102, -677.741, -1440.63, -1467.04, -543.371],
                      [ 24.7875,  31.3614, -364.617, -747.352, -395.401],
                      [ 75.8917,  816.426,  1565.03,  1453.73,  476.475],
                      [ 43.0076,  339.957,  996.723,  1361.41,  612.967],
                      [-31.4994, -261.362, -395.106, -261.600, -36.6589],
                      [-28.8846, -174.942, -378.436, -445.878, -178.948]])
    else:
        p = np.log10(P_CO2) # P_CO2 > 1 bar, Kadoya & Tajika (2019)
        B = np.array([[ 87.8373, -52.1056,  35.2800, -1.64935, -3.42858],
                      [ 54.9102, -49.6404, -93.8576,  130.671, -41.1725],
                      [ 24.7875,  94.7348, -252.996,  171.685, -34.7665],
                      [ 75.8917, -180.679,  385.989, -344.020,  101.455],
                      [ 43.0076, -327.589,  523.212, -351.086,  81.0478],
                      [-31.4994,  235.321, -462.453,  346.483, -90.0657],
                      [-28.8846,  284.233, -469.600,  311.854, -72.4874]])

    PP = np.array([p**0, p, p**2, p**3, p**4])
    BPt = B @ PP.T
    a = BPt[6]
    b = BPt[5]
    c = BPt[4]
    d = BPt[3]
    e = BPt[2]
    f = BPt[1]
    g = BPt[0] + I0 - S_avg # Kadoya & Tajika (2019)

    def func(x):
        return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g
    
    x = 0.1 # 0.05*(4.7 + 0.4*np.log10(P_CO2))**2
    x0 = newton(func, x, args=(), tol=1e-10, maxiter=1000)
    Temp = x0 * 100 + 250 # Kadoya & Tajika (2019)
    return Temp


# %%
# PCO2 as a function of time from Krissansen-Totton et al. (2018)
def import_PCO2Func(csvData): 
    '''Return interpolated PCO2(time) function
    
    Parameters
    ----------
    csvData : pandas object
       Imported data file
    
    Returns
    -------
    PCO2Func : float
        Interpolated PCO2(time) function
        
    '''
    
    return interp1d(csvData['Time (Ga)'],csvData['PCO2 (bar)'],kind='cubic')


# %%
# Temp as a function of time from Krissansen-Totton et al. (2018)
def import_TempFunc(csvData): 
    '''Return interpolated Temp(time) function
    
    Parameters
    ----------
    csvData : pandas object
       Imported data file
    
    Returns
    -------
    TempFunc : float
        Interpolated Temp(time) function
        
    '''
    
    return interp1d(csvData['Time (Ga)'],csvData['T (K)'],kind='cubic')


# %%
# Cont. weath. rate as a function of time from Krissansen-Totton et al. (2018)
def import_contFunc(csvData):
    '''Return interpolated continental weathering function
    
    Parameters
    ----------
    csvData : pandas object
       Imported data file
    
    Returns
    -------
    contFunc: float
        Interpolated contFunc(time) function
        
    '''
    
    return interp1d(csvData['Time (Ga)'],csvData['Cont (Tmol/yr)'],kind='cubic',fill_value='extrapolate')


# %%
# Seaf. weath. rate as a function of time from Krissansen-Totton et al. (2018)
def import_seafFunc(csvData):
    '''Return interpolated seafloor weathering function
    
    Parameters
    ----------
    csvData : pandas object
       Imported data file
    
    Returns
    -------
    seafFunc: float
        Interpolated seafFunc(time) function
        
    '''
    
    return interp1d(csvData['Time (Ga)'],csvData['Seaf (Tmol/yr)'],kind='cubic',fill_value='extrapolate')


# %%
# Return continental weathering rate from Walker et al. (1981)
def cont_walk1981(Temp,P_CO2):
    '''Calculate continental weathering rate from Walker et al. (1981) with 
    modern rate normalized to normalized to Krissansen-Totton et al. (2018)
    
    Parameters
    ----------
    Temp : float
       Temperature (K)
    P_CO2 : float
       Partial pressure of CO2(g) (bar)
    
    Returns
    -------
    W: float
        Continental Weathering rate (Tmol/yr)
        
    '''
    
    delT    =   13.7 # K
    beta    =   0.3
    P_CO20  =   280e-6 # bar
    Tref    =   285 # K
    Wcont0  =   7.6 # Tmol/yr, Krissansen-Totton et al. (2018)
    return  Wcont0 * (P_CO2/P_CO20)**(beta) * np.exp((Temp - Tref)/delT)


# %%
# Return seafloor weathering rate from Brady & Gislason (1997)
def seaf_brad1997(Temp,P_CO2):
    '''Calculate seafloor weathering rate from Brady & Gislason (1997) with 
    modern rate normalized to normalized to Krissansen-Totton et al. (2018)
    
    Parameters
    ----------
    Temp : float
       Temperature (K)
    P_CO2 : float
       Partial pressure of CO2(g) (bar)
    
    Returns
    -------
    W: float
        Seafloor Weathering rate (Tmol/yr)
        
    '''
    
    delT    =   15.2 # K
    beta    =   0.23
    P_CO20  =   280e-6 # bar
    Tref    =   285 # K
    Wcont0  =   0.414 # Tmol/yr, Krissansen-Totton et al. (2018)
    return  Wcont0 * (P_CO2/P_CO20)**(beta) * np.exp((Temp - Tref)/delT) 
