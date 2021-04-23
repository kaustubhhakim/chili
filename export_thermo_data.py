#!/usr/bin/env python
# coding: utf-8
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
# export_thermo_data.py (import CHNOSZ thermo data and export in csv format)
#  
# If you are using this code, please cite the following publication
# Hakim et al. (2021) Lithologic Controls on Silicate Weathering Regimes of
# Temperate Planets. The Planetary Science Journal 2. doi:10.3847/PSJ/abe1b8

# %%
import pandas as pd
pd.options.mode.chained_assignment = None
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
utils = importr('utils')
utils.install_packages('CHNOSZ', repos = 'http://R-Forge.R-project.org')
CHNOSZ = importr('CHNOSZ')
CHNOSZ.add_OBIGT('SUPCRT92', 'halloysite')


# %%
speciesNames = pd.read_csv('./database/species.csv')
CHNOSZ.T_units('K')
CHNOSZ.E_units('J')
for speciesName in speciesNames['species name']:
    print('Exported thermodynamic data of', speciesName)
    speciesData = CHNOSZ.subcrt(speciesName, P = r.seq(0.01,1201.01,1), T = r.seq(273.15, 373.15, 1), grid = 'P')
    utils.write_table(speciesData,'./database/'+speciesName+'_th',sep=',',row_names=False)

