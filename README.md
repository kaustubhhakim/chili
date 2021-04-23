# CHILI #
#### Author: Kaustubh Hakim ####

# Overview #

CHILI (CHemical weatherIng model based on LIthology) is lithology (rock type) based model to calculate silicate weathering rates on temperate planets using basic principles of chemistry and physics. 

The code is written in Python. Before you can run CHILI on your computer, Python 3 needs to be installed. 

# Running CHILI #

## Example ##

After cloning CHILI on your machine, you can test if CHILI is working by running the following command. This command generates and saves a figure in PDF format.

> python plot_example.py

## All figures ##

The following command generates and saves figures in PDF format. These figures are included in Hakim et al. (2021).

> python plot_all.py

## Updating thermodynamic data ##

The thermodynamic data to compute equilibrium constants is already provided with CHILI. However, if there is a need to update the thermodynamic data, the following command needs to be run. R needs to be installed before running this command. 

> python export_thermo_data.py


# References #

Hakim et al. (2021), Lithologic Controls on Silicate Weathering Regimes of Temperate Planets. The Planetary Science Journal 2. https://doi.org/10.3847/PSJ/abe1b8
