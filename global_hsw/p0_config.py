"""Shared sample definition for Paper 0.

The main VIIRS panel starts in 2014. EOG built its 2012 and 2013 annual VNL composites from monthly
data without the stray-light correction ("vcmcfg"); from 2014 the composites use stray-light-corrected
monthly data ("vcmslcfg") (Elvidge et al. 2021, p. 2). Starting in 2014 keeps every panel year on one
processing chain. 2012 and 2013 remain in the analysis as the two DMSP-VIIRS overlap years.
"""
VIIRS_START = 2014
VIIRS_END = 2025
VIIRS_YEARS = VIIRS_END - VIIRS_START + 1  # 12
