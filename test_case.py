import glob
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from functions import *

path_to_files = 'enter_path_here'
fnames = sorted(glob.glob(path_to_files))

# change for the case you are using
yyyy     = 2022
mm       = 1
dd       = 1
timezone = 'E'

# get day of year
doy  = pd.to_datetime(datetime(year=yyyy,month=mm,day=dd)).day_of_year

# create QVPs
dvar, zdr, zdrvar, time_array, height_array, lat = create_qvp(fnames)

# calculate max growth rate for algorithm
max_growth = find_growth_rate(lat, doy)

# smooth QVP 3x before applying algorithm
n=0
while n<2:
  dvar = smooth_QVP(dvar)
  n+=1

# apply algorithm
dvar_min_hgt = find_dvar_min_v2(mm, timezone, time_array, height_array, dvar,
                  max_growth, min_growth = -300, hstep = 20, smooth=False)
            
# plot QVP with estimated PBLH      
plot_QVP(time_array, height_array, dvar, dvar_min_hgt, zmax=3)