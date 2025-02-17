# import necessary packages, functions
import os
import glob
import math
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from datetime import datetime, timedelta
from climlab.solar.orbital import OrbitalTable
from climlab.solar.insolation import daily_insolation, instant_insolation


def find_growth_rate(lat, doy):
    
    '''
    This function calculates a maximum CBL depth growth rate based on
    the latitudinal variation of daily maximum insolation and day of the
    year. Insolation values are scaled by observed maximum growth rates 
    from various WSR-88D locations across winter (minimum max growth rate
    of ~300 m/hr in northern latitudes) and summer (maximum max growth
    rate of ~1750 m/hr in southern latitudes). Scaling is latitudinally 
    linear and applied equally across the entire year.
    
    ---------------------------------------------------------------------
    
    Inputs:
    
    lat : int or float
        Latitude (degrees N).
    doy : int
        Day of year.
    
    Outputs:
    
    max_growth_rate : float
        Maximum CBL depth growth rate expected at given latitude on
        given doy in units of m/hr.
    
    ---------------------------------------------------------------------
    '''
    
    min_lat = 24 # approximate southernmost CONUS latitude (KBYX)
    max_lat = 50 # approximate northernmost CONUS latitude (KMBX)
    
    # generate insolation values across entire year for given latitude range
    lats  = np.arange(min_lat,max_lat+.01,.1)
    days  = np.arange(0,365,1)
    insol = instant_insolation(lats, days, lon = -180)
    
    min_insol = np.min(insol)
    max_insol = np.max(insol)
    
    # upper scaling factor = maximum desired growth rate at 24 degN (1750 m/hr) 
    # divided by insolation 24 degN on summer solstice
    upper_fac = 1750/max_insol
    
    # lower scaling factor = minimum desired growth rate at 50 degN (300 m/hr)
    # divided by insolation 50 degN on winter solstice
    lower_fac = 300/min_insol
    
    # generate an array with many steps between lower_fac and upper_fac 
    spaced    = np.linspace(lower_fac,upper_fac,10000)
    
    # find approximate index of latitude of interest in spaced array to determine scaling factor
    lat_index = round(((max_lat-lat)/(max_lat-min_lat))*len(spaced))
    mfac      = spaced[lat_index]
    
    # calculate maximum growth rate from daily max insolation, scaled by mfac
    max_growth_rate = mfac*instant_insolation(lat, doy, lon = -180)
    
    return max_growth_rate
    
def find_dvar_min_v2(mm, timezone, time_array, height_array, dvar,
                  max_growth, min_growth = -300, hstep = 20, smooth=False):
    
    '''
    This function contains a (somewhat) simple algorithm to detect the
    channel of reduced D-Var in QVPs that is associated with CBL top.
    
    ---------------------------------------------------------------------
    
    Inputs:
    
    mm : int
        Month of year.
    timezone : str
        Timezone of radar site (options: 'E', 'C', 'M', 'P'; e.g., 'C' for 
        Central US).
    time_array : array
        Array containing UTC times of radar scans as fractional hours.
    height_array : array
        Array containing height of radar beam at each range gate.
    dvar : array with shape [len(height_array), len(time_array)]
        Array containing calculated values of D-Var.
    max_growth : float
        Maximum growth rate of CBL depth (m/hr) calculated from max_growth_rate
        function.
    min_growth : int or float
        Minimum growth rate of CBL depth (m/hr; <0 allows for CBL depth to
        decrease after a certain time of day). Currently preset to -300 but
        likely to depend on month in the future (2/15/2024).
    hstep : int
        Approximate difference between consecutive heights (m) in height_array.
        Currently preset to 20 m as that is the value for QVPs using 4-4.5
        degree elevation angle in the lowest 3 km.
    smooth : boolean, default = False
        If true, applies smoothing to the output array of estimated CBL depth.
    
    Outputs:
    
    dvar_min_hgt : array
        Array of values estimating CBL depth for each scan time.
    
    ---------------------------------------------------------------------
    '''

    # define variables, create empty arrays
    ntimes       = len(time_array)
    max_steps    = np.zeros([ntimes-1])
    min_steps    = np.zeros([ntimes-1])
    dvar_min_hgt = np.zeros([ntimes-1])
    dvar_min_val = np.zeros([ntimes-1])
    dvar_min_ind = np.zeros([ntimes-1])
    winter_mms   = [11,12,1,2]
    
    # determine time adjustment from LST to UTC
    if timezone   == 'E': t_adj = 4
    elif timezone == 'C': t_adj = 5
    elif timezone == 'M': t_adj = 6
    elif timezone == 'P': t_adj = 7


    # restrict to increasing during first number of hours of the QVP, set min_growth rate for the rest
    # force an increase dictated by mod_num over a period from first time to t_force_inc and while below
    # min_hgt_thresh based on winter/non-winter
    # also define init_hours = first xx hours to search for initial minimum
    if mm in winter_mms:
        t_force_inc    = 12 + t_adj
        mod_num        = 2
        min_hgt_thresh = 300
        init_hours     = 3
        fac1           = 1
        fac2           = 1
    else:
        t_force_inc    = 12 + t_adj
        mod_num        = 1
        min_hgt_thresh = 500
        init_hours     = 2
        fac1           = .5
        fac2           = .3
        
    # assign maximum and minimum steps in height to search for D-Var minimum
    for t_ind in range(0,ntimes-1):
        # calculate time steps in fractional hours between each radar scan
        tstep = time_array[t_ind+1] - time_array[t_ind]

        # calculate constant possible growth rate through the entire day
        if (time_array[t_ind] < 14 + t_adj):
            max_steps[t_ind] = int(round(max_growth*tstep/hstep))
        elif (time_array[t_ind] < 16 + t_adj):
            max_steps[t_ind] = int(round(fac1*max_growth*tstep/hstep))
        else:
            max_steps[t_ind] = max(1,int(round(fac2*max_growth*tstep/hstep)))

        # set minimum step during forced growth based on variables above
        if (time_array[t_ind] < t_force_inc):
            if (np.mod(t_ind,mod_num) == 0):
                min_steps[t_ind] = 1
            else:
                min_steps[t_ind] = 0
            
        # set minimum step during rest of QVP
        # keeps CBL depth constant or increasing during first xx hours in summer months
        elif (time_array[t_ind] < 12 + t_adj) and mm not in winter_mms:
            min_steps[t_ind] = 0
        else:
            min_steps[t_ind] = int(round(min_growth*tstep/hstep))
    #print(max_steps)
    #print(min_steps)
    # search for initial minimum at lowest data levels during first init_hours of data
    # looks in three levels (1, 3, 5; ~ lowest 100 m of data) then averages the indices
    start_time  = round(time_array[0])
    t_end       = np.where(time_array > start_time + init_hours)[0][0]
    init_ind1   = np.argmin(dvar[1,0:t_end])
    init_ind2   = np.argmin(dvar[3,0:t_end])
    init_ind3   = np.argmin(dvar[5,0:t_end])
    init_ind    = round(np.mean([init_ind1,init_ind2,init_ind3]))
    
    # mask bottom levels of D-Var with large value after init_ind to keep algorithm from
    # sticking to the bottom of the QVP (which can be noisy with low D-Var values)
    modified_dvar = dvar.copy()
    modified_dvar[0:2,init_ind+1:] = 100
    
    # initialize current index
    current_ind = 0
    
    # loop over times with matching min/max steps
    for t_ind, min_step, max_step in zip(range(0,ntimes-1), min_steps, max_steps):

        # set D-Var minimum index to zero at times before the initial minimum index
        if t_ind <= init_ind:
            current_ind  = 0
            dvar_min_hgt[t_ind] = height_array[current_ind]
            
        # search the remaining time steps for the next D-Var minimum 
        # within the min/max steps surrounding the current index    
        else:

            # if the time is before 12 LST and the current height is less than min_hgt_thresh,
            # hold growth to constant or increasing...this helps keep the algorithm from decreasing
            # back toward the lowest levels and allows it to latch on to the channel of minimum values
            if (time_array[t_ind] < 12 + t_adj) and height_array[current_ind] > min_hgt_thresh:
                min_step = 0
                
            # define indices of lower and upper search bounds
            lower_bound  = max(0,int(current_ind + min_step))
            upper_bound  = int(current_ind + max_step)
            #print(time_array[t_ind])
            #print(current_ind,min_step,max_step)
            #print(lower_bound,upper_bound)
            # find value of D-Var that is an additional max_step above current index
            upper_plus_max = modified_dvar[int(upper_bound+max_step),t_ind]
            
            # conditional statement that helps the algorithm with cases of rapid growth where
            # the vertical gradient in D-Var is weak or slightly positive and the channel
            # is not the minimum value within the search bounds
            # these cases typically occur before 16 LST
            # triggers when the value of D-Var at an additional max_step above current index
            # is lower than the value at the upper bound two time steps forward
            # additional requirement ensures that trigger only occurs in regions of relatively
            # low values of D-Var
            if (9 + t_adj < time_array[t_ind] < 13 + t_adj) and \
                (upper_plus_max < modified_dvar[upper_bound,t_ind+2]) and upper_plus_max < 80:
                
                # shift the search window up by max_step
                lower_bound = upper_bound
                upper_bound = int(round(upper_bound+(max_step/2)))
                min_step   += max_step
            
            # very rarely, there will be two consecutive radar files within a very short time,
            # leading to lower_bound = upper_bound
            # quick fix here makes sure that the following np.argmin() is not searching in an
            # empty sequence

            if lower_bound >= upper_bound:
                upper_bound = lower_bound + 1
            
            # search for index of minimum value within search window, add min_step since argmin
            # returns 0 for lowest index
            #print(time_array[t_ind],lower_bound,upper_bound)
            current_ind += int(np.argmin(modified_dvar[lower_bound : upper_bound, t_ind]) + min_step)
            
            # ensure that the current index does not decrease to the lowest level of D-Var or become negative
            current_ind  = max(1,current_ind)

            # fill out empty D-Var minimum height array with height of the current index
            dvar_min_hgt[t_ind] = height_array[current_ind]
            dvar_min_val[t_ind] = dvar[current_ind,t_ind]
            dvar_min_ind[t_ind] = current_ind

    # since last time is left out of loop above, append an extra value to the end of D-Var minimum height array
    dvar_min_hgt = np.insert(dvar_min_hgt,-1,height_array[current_ind])
    dvar_min_val = np.insert(dvar_min_val,-1,dvar[current_ind,t_ind])
    dvar_min_ind = np.insert(dvar_min_ind,-1,current_ind)
    
    # apply smoothing to final array if wanted
    if smooth: 
        dvar_min_hgt = savgol_filter(dvar_min_hgt, window_length=25, polyorder=2)
    
    return dvar_min_hgt, dvar_min_val, dvar_min_ind