# import necessary packages, functions
import os
import glob
import math
import pyart
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
    
    return dvar_min_hgt
    
##########################################################################################################################################
##########################################################################################################################################
# Other functions that may be useful but are not necessary to reproduce findings:
##########################################################################################################################################
##########################################################################################################################################

# function from pyart to generate QVP of azimuthal mean value
def quasi_vertical_profile(radar, desired_angle=None, fields=None, gatefilter=None):
    #Creating an empty dictonary
    qvp = {}

    # Setting the desired radar angle and getting index value for desired radar angle
    if desired_angle is None:
        desired_angle = 20.0
    index = abs(radar.fixed_angle['data'] - desired_angle).argmin()
    radar_slice = radar.get_slice(index)

    # Setting field parameters
    # If fields is None then all radar fields pulled else defined field is used
    if fields is None:
        fields = radar.fields

        for field in fields:

            # Filtering data based on defined gatefilter
            # If none is defined goes to else statement
            if gatefilter is not None:
                get_fields = radar.get_field(index, field)
                mask_fields = np.ma.masked_where(gatefilter.gate_excluded[radar_slice],
                                                 get_fields)
                radar_fields = np.ma.mean(mask_fields, axis=0)
            else:
                radar_fields = radar.get_field(index, field).mean(axis=0)

            qvp.update({field:radar_fields})

    else:
        # Filtereing data based on defined gatefilter
        # If none is defined goes to else statement
        if gatefilter is not None:
            get_field = radar.get_field(index, fields)
            mask_field = np.ma.masked_where(gatefilter.gate_excluded[radar_slice],
                                            get_field)
            radar_field = np.ma.mean(mask_field, axis=0)
        else:
            radar_field = radar.get_field(index, fields).mean(axis=0)

        qvp.update({fields:radar_field})

    # Adding range, time, and height fields
    qvp.update({'range': radar.range['data'], 'time': radar.time})
    _, _, z = antenna_to_cartesian(qvp['range']/1000.0, 0.0,
                                   radar.fixed_angle['data'][index])
    qvp.update({'height': z})
    return qvp
    
    
# function adapted from pyart to generate QVP of azimuthal variance
def quasi_vertical_profile_variance(radar, desired_angle=None, fields=None, gatefilter=None):
    #Creating an empty dictonary
    qvp = {}

    # Setting the desired radar angle and getting index value for desired radar angle
    if desired_angle is None:
        desired_angle = 20.0
    index = abs(radar.fixed_angle['data'] - desired_angle).argmin()
    radar_slice = radar.get_slice(index)

    # Setting field parameters
    # If fields is None then all radar fields pulled else defined field is used
    if fields is None:
        fields = radar.fields

        for field in fields:

            # Filtering data based on defined gatefilter
            # If none is defined goes to else statement
            if gatefilter is not None:
                get_fields = radar.get_field(index, field)
                mask_fields = np.ma.masked_where(gatefilter.gate_excluded[radar_slice],
                                                 get_fields)
                radar_fields = np.ma.var(mask_fields, axis=0)
            else:
                radar_fields = radar.get_field(index, field).var(axis=0)

            qvp.update({field:radar_fields})

    else:
        # Filtereing data based on defined gatefilter
        # If none is defined goes to else statement
        if gatefilter is not None:
            get_field = radar.get_field(index, fields)
            mask_field = np.ma.masked_where(gatefilter.gate_excluded[radar_slice],
                                            get_field)
            radar_field = np.ma.var(mask_field, axis=0)
        else:
            radar_field = radar.get_field(index, fields).var(axis=0)

        qvp.update({fields:radar_field})

    # Adding range, time, and height fields
    qvp.update({'range': radar.range['data'], 'time': radar.time})
    _, _, z = antenna_to_cartesian(qvp['range']/1000.0, 0.0,
                                   radar.fixed_angle['data'][index])
    qvp.update({'height': z})
    return qvp  
    
# function from pyart to convert polar to cartesian
def antenna_to_cartesian(ranges, azimuths, elevations):
    theta_e = elevations * np.pi / 180.0    # elevation angle in radians.
    theta_a = azimuths * np.pi / 180.0      # azimuth angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0     # effective radius of earth in meters.
    r = ranges * 1000.0                 # distances to gates in meters.

    z = (r ** 2 + R ** 2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z
    
def smooth_QVP(var):
  sz   = np.shape(var)
  lend = sz[0]
  mend = sz[1]

  for l in range(3,lend-2):
    for m in range(2,mend-1):
      part0    = var[l-2,m-1]+var[l-2,m]+var[l-2,m+1]
      part1    = var[l-1,m-1]+var[l-1,m]+var[l-1,m+1]
      part2    = var[l,m-1]+var[l,m]+var[l,m+1]
      part3    = var[l+1,m-1]+var[l+1,m]+var[l+1,m+1]
      part4    = var[l+2,m-1]+var[l+2,m]+var[l+2,m+1]
      var[l,m] = (1/15)*(part0+part1+part2+part3+part4)
  
  return var


def create_qvp(fnames,elevation_angle=4.5):
  '''
  This function contains a (somewhat) simple algorithm to detect the
  channel of reduced D-Var in QVPs that is associated with CBL top.
  
  ---------------------------------------------------------------------
  
  Inputs:
  
  mm : list
      List of paths to radar files that will be processed.
  elevation_angle : float (default 4.5)
      Desired elevation angle to be used when creating QVPs. If
      the desired angle is not available, will default to the next
      lowest elevation angle.
  
  Outputs:
  
  dvar : array with shape [len(height_array), len(time_array)]
      Array containing calculated values of D-Var.
  zdr : array with shape [len(height_array), len(time_array)]
      Array containing calculated values of ZDR.
  zdrvar : array with shape [len(height_array), len(time_array)]
      Array containing calculated values of ZDR variance.
  time_array : array
      Array containing UTC times of radar scans as fractional hours.
  height_array : array
      Array containing height of radar beam at each range gate.
  lat : float
      Latitude of radar (degrees N).
  ---------------------------------------------------------------------
  '''
  # get number of files and length of file name
  nfiles   = len(fnames)
  file_len = len(fnames[0])
  
  # read first file outside of loop to get the lat, lon & height array
  radtest      = pyart.io.read_nexrad_archive(fnames[0])
  lat          = radtest.latitude['data'][0]
  lon          = radtest.longitude['data'][0]
  vartest      = quasi_vertical_profile(radar=radtest,desired_angle=elevation_angle,fields='differential_reflectivity')
  height_array = vartest['height'] # height stays the same for each file
  nheight      = len(height_array)
  
  # empty matrices to store QVPs, times
  zdr        = np.empty([nfiles,nheight])
  zdrvar     = np.empty([nfiles,nheight])
  time_array = np.empty(nfiles)
  
  # loop through all radar files
  for idx, fname in enumerate(fnames):

    print('Reading data of {}'.format(fname))
    
    # read in each file using pyart, occasionally need to skip one if
    # there is an error reading the file
    try:
      radar        = pyart.io.read_nexrad_archive(fname)
    
    # if there is an error, copy QVPs from previous file into current time
    except:
      zdr[idx]        = zdr[idx-1]
      zdrvar[idx]     = zdrvar[idx-1]
      time_str        = fname[file_len-10:file_len-4] # should work as long as files end with 'V06'
      hour            = float(time_str[0:2])
      minute          = float(time_str[2:4])
      second          = float(time_str[4:6])
      time_array[idx] = hour + (minute*60 + second)/3600 # UTC time
      #os.remove(fname)
      continue
      
    # converts data to QVP using pyart
    var1      = quasi_vertical_profile(radar=radar,desired_angle=elevation_angle,fields='differential_reflectivity')
    var2      = quasi_vertical_profile_variance(radar=radar,desired_angle=elevation_angle,fields='differential_reflectivity')
    
    # extract necessary variables
    zdr[idx]    = var1['differential_reflectivity']
    zdrvar[idx] = var2['differential_reflectivity']
    
    # get the time from the radar file name, convert to fractional hour
    time_str        = fname[file_len-10:file_len-4]
    hour            = float(time_str[0:2])
    minute          = float(time_str[2:4])
    second          = float(time_str[4:6])
    time_array[idx] = hour + (minute*60 + second)/3600 # UTC
    #os.remove(fname)
    
  print('Done with QVP creation')
  
  # transpose matrices for plotting/algorithm purposes
  zdr    = np.matrix.transpose(zdr)
  zdrvar = np.matrix.transpose(zdrvar)
  
  # compute DVar
  dvar   = (np.absolute(zdr) + 1) * zdrvar
  
  return dvar, zdr, zdrvar, time_array, height_array, lat
  
def plot_QVP(time, height, dvar, dvar_min_hgt, zmax=3.5):
    
  plt.figure()
  vmin = 0
  vmax = 80
  pm = plt.contourf(time,height/1000,dvar,cmap='nipy_spectral',vmin=vmin,vmax=vmax,levels=np.arange(vmin,vmax+.1,.5),extend='max')
  plt.plot(time,dvar_min_hgt/1000,color='white')
  
  plt.colorbar(pm,orientation='vertical',ticks=np.arange(vmin,vmax+1,10),label='D-Var (dB$^3$)')
  
  plt.ylim(0,zmax)
  plt.ylabel('Height (km)')
  plt.xlabel('Time (UTC)')
  plt.show()
  
  return
  
