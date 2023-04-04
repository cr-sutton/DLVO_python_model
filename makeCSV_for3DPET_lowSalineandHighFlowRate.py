# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:25:15 2021

@author: Czahasky
"""


# Only packages called in this script need to be imported
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import StrMethodFormatter

from scipy import integrate


# package for making font bigger
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
fs = 12
plt.rcParams['font.size'] = fs


path2data = "C:\\Users\\colli\\Documents\\Research\\Manuscripts\\Kaolinite Multivalent Cations\\manuscript_data\\ottawaSand_kaolinite_PET_data\\" # VY PC - for May2022 exp
filename ='64Cu_120min_low_saline_transition_32_32_159_60.raw'
z_axis_crop = [27, 147] #050922 - cmc_low_repeat
# manually set the dimensions of the raw file
img_dim = [32, 32, 159, 60] #should crop the same size square for both Tuesday and wednesday data if use the same core

# factor by which data will be coarsened/averaged (i.e. combine 2 voxels in 1 for 1D, or 8 into 1 voxels for 3D)
coarsen_factor = 2
# voxel size. This information can be found in the .hdr img file. (Fons drive > Permanent data)
vox_size = np.array([0.07763, 0.07763, 0.0796])*coarsen_factor # (These are the default values) # numbers are voxel size cm 
# timestep size depending on reconstruction. This information can also be found in the header files
timestep_size = 60 # seconds

##### END MAIN INPUT #####

def plot_2d(map_data, dx, dy, colorbar_label, cmap, *args):

    r, c = np.shape(map_data)
    # define grid
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    X, Y = np.meshgrid(x_coord, y_coord)
    
    # define figure and with high res
    plt.figure(figsize=(10, 3), dpi=200)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.01)
    plt.gca().set_aspect('equal')  
    # 06/2022 UPDATE: font times new roman size 12
    plt.rcParams['font.size'] = '12'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # add a colorbar
    cbar = plt.colorbar() 
    # label the colorbar
    cbar.set_label(colorbar_label)
    plt.tick_params(axis='both', which='major')
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    
def plot_2d_sub_profile(map_data, dx, dy, cmax, colorbar_label, cmap):

    map_dim = np.shape(map_data)
    # define grid
    x_coord = np.linspace(0, dx*map_dim[2], map_dim[2])
    y_coord = np.linspace(0, dy*map_dim[0], map_dim[0])
    X, Y = np.meshgrid(x_coord, y_coord)
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 7), dpi=300, 
                            gridspec_kw={'height_ratios': [1.6, 1.0]})
    # define overall title    
    fig.suptitle(colorbar_label)
    # extract center slice
    center_slice_data = np.nanmean(map_data, axis=1)
    
    # define consistent max scale if not predefined in function
    if cmax == 0:
        cmax = np.nanmax(center_slice_data)
    
    # plot slice
    im1 = axs[0].pcolormesh(X, Y, center_slice_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.01, vmin=0, vmax=cmax)
    # box = axs[0].get_position()
    # axColor = plt.axes([box.x0, box.y0 +box.height* 1.05, box.width, 0.01])
    fig.colorbar(im1, ax=axs[0], orientation = 'horizontal', aspect = 50, pad = 0.2)
    axs[0].set(xlabel='Distance from inlet of column (cm)', xlim= (0, dx*map_dim[2]), ylim=(0, dy*map_dim[0]), aspect='equal', title='2D center slice average')
    axs[0].vlines(x=[0.18, 2.15, 6.5, 8.7], ymin = 0, ymax =17, color ="red", linestyle ="--")  # for 050922+low repeat

    # plot profile
    slice_average = np.nanmean(map_data, axis=(0,1))
    axs[1].plot(x_coord, slice_average, color='black', label='')
    axs[1].set(xlabel='Distance from inlet of column (cm)', xlim= (0, dx*map_dim[2]), ylim=(0, cmax), title='1D slice average')
    axs[1].vlines(x=[0.18, 2.15, 6.5, 8.7], ymin = 0, ymax =17, color ="red", linestyle ="--")  # for 050922+low repeat
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))
    plt.tight_layout()

def coarsen_slices(array3d, coarseness):
    array_size = array3d.shape
    if len(array_size) ==3:
        coarse_array3d = np.zeros((int(array_size[0]/coarseness), int(array_size[1]/coarseness), int(array_size[2]/coarseness)))
        # for z in range(0, array_size[2]):
        for z in range(0, int(array_size[2]/coarseness)):
            sum_slices = np.zeros((int(array_size[0]/coarseness), int(array_size[1]/coarseness)))
            for zf in range(0, coarseness-1):
                array_slice = array3d[:,:, z*coarseness + zf]
                
                # array_slice = array3d[:,:, z]
                # coarsen in x-y plan
                temp = array_slice.reshape((array_size[0] // coarseness, coarseness,
                                        array_size[1] // coarseness, coarseness))
                smaller_slice = np.mean(temp, axis=(1,3))
                # record values for each slice to be averaged
                sum_slices = sum_slices + smaller_slice
            # after looping through slices to be averaged, calculate average and record values
            coarse_array3d[:,:, z] = sum_slices/coarseness
            
    elif len(array_size) == 4:
        coarse_array3d = np.zeros((int(array_size[0]/coarseness), 
                    int(array_size[1]/coarseness), int(array_size[2]/coarseness), int(array_size[3])))
        print('coarsening data assuming time is 4th dimension, with no time averaging')
        # loop through time
        for t in range(0, int(array_size[3])):
            for z in range(0, int(array_size[2]/coarseness)):
                sum_slices = np.zeros((int(array_size[0]/coarseness), int(array_size[1]/coarseness)))
                for zf in range(0, coarseness-1):
                    array_slice = array3d[:,:, z*coarseness + zf, t]
                    
                    # array_slice = array3d[:,:, z]
                    # coarsen in x-y plan
                    temp = array_slice.reshape((array_size[0] // coarseness, coarseness,
                                            array_size[1] // coarseness, coarseness))
                    smaller_slice = np.mean(temp, axis=(1,3))
                    # record values for each slice to be averaged
                    sum_slices = sum_slices + smaller_slice
                # after looping through slices to be averaged, calculate average and record values
                coarse_array3d[:,:, z, t] = sum_slices/coarseness
            
    return coarse_array3d

def crop_core(array4d):
    # crop outside of core and replace values with nans
    crop_dim = array4d.shape
    # crop_dim = raw_data.shape
    rad = (crop_dim[1]-1)/2
    cy = crop_dim[1]/2 
    dia = crop_dim[1]
    for ii in range(0, dia):
        yp = np.round(cy + math.sqrt(rad**2 - (ii-rad)**2) - 1E-8)
        ym = np.round(cy - math.sqrt(rad**2 - (ii-rad)**2) + 1E-8)
        
        if yp <= dia:
            array4d[ii, int(yp):, :, :] = np.nan
        
        if ym >= 0:
            array4d[ii, 0:int(ym), :, :] = np.nan

    return array4d

def kc_calc_insitu(pet_data, times, S_timeframe, dim):
    # Use equation 14 from https://pubs.acs.org/doi/10.1021/es025871i to interpret despositional patterns
    # note that porosity and bulk density are neglected to make units consistent between S and C
    
    # determine input data size
    petdim = pet_data.shape
    # calculate kc in 1, 2, or 3 dimensions
    if int(dim) == 1 :
        kc = np.zeros((petdim[2]), dtype=float)
        C_int = np.zeros((petdim[2]), dtype=float)
        slice_sum_c = np.nansum(pet_data[:,:,:,:], axis=(0,1))

        for cslice in range(0, petdim[2]):
            # numerically integrate concentration in slice with respect to time
            c_slice_int = np.trapz(slice_sum_c[cslice,:S_timeframe], times[:S_timeframe])
            C_int[cslice] = c_slice_int
            ### NOTE ####
            # There are two ways this could be done. Either use a single time frame to calculate S
            s_slice = slice_sum_c[cslice, S_timeframe]
            # or use the average of several late time frames
            s_slice = np.nanmean(slice_sum_c[cslice, S_timeframe:S_timeframe+5])
            kc[cslice] = s_slice/c_slice_int
            
    elif int(dim) == 2: 
        raise ValueError("kc map calculation has not yet been implemented in 2D")
        # Implement 2D kc calc...
        
    elif int(dim) == 3:
        # preallocate kc
        kc = np.zeros((petdim[0:3]), dtype=float)
        C_int = np.zeros((petdim[0:3]), dtype=float)
        # slice_sum_c = np.nansum(np.nansum(pet_data[:,:,:,:], axis=0), axis=0)

        for cslice in range(0, petdim[2]):
            # numerically integrate concentration in slice with respect to time
            # c_slice_int = np.trapz(slice_average_c[cslice,:], times)
            # c_slice_check = 0
            for row in range(0, petdim[0]):
                for col in range(0, petdim[1]):
                    # check that voxel is inside of the column
                    if np.isfinite(pet_data[row,col,cslice,0]):
                        # define breakthrough curve for voxel
                        vox_c = np.squeeze(pet_data[row, col, cslice,:])
                        # integrate voxel btc
                        c_vox_int = np.trapz(vox_c[:S_timeframe], times[:S_timeframe])
                        
                        #### TO DO #### 
                        ## Need to remove the effect of the immobile activity from integral to more accurately calculate kc
                        ## Current approach underestimates kc
                        ## timethreshold option #1 = first moment
                        # first_moment = np.trapz(vox_c*times, times)/c_vox_int
                        # first_moment_index = np.argmin(abs(times-first_moment))
                        ## option #2 = peak activity
                        
                        ## mean S*
                        # sstar_aprox = np.mean(vox_c[first_moment_index:])*(times[-1] - times[first_moment_index])*1.2
                        # c_vox_int_mobile = c_vox_int - sstar_aprox
                        
                        
                        # save integration of concentration curve
                        C_int[row, col, cslice] = c_vox_int
                        ### NOTE ####
                        # There are two ways this could be done. Either use a single time frame to calculate S
                        # s_vox = pet_data[row, col, cslice, S_timeframe]
                        # or use the average of several late time frames
                        s_vox = np.nanmean(pet_data[row, col, cslice, S_timeframe:S_timeframe+5])

                        kc[row, col, cslice] = s_vox/c_vox_int
                    
    else:
        raise ValueError("dim variable describes dimensionality of kc map, it must be 1, 2, or 3")
    return kc, C_int

# Calculate probability density functions of input 'data'
def pdf_plot(data, nbins, thresh):
    # remove all nan values
    data = data[~np.isnan(data)]
    # remove all zeros and calculate histogram
    den, b = np.histogram(data[data>thresh], bins=nbins, density=True)
    # calculate probility density when x-axis doesn't go from 0-1
    uden = den / den.sum()
    bin_centers = b[1:]- ((b[2]- b[1])/2)
    
    return uden, bin_centers

## LOAD DATA and reshape
path = os.path.join(path2data, filename)
raw_data = np.fromfile(path, dtype =np.float32)
raw_data = np.reshape(raw_data, (img_dim[3], img_dim[2], img_dim[1], img_dim[0])) #reshape array from 1D to 4D
raw_data = np.transpose(raw_data, (2, 3, 1, 0)) # reorient data into order: x, y ,z time
# flip so that correctly oriented on slice plots with 0,0 in lower left
raw_data = np.flip(raw_data, axis=0) # flipping upside down to look exactly like Image J (plot_2D(raw_data) to confirm) bc ImageJ y axis goes from 0 at bottom upward but python matrix goes from top to bottomw
# crop extra long timesteps
raw_data = raw_data[:,:,z_axis_crop[0]:z_axis_crop[1],:] # for when ODD number of voxels cropped, (crop from outlet (left) to inlet (right) like ImageJ)
# flip so that tracer flows from left to right
raw_data = np.flip(raw_data, axis=2 )# flip left <--> right. Tracer now enters from the left
# extract dimension
r, c, z, t = np.shape(raw_data) #r = rows (y axis)' c= columsn( x axis), z = z-azis/slices, t= time ()


# coarsen data
if coarsen_factor > 1:
    raw_data = coarsen_slices(np.squeeze(raw_data), coarsen_factor)
raw_data = crop_core(raw_data)

plot_2d(raw_data[:,:,4, 4], vox_size[0], vox_size[1], 'Check crop', cmap='Greys') #showing 2D slice at slice #10, t =6 sec
# update dimensions
r, c, z, t = np.shape(raw_data)
plt.show()

### Normalize the data
# Option 1: Use center cells of inlet slice to normalize
# mean_max_inlet = np.max(np.mean(np.max(np.max(raw_data[:,:,0:1,:], axis=0), axis=0), axis=0), axis=0)
area = 3.1415*(2.54/2)**2
# Option 2: sum all amount of tracer
sum_inlet = np.nansum(raw_data[:,:,:,:], axis=(0,1,2)) /(r*c*z) ### -> objective: find total amount of tracer injected into the column C0_TRACER
# Plot core average to check
plt.figure(figsize=(7, 4), dpi=200)
######plt.plot(np.arange(0,60), sum_inlet)
plt.plot(np.arange(0,60), sum_inlet)
plt.xlabel('Timeframe')
plt.ylabel('Total activity (uncalibrated values)'), plt.show(), plt.close()
raw_data = raw_data/ np.sum(sum_inlet)

# define time array
times =  np.linspace(0, timestep_size*t, t)+(timestep_size/2) # Q: WHY add timestep_size/2???
# define grid coordinates
z_coord = np.linspace(0, vox_size[2]*z, z)+(vox_size[2]/2)

### CENTER SLICE AVERAGE ANALYSIS OF ADVECTIVE PERIOD
# Specify timesteps to plot concentration maps 
t1 = 12 # 050922 pulse completely left after 12 min
t2 = 16#  min  
# center slice average
plot_2d(np.nanmean(raw_data[:,:,:,t1], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.xlabel('Distance from inlet (cm)')
plt.clim(0.0, 0.4) #0,0.8
# plt.clim(0.0, 0.7) # for 050922_low_repeat
plt.title('Center slice average '+ str(t1)+ ' min')
# plt.savefig(filename[0:32]+'_t6.svg', dpi=300)
plot_2d(np.nanmean(raw_data[:,:,:,t2], axis=1), vox_size[0], vox_size[2], 'C/C$_0$', cmap='Reds')
plt.xlabel('Distance from inlet (cm)')
plt.clim(0.0, 0.04)
plt.title('Center slice average '+ str(t2)+ ' min')
plt.show(), plt.close()

data_size = raw_data[:,:,:,0].shape
save_filename = 'may2022_cu64_lowSaline_0.csv'
save_data = np.append(raw_data[:,:,:,0].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')

data_size = raw_data[:,:,:,15].shape
save_filename = 'may2022_cu64_lowSaline_15.csv'
save_data = np.append(raw_data[:,:,:,15].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')

data_size = raw_data[:,:,:,28].shape
save_filename = 'may2022_cu64_lowSaline_28.csv'
save_data = np.append(raw_data[:,:,:,27].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')

data_size = raw_data[:,:,:,29].shape
save_filename = 'may2022_cu64_lowSaline_29.csv'
save_data = np.append(raw_data[:,:,:,59].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')

data_size = raw_data[:,:,:,35].shape
save_filename = 'may2022_cu64_lowSaline_35.csv'
save_data = np.append(raw_data[:,:,:,59].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')


data_size = raw_data[:,:,:,45].shape
save_filename = 'may2022_cu64_lowSaline_45.csv'
save_data = np.append(raw_data[:,:,:,50].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')


data_size = raw_data[:,:,:,59].shape
save_filename = 'may2022_cu64_lowSaline_59.csv'
save_data = np.append(raw_data[:,:,:,55].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')


data_size = raw_data[:,:,:,45].shape
save_filename = 'may2022_cu64_lowSaline_45.csv'
save_data = np.append(raw_data[:,:,:,45].flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')




