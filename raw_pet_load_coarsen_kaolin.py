# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:25:15 2021

@author: Czahasky
"""


# Only packages called in this script need to be imported
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

from scipy import integrate
mpl.rcParams['figure.dpi'] = 300
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 10})

# load data
# filename = '02May2022_F18_45min_1mL_ottawa_32_32_159_45.raw'
# img_dim = [32, 32, 159, 45]

# filename = '64Cu_55min_1mL_highSaline_kaolinite_32_32_159_55.raw'
# img_dim = [32, 32, 159, 55]

path2data = "C:\\Users\\colli\\Documents\\Research\\Manuscripts\\Kaolinite Multivalent Cations\\manuscript_data\\ottawaSand_kaolinite_PET_data\\" # VY PC - for May2022 exp
filename ='64Cu_120min_low_saline_transition_32_32_159_60.raw'
img_dim = [32, 32, 159, 60] #full PET


vox_size = [0.07763, 0.07763, 0.0796]
timestep_size = 60 # seconds
# timestep_size = 120 # seconds
# (easiest to just check in FIJI)
end_slice_locations = [27, 147]


def plot_2d(map_data, dx, dy, colorbar_label, cmap, *args):

    r, c = np.shape(map_data)

    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    
    X, Y = np.meshgrid(x_coord, y_coord)
    
    # fig, ax = plt.figure(figsize=(10, 10) # adjust these numbers to change the size of your figure
    # ax.axis('equal')          
    # fig2.add_subplot(1, 1, 1, aspect='equal')
    # Use 'pcolor' function to plot 2d map of concentration
    # Note that we are flipping map_data and the yaxis to so that y increases downward
    plt.figure(figsize=(12, 4), dpi=200)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.01)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    if args:
        plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label)
    # make colorbar font bigger
    # cbar.ax.tick_params(labelsize= (fs-2)) 
    # make axis fontsize bigger!
    plt.tick_params(axis='both', which='major')
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    

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


raw_data = np.fromfile((path2data + '\\' + filename), dtype=np.float32)
raw_data = np.reshape(raw_data, (img_dim[3], img_dim[2], img_dim[1], img_dim[0]))
raw_data = np.transpose(raw_data, (2, 3, 1, 0))
# flip so that correctly oriented on slice plots with 0,0 in lower left
raw_data = np.flip(raw_data, axis=0)
# crop extra long timesteps
raw_data = raw_data[:,:,end_slice_locations[0]:end_slice_locations[1],:]
raw_data = np.flip(raw_data, axis=2)

coarse_PET = coarsen_slices(raw_data, 2)
coarse_PET = coarse_PET[:,:,:,:29]

colorbar_max = 0.05
plot_2d(raw_data[:,16,:,5], vox_size[1], vox_size[2], '[-]', cmap='viridis')
plt.clim(0.0, colorbar_max)
plot_2d(raw_data[:,:,10,5], vox_size[0], vox_size[1], '[-]', cmap='viridis')
plt.clim(0.0, colorbar_max)


plot_2d(coarse_PET[:,8,:,5], vox_size[1]*2, vox_size[2]*2, '[-]', cmap='viridis')
plt.clim(0.0, colorbar_max)
plot_2d(coarse_PET[:,:,5,5], vox_size[0]*2, vox_size[1]*2, '[-]', cmap='viridis')
plt.clim(0.0, colorbar_max)
# plot_2d(raw_data[15,:,46:120,-1], vox_size[0], vox_size[1], '[-]', cmap='viridis')
# plt.clim(0., 0.3)


## Define time array
time_array = np.arange(timestep_size/2, timestep_size*(img_dim[3]-2), timestep_size)
r,c,s,ts = raw_data.shape
z_coord = np.linspace(0, vox_size[2]*s, s)

#timesteps_to_plot = 60
timesteps_to_plot = 5 #for tracer
#timesteps_to_plot = 55 #for high saline
#timesteps_to_plot = 29 # for low saline
#tcolors = plt.cm.Greys(np.linspace(0,1,(timesteps_to_plot)))
tcolors = plt.cm.Reds(np.linspace(0,1,(timesteps_to_plot)))
#tcolors = plt.cm.Greys(np.linspace(0,1,(7)))
# Breakthrough curves
fig, axis = plt.subplots(1,1, figsize=(7,3), dpi=300)
ind = 0
# # loop through time and plot concentration profiles
# for ti in range(29, timesteps_to_plot):
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind])
#     ind +=1

# plt.xlabel('Distance from inlet [cm]')
# plt.title('High flow rate injection')

#pore volumes for time related x-axis for tracer
# poreV = np.linspace(0, time_array[-1]/60, 120)
# poreV = poreV/18.41
# loop through time and plot concentration profiles for tracer
#for ti in range(0, timesteps_to_plot,8):
# for ti in range(0, 44): 
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)
#     slice_average_concentration = slice_average_concentration
#     pv = ti*2/18.41

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
#     ind +=1

# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('Tracer injection')
# #plt.legend(prop={'size': 9})
# plt.savefig('tracerBTC.png', transparent=True)
# plt.close()
#####plots these specific lines
x = [0, 3, 9, 29]
labels1 = ["PV = 0", "PV = 0.22", 'PV = 0.49', "PV = 0.87"]
# loop through time and plot concentration profiles for tracer
ind = 0
for ti in x[:]:
    slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)
    slice_average_concentration = slice_average_concentration
    pv = ti*2/18.41

    # plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
    plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label = labels1[ind])
    
    ind +=1
plt.yscale("log")
plt.ylim(2.5, 1E3)
plt.xlabel('Distance from inlet [cm]')
#plt.ylabel('Slice Average Radioconcentration')
#plt.title('Radiotracer Injection')
plt.legend(loc='lower right', fontsize=12)
#plt.savefig('highSaline_may_BTC.png', transparent=True)
# plt.savefig('ottawa_PET_tracer_AGU2022.png', transparent=True)
plt.show(), plt.close()
# plt.legend()

a = (raw_data[:,:,:, -1])
from scipy import ndimage
cm = ndimage.center_of_mass(a)

fig, ax = plt.subplots(1,1)
colorbar_max = 0.05
# plot_2d(raw_data[:,16,:,5], vox_size[1], vox_size[2], '[-]', cmap='viridis')

# ax.imshow(raw_data[:,:,10,5])
# # plt.clim(0.0, colorbar_max)
# ax.scatter(cm[0], cm[1])
# plot_2d(raw_data[:,:,10,5], vox_size[0], vox_size[1], '[-]', cmap='viridis')
# plt.clim(0.0, colorbar_max)

ax.imshow(raw_data[:,16,:,29], origin = "lower")
# plt.clim(0.0, colorbar_max)
ax.scatter(cm[2], cm[0])

cmMM = 0.0796*cm[2]

# x = [3, 6, 12, 54]
# # loop through time and plot concentration profiles for tracer
# for ti in x[:]:
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)
#     slice_average_concentration = slice_average_concentration
#     pv = ti*2/18.41

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
#     ind +=1
# plt.yscale("log")
# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('Radiolabeled Kaolinite + 100mM NaCl injection')
# #plt.savefig('highSaline_may_BTC.png', transparent=True)
# plt.show(), plt.close()
# # plt.legend()
# ind = 0
# x = [3, 6, 12, 59]
# # loop through time and plot concentration profiles for tracer
# for ti in x[:]:
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)
#     slice_average_concentration = slice_average_concentration
#     pv = ti*2/10.08

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
#     ind +=1
# plt.yscale("log")
# plt.ylim(2.5, 1E3)
# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('1mM NaCl Transition Injection')
# #plt.savefig('highSaline_may_BTC.png', transparent=True)
# plt.show(), plt.close()
# plt.legend()

# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('Radiolabeled Kaolinite + 100mM NaCl injection')
# #plt.savefig('highSaline_may_BTC.png', transparent=True)
# plt.show(), plt.close()
####plot all tracer lines
# for ti in range(0, 44):
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)
#     slice_average_concentration = slice_average_concentration
#     pv = ti*2/18.41

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
#     ind +=1


# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('Radiolabeled Kaolinite + 100mM NaCl injection')
# #plt.savefig('highSaline_may_BTC.png', transparent=True)
# plt.show(), plt.close()
# # plt.legend()

# loop through time and plot concentration profiles for low saline transition
# x = [3, 6, 12, 54]
# for ti in x[:]:
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)
#     slice_average_concentration = slice_average_concentration/2
#     pv = ti*2/10.08

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
#     ind +=1

# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('1mM NaCl transition injection')
# plt.savefig('lowSaline_may_BTC.png', transparent=True)
# #plt.legend()



#####plot all lines
# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('Radiolabeled Kaolinite + 100mM NaCl injection')
# #plt.savefig('highSaline_may_BTC.png', transparent=True)
# plt.show(), plt.close()
# # plt.legend()

# # loop through time and plot concentration profiles for low saline transition
# for ti in range(0, 59):
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:, ti], axis=0), axis=0)
#     slice_average_concentration = slice_average_concentration/2
#     pv = ti*2/10.08

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
#     ind +=1

# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('1mM NaCl transition injection')
# plt.savefig('lowSaline_may_BTC.png', transparent=True)
# #plt.legend()


# for ti in range(0, 3):
#     slice_average_concentration = np.sum(np.sum(raw_data[:,:,:,ti], axis=0), axis=0)
#     slice_average_concentration = slice_average_concentration/2
#     pv = ti*2/10.08

#     plt.plot(z_coord, slice_average_concentration, color=tcolors[ind+1], label='pv=%s' % pv)
#     ind +=1

# plt.xlabel('Distance from inlet [cm]')
# plt.ylabel('Slice Average Radioconcentration')
# plt.title('1mM NaCl transition injection')
# plt.savefig('lowSaline_may_BTC.png', transparent=True)
# #plt.legend()


# axis[0].set_title('Homogeneous permeability')
# axis[0].set(ylabel='$C/C_0$ [-]',xlabel ='PV [-]')
# axis[0].set_ylim([0, 0.08])
# axis[0].set(ylabel='$C/C_0$ [-]',xlabel ='Time [minutes]', yscale='log')
# axis[0].set_ylim([1E-4, 1])
# axis[0].legend(loc ='upper right')





# ts = 60
# plot_2d(denoise3d[:,:,ts], vox_size[0], vox_size[1], 'denoise', cmap='viridis')
# plot_2d(raw_data[:,:,ts, -1], vox_size[0], vox_size[1], 'raw', cmap='viridis')
# plt.clim(400, 1600)

# data_size = Por.shape
# save_filename = 'D:\\Dropbox\\Codes\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_data'  + '\\' 'Navajo_porosity_coarsened.csv'
# save_data = np.append(Por.flatten('C'), [data_size, vox_size])
# np.savetxt(save_filename, save_data, delimiter=',')



# # # Breakthrough curves
# fig, axis = plt.subplots(1,1, figsize=(7,5), dpi=300)
# ind = 0
# # for i in range(img_dim[0]):
# i=7
# for j in range(img_dim[1]):
#     btc = data[i,j,slice_test,:]
#     smooth_data = signal.filtfilt(B,A, btc)
#     plt.plot(time_array, smooth_data, color=tcolors[ind])
#     ind +=1
    
    
    
