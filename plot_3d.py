# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:45:58 2022

@author: Czahasky
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
fs = 12
plt.rcParams['font.size'] = fs

# os.chdir("Z:\Le\PET_data_cz") # Vy PC path to Nov conc data
os.chdir("C:\\Users\colli\Documents\Python Scripts") #path to 050322 conc data

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def half_core(data):
    r,c,s = np.shape(data)
    data = data[:,:-round(c/2),:]
    ncol = round(c/2)
    return data, ncol


# Import arrival time data
# data_filename = '112421_cfc_baa_3min_conc'
data_filename ='may2022_cu64_lowSaline_0'
# Import data
data3d = np.loadtxt(data_filename + '.csv', delimiter=',')

dz = data3d[-1] # voxel size in z direction (parallel to axis of core)
dy = data3d[-2] # voxel size in y direction
dx = data3d[-3] # voxel size in x direction
nslice = int(data3d[-4])
nrow = int(data3d[-5])
ncol = int(data3d[-6])

data3d = data3d[:-6].reshape(nrow, ncol, nslice)
data3d[np.isnan(data3d)]=0

# crop core
data3d, ncol = half_core(data3d)

# swap axes
#data3d = np.flip(data3d, 0)
data3d = np.swapaxes(data3d,0,2)

# generate grid    
X, Y, Z = np.meshgrid(np.linspace(dy/2, (ncol-2)*dy+dy/2, num=(ncol+1)), \
                      np.linspace(dz/2, (nslice-2)*dz+dz/2, num=(nslice+1)), \
                      np.linspace(dx/2, (nrow-2)*dx+dx/2, num=(nrow+1)))


angle = -25
fig = plt.figure(figsize=(12, 9), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.view_init(30, angle)
# ax.set_aspect('equal') 
 
# norm = matplotlib.colors.Normalize(vmin=data3d.min().min(), vmax=data3d.max().max())
# norm = matplotlib.colors.Normalize(vmin=data3d.min().min(), vmax=0.000151)
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1) #set scale by max value
# ax.voxels(filled, facecolors=facecolors, edgecolors='gray', shade=False)
ax.voxels(X, Y, Z, data3d, facecolors=plt.cm.Reds(norm(data3d)), \
          edgecolors='grey', linewidth=0.2, shade=False, alpha=0.7)   #linewidth=0.2, shade=False, alpha=0.7

m = cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
m.set_array([])

divider = make_axes_locatable(ax)
# adjust size of colorbar
plt.colorbar(m, shrink=0.24) #shrink=0.24
# format axes
set_axes_equal(ax)
# ax.set_xlim3d([0, 4])
ax.set_axis_off()
# ax.invert_zaxis()
# Set background color to white (grey is default)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.savefig('may2022_cu64_lowSaline_0.png', transparent=True)
# i=1
# plt.savefig(data_filename + '.svg', format="svg")
# plt.show()
plt.close()


