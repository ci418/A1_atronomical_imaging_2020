# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:17:40 2020

@author: Charalambos Ioannou

Tests the circular aperture on a part of the image
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm

#%%
def set_aperture(data, c_index, r_index, radius):
    'Creates a circular aperture around the a point'
    x_0 = c_index
    y_0 = r_index
    
    x_low = x_0 - radius
    x_high = x_0 + radius
    
    x = np.arange(x_low, x_high)
    
    y_low = y_0 - radius
    y_high = y_0 + radius
    
    y = np.arange(y_low, y_high)
    
    x_indices = []
    y_indices = []
    
    for i in x:
        for j in y:
            
            if ((i - x_0) ** 2) + ((j - y_0) ** 2) <= radius ** 2:
                x_indices.append(i)
                y_indices.append(j)
    
    data = data[(y_indices, x_indices)]
    return data, y_indices, x_indices

#%%
#create a 2d array
x = np.arange(0, 100)
y = np.arange(0, 100)

X,Y = np.meshgrid(x,y)

plt.imshow(X)
plt.show()

radius = 10 #radius of aperture

#coordinates of centre of aperture
c_index = 50 
r_index = 50

test, y_indices, x_indices = set_aperture(X, c_index, r_index, radius)

#mask the aperture and plot the data to check its shape
for i in range(len(x_indices)):
    X[y_indices[i]][x_indices[i]] = 0

plt.imshow(X)
plt.show()
#%%
hdulist = fits.open("mosaic.fits")
data = hdulist[0].data
test_data = data[625:913, 1661:1949]

ZP = hdulist[0].header["MAGZPT"]
ZP_error = hdulist[0].header["MAGZRR"]


plt.imshow(test_data, norm = LogNorm())
plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])

#%%
def max_value(x):
    'Returns position of maximum value in 2d array'
    pos = np.where(x == np.max(x))
    row_index = pos[0][0]
    column_index = pos[1][0]
    return row_index, column_index

def calibration(ZP, counts):
    'Calibrates the value of the flux of the source'
    z = ZP - 2.5 * np.log10(counts)
    return z

def calibration_error(ZP_error, counts, counts_error):
    'Determines the eror of the flux after calibration'
    error_log = (2.5) * (1 / np.log(10)) * (1 / counts) * (counts_error)
    error_calib = np.sqrt((ZP_error * ZP_error) + (error_log * error_log))
    return error_calib
#%%

shape = np.shape(test_data)


low_threshold = 3500

background = []
source = []


test_data_loop = test_data.copy()

r_list = []
c_list = []
radius_list = []
catalogue = []

while(True):

    r, c = max_value(test_data)
        
    radius = 2
    
    source = []
    background = []
    
    if test_data[r][c] < low_threshold:
        break
            
    elif test_data[r][c] >= low_threshold:
        
        while len(background) < len(source) + 1:
            
            aperture, y_indices, x_indices = set_aperture(test_data_loop, c, r, radius)
           
            radius += 1
            
            for i in range(len(x_indices)):
                if test_data[y_indices[i]][x_indices[i]] >= low_threshold:
                    test_data[y_indices[i]][x_indices[i]] = 0
                    
            background = aperture[aperture < low_threshold]
            source = aperture[aperture >= low_threshold]
            
            total_flux = sum(source)
            flux_error = 0.1 * total_flux
            
            mean_background = np.mean(background)
            mean_error_background = np.std(background)
            total_background = mean_background * len(source)
            error_background = mean_error_background * len(source)
            
            true_flux = total_flux - total_background
            true_flux_error = np.sqrt((flux_error ** 2) + \
                                      (error_background ** 2))
    
    calibrated_flux = calibration(ZP, true_flux)
    calibrated_error = calibration_error(ZP_error, true_flux, true_flux_error)
    
    #if source is too faint do not catalogue it
    if calibrated_flux > 20:
        continue
    
    catalogue.append([r, c, calibrated_flux, calibrated_error])
    
    r_list.append(r)
    c_list.append(c)
    radius_list.append(radius)

#%%
params = {
   'axes.labelsize': 18,
   'font.size': 18, 
} 
plt.rcParams.update(params)

#Plot the detected sources on the data

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for i in range(len(c_list)):
    circle = plt.Circle((c_list[i], r_list[i]), radius_list[i],
                   color='r',  fill=False)
    ax.add_artist(circle)
plt.scatter(c_list[0:], r_list[0:], c='r', s=5)

plt.imshow(test_data)
plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(len(c_list)):
    circle = plt.Circle((c_list[i], r_list[i]), radius_list[i],
                   color='r',  fill=False)
    ax.add_artist(circle)
plt.scatter(c_list[0:], r_list[0:], c='r', s=5)
            
plt.imshow(test_data_loop, norm = LogNorm())
plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.show()
