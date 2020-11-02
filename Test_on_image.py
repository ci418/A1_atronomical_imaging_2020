# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:52:06 2020

@author: Charalambos Ioannou

Tests the code on a small part of the image
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm

#%%

hdulist = fits.open("mosaic.fits")
data = hdulist[0].data
test_data = data[625:913, 1661:1949]

ZP = hdulist[0].header["MAGZPT"] #calibration value
ZP_error = hdulist[0].header["MAGZRR"] #calibration error

#plots data to be tested
plt.imshow(test_data, norm = LogNorm())
plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.show()
#%%
def max_value(x):
    'Returns position of maximum value in 2d array'
    pos = np.where(x == np.max(x))
    row_index = pos[0][0]
    column_index = pos[1][0]
    return row_index, column_index

r = 3 #initial aperture radius

def make_aperture(c_index, r_index, radius):
    'Defines square aperture around maxumum point'
    x_1 = c_index - radius
    x_2 = c_index + radius
    y_1 = r_index - radius
    y_2 = r_index + radius
    return x_1, x_2, y_1, y_2

def calibration(ZP, counts):
    'Calibrates the value of the flux of the source'
    z = ZP - 2.5 * np.log10(counts)
    return z

#%%

test_data = data[625:913, 1661:1949].copy()

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
        
    radius = 2 #initial radius
    
    source = []
    background = []
    
    if test_data[r][c] < low_threshold:
        break
            
    elif test_data[r][c] >= low_threshold:
            
        while len(background) < len(source) + 1:
            
            x_1, x_2, y_1, y_2 = make_aperture(c, r, radius)
            
            #stop aperture if it reaches the edge of the data set
            if x_1 < 0 or x_2 > shape[1] or y_1 < 0 or y_2 > shape[0]:
                test_data[y_1:y_2, x_1:x_2][test_data[y_1:y_2, x_1:x_2] \
                                            >= low_threshold] = 0
                break
            
            aperture = test_data_loop[y_1:y_2, x_1:x_2]
                
            background = aperture[aperture < low_threshold]
            source = aperture[aperture >= low_threshold]
            
            radius += 1
            
            #mask the detected source
            test_data[y_1:y_2, x_1:x_2][test_data[y_1:y_2, x_1:x_2] \
                                        >= low_threshold] = 0
                
            background = aperture[aperture < low_threshold]
            source = aperture[aperture >= low_threshold]
            
            total_flux = sum(source)
            mean_background = np.mean(background)
            total_background = mean_background * len(source)
            true_flux = total_flux - total_background
    
    calibrated_flux = calibration(ZP, true_flux) #calibrates flux
    catalogue.append([r, c, calibrated_flux]) #appends to catalogue
            
    r_list.append(r)
    c_list.append(c)
    radius_list.append(radius)

#%%
    
#Plots the detected sources on the data to check how accurate the code is
params = {
   'axes.labelsize': 18,
   'font.size': 18, 
} 
plt.rcParams.update(params)

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