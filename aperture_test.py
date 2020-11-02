# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:33:19 2020

@author: Charalambos Ioannou

Module testing the discovery and masking of sources
by using a square expanding aperture.
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

#Determine the brightest point in a 2d array
test_data = np.random.rand(10,15) #random data set
pos = np.where(test_data == np.max(test_data))
print(pos[0][0], pos[1][0]) #coordinates of highest point

#%%
copied_data = np.copy(test_data)
#print(np.amax(test_data))


def max_value(x):
    'Returns position of maximum value in 2d array'
    pos = np.where(x == np.max(x))
    row_index = pos[0][0]
    column_index = pos[1][0]
    return row_index, column_index

r = 3 #initial aperture radius

def aperture(c_index, r_index, radius):
    'Defines square aperture around maxumum point'
    x1 = c_index - radius
    x2 = c_index + radius
    y1 = r_index - radius
    y2 = r_index + radius
    return x1, x2, y1, y2

m_row, m_column = max_value(copied_data)#
#print max value and its coordinates
print(copied_data[m_row][m_column], m_column, m_row)

x1_test, x2_test, y1_test, y2_test = aperture(m_column, m_row, r)
#print edges of aperture and its radius
print(x1_test, x2_test, y1_test, y2_test, r) 

#%%
    
def Gaussian(sigma, shape):
    'Defines a 2D Gaussian to test code'
    r = -10
    l = 10
    mean = 0
    x, y = np.meshgrid(np.linspace(r, l, shape[0]), np.linspace(r, l, shape[1]))
    var = ((x * x) + (y * y))
    z = np.exp(- ((var - mean)**2) / (2 * (sigma * sigma)))
    return z 

sigma = 20
shape = [100, 100]
gauss1 = Gaussian(sigma, shape)

plt.imshow(gauss1)
plt.show()

#pixels with value lower than this are considered background
low_threshold = 1e-1 

radius = 2 #initial radius

#lists for background and source pixels to be appended
background = []
source = []

gauss_loop = gauss1.copy() #copy of data that will not be masked

#lists for the row and collumn coordinates to be appended
r_list = []
c_list = []

while(True):

    r, c = max_value(gauss1) #coordinates of max value
      
    r_list.append(r)
    c_list.append(c)    
    
    #if below low threshold break the while loop
    if gauss1[r][c] < low_threshold:
        break
            
    else:
        #condition determining the final size of the aperture   
        while len(background) < len(source) + 1:
            
            x1, x2, y1, y2 = aperture(c, r, radius)
            
            box = gauss_loop[y1:y2, x1:x2] #creates aperture
            
            radius += 1
            
            #mask the data corresponding to a source
            gauss1[y1:y2, x1:x2][gauss1[y1:y2, x1:x2] >= low_threshold] = 0
            
            #appends pixel to corresponding list depending on its value
            background = box[box < low_threshold]
            source = box[box >= low_threshold]
            
            #determines true flux of source
            total_flux = sum(source)
            mean_background = np.mean(background)
            total_background = mean_background * len(source)
            true_flux = total_flux - total_background
# check background and source data to make sure they make sense
# check gauss1 data to make sure that the source is eliminated

#plots the masked data to check if source is masked
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
circle = plt.Circle((r_list[0], c_list[0]), radius, color='r', fill = False)
ax.add_artist(circle)
plt.scatter(r_list[0], c_list[0], c='r')
plt.imshow(gauss1)
plt.show()