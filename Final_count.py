# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:11:19 2020

@author: Charalambos Ioannou

Module producing the catalogue for the whole image and analysing the data
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

#%%
hdulist = fits.open("mosaic.fits")
data = hdulist[0].data
test_data = data

#calibration value and error
ZP = hdulist[0].header["MAGZPT"]
ZP_error = hdulist[0].header["MAGZRR"]

#%%

#Bleed is masked manually at these points
data[3000:3400, 1219:1642] = 0 #eliminate the large source

data[0:, 1425:1453] = 0 #mask vertical bleed from large source

data[425:433, 1103:1652] = 0 #mask horizontal bleed
data[433:442, 1310:1526] = 0 #mask horizontal bleed
data[442:471, 1378:1477] = 0 #mask horizontal bleed

data[314:320, 1019:1702] = 0 #mask horizontal bleed
data[320:335, 1309:1535] = 0 #mask horizontal bleed
data[335:363, 1397:1474] = 0 #mask horizontal bleed

data[202:263, 1390:1475] = 0 #mask horizontal bleed

data[124:128, 1291:1524] = 0 #mask horizontal bleed
data[128:129, 1342:1505] = 0 #mask horizontal bleed
data[129:130, 1350:1490] = 0 #mask horizontal bleed
data[130:153, 1369:1481] = 0 #mask horizontal bleed
data[153:161, 1419:1432] = 0 #mask horizontal bleed

data[117:124, 1390:1467] = 0 #mask horizontal bleed

data[117:139, 1526:1539] = 0 #mask horizontal bleed

data[333:354, 1641:1649] = 0 #mask horizontal bleed

data[424:437, 1027:1043] = 0 #mask vertical bleed
data[437:451, 1036:1039] = 0 #mask vertical bleed

data[1403:1452, 2063:2115] = 0 #mask vertical bleed
data[1400:1403, 2089:2092] = 0 #mask vertical bleed

data[2276:2337, 2105:2157] = 0 #mask vertical bleed

data[3382:3446, 2440:2496] = 0 #mask vertical bleed

data[3733:3803, 2102:2164] = 0 #mask vertical bleed
data[3706:3803, 2129:2140] = 0 #mask vertical bleed

data[411:456, 1439:1480] = 0 #mask vertical bleed

data[4072:4123, 535:587] = 0 #mask vertical bleed - left side

data[3202:3418, 730:828] = 0 #mask vertical bleed - left side

data[2737:2836, 934:1020] = 0 #mask vertical bleed - left side
data[2702:2737, 969:979] = 0 #mask vertical bleed - left side

data[2223:2357, 871:937] = 0 #mask vertical bleed - left side

data[4380:4416, 1295:1331] = 0 #mask vertical bleed - left side

data[4314:4348, 1348:1383] = 0 #mask vertical bleed - left side

#eliminate edge effects

data[4511:4611, 0:] = 0 #top horizontal
data[0:115, 0:] = 0 #bottom horizontal

data[0:, 0:100] = 0 #left vertical
data[0:, 2470:2570] = 0 #right vertical

#plots the masked image
plt.imshow(test_data, norm = LogNorm())
plt.colorbar()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])

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
test_data = data.copy()

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
            true_flux_error = np.sqrt((flux_error ** 2) + (error_background ** 2))
        
    calibrated_flux = calibration(ZP, true_flux)
    calibrated_error = calibration_error(ZP_error, true_flux, true_flux_error)
    
    #eliminates faint sources from catalogue
    if calibrated_flux > 20:
        continue
    
    catalogue.append([r, c, calibrated_flux, calibrated_error])
    
    r_list.append(r)
    c_list.append(c)
    radius_list.append(radius)
    
#saves the catalogue
#np.savetxt('Catalogue', catalogue, delimiter='     ,    ', fmt='%.2f', \
#           header = 'Row coord      Column coord      Brightness')

#%%
params = {
   'axes.labelsize': 18,
   'font.size': 18, 
} 
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


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

#%%
#obtain the flux values from the catalogue
fluxes = []

for i in catalogue:
    fluxes.append(i[2])

print(min(fluxes), max(fluxes))

fluxes = np.asarray(fluxes)

#Form arrays based on a magnitude limit
flux_1 = fluxes[fluxes < 10]
flux_2 = fluxes[fluxes < 11]
flux_3 = fluxes[fluxes < 12]
flux_4 = fluxes[fluxes < 13]
flux_5 = fluxes[fluxes < 14]
flux_6 = fluxes[fluxes < 15]
flux_7 = fluxes[fluxes < 16]
flux_8 = fluxes[fluxes < 17]
flux_9 = fluxes[fluxes < 18]
flux_10 = fluxes[fluxes < 19]
flux_11 = fluxes[fluxes < 20]
#%%
#number of sources with brightness higher then a limit
n_1 = len(flux_1)
n_2 = len(flux_2)
n_3 = len(flux_3)
n_4 = len(flux_4)
n_5 = len(flux_5)
n_6 = len(flux_6)
n_7 = len(flux_7)
n_8 = len(flux_8)
n_9 = len(flux_9)
n_10 = len(flux_10)
n_11 = len(flux_11)

N = np.array([n_1, n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9, n_10, n_11])

def gal_count_error(x, deg_squared):
    'determines error of counts'
    error = np.sqrt(x) / deg_squared
    return error

deg_squared = 0.06 #number of degrees squared that our image covers

N_new = N / deg_squared
N_error = gal_count_error(N, deg_squared)

def log_error(N_new, N_error):
    'determines error of y axis'
    error = (1 / np.log(10)) * (1 / N_new) * (N_error)
    return error

N_error_log = log_error(N_new, N_error)

x_axis = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_axis = np.log10(N_new)

plt.errorbar(x_axis, y_axis, yerr = N_error_log, fmt = 'x')

#%%

#fits a linear function on the plot
def function(x, m, c):
    'linear function to be fitted'
    n = (m * x) + c
    return n

x_fit = np.linspace(10, 20, 1000)

pop, pcov = curve_fit(function, x_axis, y_axis)

plt.errorbar(x_axis, y_axis, yerr = N_error_log, fmt = 'x')
plt.plot(x_fit, function(x_fit, pop[0], pop[1]))
print(pop[0], pcov[0][0], pop[1])
plt.xlabel('m')
plt.ylabel('N(m)')
plt.show()