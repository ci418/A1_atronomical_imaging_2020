# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:06:53 2020
@author: Charalambos Ioannou

Module producing a histogram with respect to brightness of the data
and analysing it in order to determine the mean background count.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#%%
hdulist = fits.open("mosaic.fits")
data = hdulist[0].data

#%%
data_1d = data.ravel()
#print(max(data_1d))
#Plots histogram of data within the range of the background count
bins = plt.hist(data_1d, bins = 75, range = (3350, 3500), \
                label = "number of bins = 100")
plt.title("Histogram of flux")
plt.xlabel("Flux")
plt.ylabel("Count")
plt.legend()
plt.show()

def Gaussian(x, m, sigma, A):
    "Defines Gaussian used to fit our histogram"
    return A * np.exp( - ((x - m) / sigma) ** 2)

counts = bins[0] #counts per bin
bin_edges = bins[1] #bin locations
x_points = []
for i in range(len(bin_edges) - 1):
    x = (bin_edges[i+1] + bin_edges[i])/2 #centre of bin
    x_points.append(x)

plt.plot(x_points, counts, 'o', label = 'Data') 

#Fits the data points with a gaussian
popt, pcov = curve_fit(Gaussian, x_points, counts, p0 = [3421, 20, 1200000])
print(popt[0], np.sqrt(pcov[0][0]), popt[1],  np.sqrt(pcov[1][1]))
plt.plot(x_points, Gaussian(x_points, *popt), label = 'Fit')
plt.title("Gaussian Fit on background noise")
plt.xlabel("Flux")
plt.ylabel("Count")
plt.legend()
plt.show()
