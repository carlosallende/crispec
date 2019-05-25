#!/usr/bin/python
'''
Data reduction package for CRISPEC

'''
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rim(file):
# read a r,g,b image into 3 numpy arrays

  im = Image.open(file)
  r,g,b = im.split()
  r = np.array(r)
  g = np.array(g)
  b = np.array(b)

  return r,g,b
  
def show(file):
# display an image in a file
  im = Image.open(file)
  im.show()

  return()

def merge(r,g,b):
# merges r,g,b channel images into one 2D array
# converting from byte to float on the fly

  rgb = np.array(r,dtype=float) + np.array(g,dtype=float) + np.array(b,dtype=float)

  return(rgb)

def collapse(file,rows,cols):
# collapses a spectrum (from 2D image to 1D image)
# the minimum signal is subtracted as background 
# rows (2-element tuple) range of rows to collapse
# cols (2-element tuple) range of columns to collapse
# backrows (2-element tuple) range of rows to determine background 
# (backcols is the same as cols) 


  r,g,b = rim(file)
  s = merge(r,g,b)
  s = s - np.min(s)
  s = s[rows[0]:rows[1]+1,cols[0]:cols[1]+1]
  x = np.sum(s,0) 

  return(x)

def findrows(file):

  r,g,b = rim(file)
  if (np.max(r) > 254): print('Warning: the r channel of this image (',file,') is saturated')
  if (np.max(g) > 254): print('Warning: the g channel of this image (',file,') is saturated')
  if (np.max(b) > 254): print('Warning: the b channel of this image (',file,') is saturated')
  s = merge(r,g,b)
  x = np.sum(s,1)
  plt.plot(x)
  row1 = np.min(np.where(x > 0.1*np.max(x)))
  row2 = np.max(np.where(x > 0.1*np.max(x)))
  plt.plot([row1,row1],[0,np.max(x)])
  plt.plot([row2,row2],[0,np.max(x)])
  plt.show()

  return (row1,row2)

def fitline(x,y):
# y = mx + c

  A = np.vstack([x, np.ones(len(x))]).T 
  m, c = np.linalg.lstsq(A, y, rcond=None)[0]
 
  return (m,c)

def lambdaref_led():
# load the reference (LED lamp) lambda spectrum (P4170230.JPG)
# 
# calibration based on 
# Hg lamp spectrum P4170229.JPG
# 3 most intense lines at
#  pixels 605.   1534.  1802.
#  lambda 435.8  546.1  578.0 
#  see, e.g., https://en.wikipedia.org/wiki/Mercury-vapor_lamp
#  reddest line is a blend, so I took an average
#  leading by least squares fitting to 
#  lambda= 0.11877957016041925 * pixel +  363.9299046659294
#  cal transferred to P4170230.JPG
#  both spectra extracted with rows=(1776,2503) cols=(1,3500)

  y = collapse('P4170230.JPG',(1776,2503),(1,3500))
  x = 0.11878 * np.arange(3500) +  363.9299
  
  return(x,y)

def lambdacal_led(file,rows,cols):
# read and 
# calibrate a lambda spectrum with the reference lambda spectrum
  y = collapse(file,rows,cols)
  x2,y2 = lambdaref_led()
  c = np.correlate(y2,y,mode='same')
  #plt.plot(c)
  #plt.show()
  shift = np.mean(np.where(c == np.max(c))) - len(y)/2.
  shift = shift * 0.11878 # from pixels to nm
  x = x2 + shift
  print('shift=',shift,' nm') 
  
  return (x,y)

def lambdaref():
# load the reference lambda spectrum (P4170195.JPG)
# 
# calibration based on 
# Hg lamp spectrum P4170193.JPG
# 3 most intense lines at
#  pixels 573.   1501.  1768.
#  lambda 435.8  546.1  578.0 
#  see, e.g., https://en.wikipedia.org/wiki/Mercury-vapor_lamp
#  reddest line is a blend, so I took an average
#  leading by least squares fitting to 
#  lambda= 0.11895993980931158 * pixel +  367.6186370842082
#  cal transferred to P4170195.JPG
#  both spectra extracted with rows=(1776,2503) cols=(1,3500)

  y = collapse('P4170195.JPG',(1776,2503),(1,3500))
  x = 0.11896 * np.arange(3500) +  367.6186
  
  return(x,y)

def lambdacal(file,rows,cols):
# read and 
# calibrate a lambda spectrum with the reference fluorescent lamp spectrum
  y = collapse(file,rows,cols)
  x2,y2 = lambdaref()
  c = np.correlate(y2,y,mode='same')
  #plt.plot(c)
  #plt.show()
  shift = np.mean(np.where(c == np.max(c))) - len(y)/2.
  shift = shift * 0.11896 # from pixels to nm
  x = x2 - shift
  print('shift=',shift* 0.11896,' nm') 
  
  return (x,y)


def fluxcal(file,rows,cols,x):
# read and flux calibrate with the reference response spectrum
# derived from the the ratio of the sky spectrum P4170125.JPG
# extracted with (1776,2503),(1,3500)) and wavelength calibrated 
# with the reference lambda spectrum, to the calspec solar spectrum 
# smoothed to have a fwhm resolution of 8 nm
# x is the wavelength array associated to file

  y = collapse(file,rows,cols)
  r = np.loadtxt('/home/callende/python/rn.dat',dtype=float)
  my = np.mean(y)
  y = y / np.interp(x,r[:,0],r[:,1])
  y = y / np.mean(y) * my

  return y

if __name__ == "__main__":

  allfiles=os.listdir('.')
  targetfiles=[]
  for name in allfiles:
    if name.endswith('.JPG'): targetfiles.append(name)
  f=open('calfiles.lis','r')
  calfiles=f.readlines()
  print('calibration files = ',calfiles)
  rows=findrows(calfiles[0].rstrip())
  print('rows=',rows)
  for file in calfiles:
    file=file.rstrip()
    print('calibrating ',file,' ...')
    x,y = lambdacal_led(file,rows,(1,3500))
    np.savetxt(file+'.x.dat',x,delimiter=" ",fmt='%s')
    np.savetxt(file+'.y.dat',y,delimiter=" ",fmt='%s')
  for file in targetfiles:
    file=file.rstrip()
    print('collapsing ',file,' ...')
    y = collapse(file,rows,(1,3500))
    np.savetxt(file+'.y.dat',y,delimiter=" ",fmt='%s')
