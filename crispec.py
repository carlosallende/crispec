#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Data reduction package for CRISPEC

This script and library handles automatically the data reduction for crispec data. It finds the region of the images where the spectra fall, uses spectra of a Hg lamp to identify the strongest emission lines and fit a linear curve to calibrate in wavelength, collapses and wavelength calibrates all the spectra, and combines the RGB images into a single spectrum after taking ratios to a reference spectrum (usually that of an LED lamp). It also attempts to get the spectral energy distribution by using a solar model and an observation of the sun made with crispec.

The script expects a bunch of *.JPG images (crispec spectra) in the working directory, and two .txt files:

calfiles.txt, which lists the names of the .JPG images with spectra from the Hg calibration lamp to be used for wavelength calibration

reffile.txt, which includes the name of a single .JPG image to be use as reference to take flux ratios.

The script produces flat files (.dat) with a header and a data table including:
wavelengths, the signal in the R, G, and B channels, a combine flux ratio relative to the reference spectrum and its uncertainty, and a calibrated spectral energy distribution based on a solar model.
It also produces plots with the resulting RGB spectra and the combined relative fluxes in PNG format, and a log file ('crislog').

Example 
-------

python3 crispec.py

One can avoid the interactive graphical window by typing

python3 crispec.py ni

"""
import os
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from scipy.signal import find_peaks_cwt 
from scipy.optimize import curve_fit

crisver = '1.0' #software version
date = time.strftime("%c")
crisdir = os.path.dirname(os.path.realpath(__file__))


def rim(file):

  """Read a r,g,b image into 3 numpy arrays

  Parameters
  ----------
  file: str
      file name for a JPG image containing a crispec spectrum

  Returns
  -------
  r: numpy array of integers
      R-band image
  g: numpy array of integers
      G-band image
  b: numpy array of integers
      B-band image
  exif: numpy array of strings
      image header 
  """

  im = Image.open(file)
  r,g,b = im.split()
  r = np.array(r)
  g = np.array(g)
  b = np.array(b)

  exif = { ExifTags.TAGS[k]: v for k, v in im._getexif().items() if k in ExifTags.TAGS }


  return r,g,b,exif

def timestamp(file):
  """Provides a single float that gives the time in hours from the image header, useful for sorting spectra and assigning Hg calibration spectra

  Parameters
  ----------
  file: str
      file name for a JPG image containing a crispec spectrum

  Returns
  -------
  hour: float
      hour at which the image was taken (from the camera)

  """
  

  r,g,b,exif = rim(file)
  cadena = exif['DateTime']
  date_time = cadena.split(' ')
  date = date_time[0].split(':')
  time = date_time[1].split(':')
  hour = float(time[0]) + float(time[1])/60. + float(time[2])/3600.

  return(hour) 
  
def imshow(file):
  """Displays an image

  Parameters
  ----------
  file: str
      file name for a JPG image containing a crispec spectrum

  Returns
  -------
  empty

  """

  im = Image.open(file)
  nx, ny = im.size
  im = im.resize( ( int(nx/2**3), int(ny/2**3) ) )
  im.show()

  return()

def listfiles(lista):
  """Collects files that match in one or more of a sequence of strings (linux wildcards allowed) into a list, avoiding repeats


  Parameters
  ----------
  lista: str
      string that identifies the files (linux wildcards ok)

  Returns
  -------
  files: list
       list of image file names

  """


  files = glob.glob(lista[0])
  for entry in lista[1:]:
    b = glob.glob(entry)
    for file in b: files.append(file)

  return(sorted(list(set(files))))

def errors(a):
#estimates photon noise for an observed spectrum by taking the square root of the input

  return ( np.sqrt(a) )

def errors2(a):
#estimates noise for an observed spectrum  from the scatter after smoothing

  step = 50
  k = np.ones(step)
  k = k / np.sum(k)
  b = np.convolve(a, k, 'same')
  er = np.zeros(len(a))
  for i in range(len(a)): 
    er[i] = np.std( b[np.max([0,int(i-step/2)]):min([ int(i+step/2),len(a)-1])] )

  return ( er )


def ratio(a,b,ea,eb):
#computes the ratio of two spectra, propagating errors linearly

  f = np.true_divide(a,b,where=(b != 0.))
  ef = np.true_divide( 1. , b, where=(b != 0.)) * np.sqrt( ea**2 + f**2 * eb**2) 

  return( f, ef)


def merge(r,g,b):
# adds r,g,b channel images (2d or collapsed) into one array
# converting from byte to float on the fly

  rgb = np.array(r,dtype=float) + np.array(g,dtype=float) + np.array(b,dtype=float)

  return(rgb)

def wmerge(r,g,b,er,eg,eb):
#merges r,g,b channel computing a weighted average spectrum (and error bar)

  ysum =  np.true_divide(r,er**2) + np.true_divide(g,eg**2) + np.true_divide(b,eb**2) 
  wsum =  np.true_divide(1.,er**2) + np.true_divide(1.,eg**2) + np.true_divide(1.,eb**2) 

  res = np.true_divide (ysum, wsum)
  eres = np.sqrt(np.true_divide(1.,wsum))

  return (res, eres )


def collapse(im,rows,cols):
# collapses a spectrum (from 2D image to 1D image)
# the minimum signal is subtracted as background 
# rows (2-element tuple) range of rows to collapse
# cols (2-element tuple) range of columns to collapse
# backrows (2-element tuple) range of rows to determine background 
# (backcols is the same as cols) 


  im = im - np.min(im)
  im = im[rows[0]:rows[1]+1,cols[0]:cols[1]+1]
  y = np.sum(im,0) 

  return(y)

def file_collapse(file,rows,cols):
# collapses all 3 passbands in a spectrum file

  r,g,b, exif = rim(file)
  rr = collapse(r,rows,cols)
  gg = collapse(g,rows,cols)
  bb = collapse(b,rows,cols)

  return(rr, gg, bb)


def findrows(file):

  r,g,b,exif = rim(file)
  if (np.max(r) > 254): print('Warning: the r channel of this image (',file,') is saturated')
  if (np.max(g) > 254): print('Warning: the g channel of this image (',file,') is saturated')
  if (np.max(b) > 254): print('Warning: the b channel of this image (',file,') is saturated')
  s = merge(r,g,b)
  x = np.sum(s,1)
  plt.plot(x)
  minx=np.min(x)
  maxx=np.max(x)
  row1 = np.min(np.where(x > minx+0.1*(maxx-minx)))
  row2 = np.max(np.where(x > minx+0.1*(maxx-minx)))
  plt.plot([row1,row1],[0,np.max(x)])
  plt.plot([row2,row2],[0,np.max(x)])
  plt.xlabel('row')
  plt.ylabel('coadded flux in all columns 1-3500')
  plt.title('selection of rows to extract spectrum')
  plt.show()

  return (row1,row2)


def getpeaklocations(y):
#Returns indices for the 3 strongest peaks in the array y

  peaks = find_peaks_cwt(y,widths=[40]) 
  index3 = np.argsort(y[peaks])[-3:]
  return (sorted(peaks[index3]))

def getpeaklocations2(y):
#Returns indices for the 3 strongest peaks in the array y

  peaks = find_peaks_cwt(y,widths=[2,4,10,20]) 
  index3 = np.argsort(y[peaks])[-3:]
  return (sorted(peaks[index3]))

def fitline(x,y):
# y = mx + c

  A = np.vstack([x, np.ones(len(x))]).T 
  m, c = np.linalg.lstsq(A, y, rcond=None)[0]
 
  return (m,c)


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))



def lambdacal(file,rows,cols):
# read and 
# calibrate a lambda spectrum with the reference lambda spectrum
  r,g,b, exif = rim(file)
  rr = collapse(r,rows,cols)
  gg = collapse(g,rows,cols)
  bb = collapse(b,rows,cols)
  y = merge(rr,gg,bb)
  k = np.ones(10)
  y2 = np.convolve(y, k/np.sum(k), 'same')
  p = getpeaklocations(y2)
  plt.plot(y)
  plt.xlabel('pixel')
  plt.ylabel('flux')
  plt.title('identification of the 3 strongest Hg lines')
  chunk = int(100) # expected half-width of lines
  sigma = chunk/3. # expected sigma of lines
  ss = []
  for entry in p: 
    plt.plot([entry,entry],[0,max(y)])
    p0 = [1., chunk/2., sigma]
    coeff, var_matrix = curve_fit(gauss, np.arange(2*chunk+1), y[entry-chunk:entry+chunk+1], p0=p0)
    # Get the fitted curve
    ss.append(np.abs(coeff[2]))
    fit = gauss(np.arange(2*chunk+1), *coeff)
    plt.plot(float(entry)-chunk+np.arange(2*chunk+1),fit)

  ss = np.array(ss) 
  plt.annotate('Resolution = '+str(np.mean(ss))+'   +/- '+ str(np.std(ss)) + ' pixels', (np.mean(p)/4.,coeff[0]))
  plt.show()

  w=[435.8,546.1,578.0]
  c = fitline( p, w )
  x = c[0]* np.arange(3500) + c[1]
  print ('dispersion, zero point = ',c,'  nm/pixel, nm')  
  ss = ss * c[0]
  print ('FWHM spectral resolution = ',str(np.mean(ss)),'   +/- ', str(np.std(ss)), ' nm')
  fwhm = np.mean(ss)
  efwhm = np.std(ss)

  plt.plot(p,w,'*',np.arange(3500),x)
  plt.xlabel('pixel')
  plt.ylabel('wavelength')
  plt.title('wavelength calibration: linear fit')
  plt.annotate('wavelength = '+str(c[0])+'* pixel + '+str(c[1]), (np.mean(p)/4.,np.mean(w)))
  plt.show()

  return (x,y, fwhm, efwhm)


def fluxcal(x,y):
#multiples by the inverse of the response, define to give the solar SED (normalized to an average of 1)
#from the ratio of the solar spectrum to that of the yellow/black LED lamp

  xres, yres = getsolar()
  y2 = y * np.interp(x,xres,yres)

  return y2

def getsolar():

  d = np.loadtxt(os.path.join(crisdir,'solresponse.dat'))
  x = d[:,0]
  y = d[:,1]
  
  return(x,y)

def flux_ratio(rr,gg,bb,rref,gref,bref,snr=50.):
#takes the ratio of two collapsed spectra in the three bands for two sources and combines when 
#by producing a weighted average of the data with a minimum snr

    err = errors(rr)
    egg = errors(gg)
    ebb = errors(bb)
    
    erref = errors(rref)
    egref = errors(gref)
    ebref = errors(bref)

    rr2, err2 = ratio(rr,rref,err,erref)
    gg2, egg2 = ratio(gg,gref,egg,egref)
    bb2, ebb2 = ratio(bb,bref,ebb,ebref)

    rmask = (rr > snr**2) & (rref > snr**2)
    gmask = (gg > snr**2) & (gref > snr**2)
    bmask = (bb > snr**2) & (bref > snr**2)

    rr3 = np.zeros(len(rr))
    gg3 = np.zeros(len(rr))
    bb3 = np.zeros(len(rr))

    rr3[rmask] = rr2[rmask]
    gg3[gmask] = gg2[gmask]
    bb3[bmask] = bb2[bmask]
    
    err3 = 100.*np.ones(len(rr))
    egg3 = 100.*np.ones(len(rr))
    ebb3 = 100.*np.ones(len(rr))

    err3[rmask] = err2[rmask]
    egg3[gmask] = egg2[gmask]
    ebb3[bmask] = ebb2[bmask]

    y, ey = wmerge(rr3,gg3,bb3,err3,egg3,ebb3)

    return (y, ey)

def rdat(datfile):
  """Reads a reduced spectrum

  Parameters
  ----------
  datfile: str
      file name for the input .dat file     

  Returns
  -------
    header: array of str
      image header and spectrum info
    x: numpy array of floats
      wavelengths (nm)
    r: numpy array of floats
      R-band spectrum (collapsed, raw)
    g: numpy array of floats 
      G-band spectrum (collapsed, raw)
    b: numpy array of floats 
      B-band spectrum (collapsed, raw)
    fratio: numpy array of floats 
      combined spectrum, ratio of input data to reference spectrum
    fratio_err: numpy array of floats
      uncertainty in the combined spectrum
    flux: numpy array of floats
      spectral energy distribution (calibrated based on the sun)

  """

  header  = []
  f = open(datfile,'r')
  line = f.readline()
  i = 0 
  while (line.rstrip() != 'Data'):
    print(line)
    header.append(line.rstrip())
    line = f.readline()
    i = i + 1
  f.close()
  data = np.loadtxt(datfile,skiprows=i+1)
  x = data[:,0]
  r = data[:,1]
  g = data[:,2]
  b = data[:,3]
  fratio = data[:,4]
  fratio_err = data[:,5]
  flux = data[:,6]
  
  return (header, x, r, g, b, fratio, fratio_err, flux)

if __name__ == "__main__":

  npar = len(sys.argv)

  assert (npar <= 2), 'Synple requires at maximum 1 input parameter'
  argumento = ""
  if npar > 1: argumento = sys.argv[1]
  if argument == 'ni': plt.ion()

  flog = open('crislog','w')
  flog.write("starting data reduction ...\n")
  flog.write(time.strftime("%c")+"\n")


  allfiles = os.listdir('.')
  targetfiles = []
  for name in allfiles:
    if name.endswith('.JPG'): targetfiles.append(name)
  targetfiles = sorted(list(set(targetfiles)))

  #read list of cal files and the name of the reference file
  flog.write("reading calfiles.txt ...\n")
  f = open('calfiles.txt','r')
  calfiles = f.readlines()
  f.close()
  flog.write("reading reffile.txt ...\n")
  f = open('reffile.txt','r')
  reffile = f.readlines()
  flog.write("reffile = "+ reffile[0] +" \n")
  assert (len(reffile) ==1 ), 'reffile.txt must have a single file name' 
  f.close()


  print('calibration files = ',calfiles)
  flog.write("calfiles = \n")
  flog.write(" ".join(calfiles))
  

  #determine spectra footprint on detector
  rows = findrows(calfiles[0].rstrip())
  cols = (1,3500)
  flog.write("determine rows to collapse from first calibration file ...\n")
  print('rows=',rows)
  flog.write("cols ="+" ".join(str(cols))+"\n")
  flog.write("rows ="+" ".join(str(rows))+"\n")


  #collapse reference spectrum
  rref, gref, bref = file_collapse(reffile[0].rstrip(), rows, cols)

  caltimes=[] #hour corresponding to each calibration
  calfwhm=[]  #FWHM resolution for each calibration
  calefwhm=[] #eFWHM resolution for each calibration
  for file in calfiles:
    file=file.rstrip()
    print('calibrating ',file,' ...')
    flog.write("calibrating file ..."+file+"\n")
    x,y, fwhm, efwhm = lambdacal(file,rows,(1,3500))
    x.tofile(file+'.x.dat',sep=" ",format='%s')
    caltimes.append(timestamp(file))
    calfwhm.append(fwhm)
    calefwhm.append(efwhm)
    flog.write("FWHM resolution is "+str(fwhm)+" +/- "+str(efwhm)+" nm \n")
  for file in targetfiles:
    file=file.rstrip()
    print('collapsing ',file,' ...')
    flog.write("collapsing file ..."+file+"\n")

    #wavelength calibration
    time = timestamp(file)
    diff = np.abs( time - np.array(caltimes) )
    wcal = np.where(diff == np.min(diff))
    chosencal = calfiles[int(wcal[0])][:-1]
    print('assigning calibration file ',chosencal,'...')
    flog.write("assigning calibration file ..."+chosencal+"\n")
    x = np.loadtxt(chosencal+'.x.dat')
    fwhm = calfwhm[int(wcal[0])]    
    efwhm = calefwhm[int(wcal[0])]

    #collapse spectrum
    rr,gg,bb = file_collapse(file, rows, cols)

    #take the ratio and combine 3 bandpasses
    y, ey = flux_ratio(rr,gg,bb,rref,gref,bref)

    #calibrate using the solar spectrum
    y2 = fluxcal(x,y)

    #write output file
    fspec = open(file+'.dat','w')
    r, g, b, header = rim(file)
    fspec.write("Instrument"+"="+'Crispec'+"\n")
    fspec.write("DataReduction"+"="+"Version "+crisver+"\n")
    fspec.write("DataReduction"+"="+"DateTime "+date+"\n")   
    fspec.write("DataReduction"+"="+"FWHM Resolution "+str(fwhm)+" +/- "+str(efwhm)+" nm  \n")   
    for entry in list(header.keys()): 
      if (type(header[entry]) == str): fspec.write(str(entry)+"="+header[entry]+"\n")
    fspec.write("Image"+"="+file+"\n")
    fspec.write("Reference"+"="+reffile[0].rstrip()+"\n")
    index = 0
    for entry in ('Wavelength nm','R','G','B','RATIO (Image / Reference)','ERROR (Image / Reference)','Flux (SED calibrated to solar/LED ratio)'): 
      fspec.write("BAND"+str(index)+"="+str(entry)+"\n")
      index = index + 1
    fspec.write("Data\n")
    for i in range(len(x)):
      #fspec.write(str(x[i])+" "+str(rr[i])+" "+str(gg[i])+" "+str(bb[i])+" "+str(y[i])+" "+str(ey[i])+"\n")      
      fspec.write('{wave:10.4f} {rflux:10d} {gflux:10d} {bflux:10d} {yflux:14.4e} {eyflux:14.4e} {yflux2:14.4e} \n'.format(wave=x[i], rflux=rr[i], gflux=gg[i], bflux=bb[i], yflux=y[i], eyflux=ey[i], yflux2=y2[i]))

    fspec.close()


    plt.clf()
    plt.plot(x,bb,x,rr,x,gg)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('flux')
    plt.title(file)
    #plt.plot(x,y)
    plt.show()
    plt.savefig(file+'.raw.PNG')

    plt.clf()
    plt.plot(x,y)
    #plt.ylim([0.0,np.max(y)*1.1])
    plt.xlabel('wavelength (nm)')
    plt.ylabel('flux')
    plt.title(file + ' Reference = '+reffile[0].rstrip())
    #plt.plot(x,y)
    plt.show()
    plt.savefig(file+'.ratio.PNG')

  flog.close()
