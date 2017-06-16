# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import numpy as np
np.seterr(all='ignore')
import os
import glob
import pathlib
import re
import numbers
try:
  import progressbar as pb
  _has_progress_bar = True
except ImportError:
  _has_progress_bar = False
  log.warn("Reccomended package: progressbar is missing")
from datastorage import DataStorage

try:
  import matplotlib.pyplot as plt
except ImportError:
  log.warn("Can't import matplotlib !")

_time_regex =      re.compile( "(-?\d+\.?\d*(?:ps|ns|us|ms)?)")
_timeInStr_regex = re.compile("_(-?\d+\.?\d*(?:ps|ns|us|ms)?)")

def getFiles(folder,basename="*.edf*",nFiles=None):
  files = glob.glob(os.path.join(folder,basename))
  files.sort()
  if nFiles is not None: files = files[:nFiles]
  return files

def getEdfFiles(folder,**kw):
  return getFiles(folder,basename="*.edf*",**kw)

def getDelayFromString(string) :
  match = _timeInStr_regex.search(string)
  return match and match.group(1) or None

_time_regex = re.compile("(-?\d+\.?\d*)((?:s|fs|ms|ns|ps|us)?)")

def strToTime(delay) :
  if isinstance(delay,bytes): delay = delay.decode('ascii')
  _time2value = dict( fs = 1e-15, ps = 1e-12, ns = 1e-9, us = 1e-6, ms = 1e-3, s = 1)

  match = _time_regex.search(delay)
  if match:
    n,t = float(match.group(1)),match.group(2)
    value = _time2value.get(t,1)
    return n*value
  else:
    return None

def timeToStr(delay,fmt="%+.0f"):
  a_delay = abs(delay)
  if a_delay >= 1:
    ret = fmt % delay + "s"
  elif 1e-3 <= a_delay < 1: 
    ret = fmt % (delay*1e3) + "ms"
  elif 1e-6 <= a_delay < 1e-3: 
    ret = fmt % (delay*1e6) + "us"
  elif 1e-9 <= a_delay < 1e-6: 
    ret = fmt % (delay*1e9) + "ns"
  elif 1e-12 <= a_delay < 1e-9: 
    ret = fmt % (delay*1e12) + "ps"
  elif 1e-15 <= a_delay < 1e-12: 
    ret = fmt % (delay*1e12) + "fs"
  elif 1e-18 <= a_delay < 1e-15: 
    ret = fmt % (delay*1e12) + "as"
  else:
    ret = str(delay) +"s"
  return ret

def removeExt(fname):
  """ special remove extension meant to work with compressed files.edf and .edf.gz files """
  if fname[-3:] == ".gz": fname = fname[:-3]
  return os.path.splitext(fname)[0]

def getBasename(fname):
  return os.path.basename(removeExt(fname));

def progressBar(N,title="Percentage"):
  if _has_progress_bar:
    widgets = [title, pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=N)
  else:
    class FakeBar(object):
      def __init__(self): pass
      def start(self):    pass
      def update(self,i): pass
      def finish(self):   pass
    pbar = FakeBar()
  pbar.start()
  return pbar


def findSlice(array,lims):
  start = np.ravel(np.argwhere(array>lims[0]))[0]
  stop  = np.ravel(np.argwhere(array<lims[1]))[-1]
  return slice(int(start),int(stop))

def removeBackground(x,data,xlims=None,max_iter=100,background_regions=[],**kw):
  from dualtree import dualtree
  if data.ndim == 1: data = data[np.newaxis,:]
  if xlims is not None:
    idx = findSlice(x,xlims)
    x = x[idx]
    data = data[:,idx].copy()
  else:
    data = data.copy(); # create local copy
  # has to be a list of lists ..
  if background_regions != [] and isinstance(background_regions[0],numbers.Real):
    background_regions = [background_regions,]
  background_regions = [findSlice(x,brange) for brange in background_regions]
  for i in range(len(data)):
    data[i] = data[i] - dualtree.baseline(data[i],max_iter=max_iter,
                        background_regions=background_regions,**kw)
  return x,np.squeeze(data)
  

def plotdata(data,x=None,plotAverage=True,showTrend=True,title=None,\
      clim='auto',fig=None):
    """ Plot 2D array to look for trends/stability or outlier

        It usually works best with normalized data

        Parameters
        ----------
        data: dict-like object or 2D array.
          if dict-like it has to have keys
          'q' used for one axis
          'data' for 2D array[numimg,nQ]
        x: array or None
          if given it has to have a length of numimg
          if None will be 0,1,2,3,...numimg and interpreted
          as image number
        plotAverage: bool
          if True, plot average
        showTrend: bool
          if True, show 2D plot (most common use)
        clim: 'auto' or tuple
          color scale to use, is 'auto' 1.5%,98.5% percentiles are used
        fig:
          matplotlib fig instance to use, if None creates a new one
        title: str
          title to use for plot (if None and data['folder'] exists
          the value is used as title
          
    """
    if isinstance(data,np.ndarray):
      q = np.arange( data.shape[-1] )
    else:
      if title is None and 'folder' in data: title = data["folder"]
      q = data["q"]
      data = data["data"]
    if not (plotAverage or showTrend): return

    if x is None: x = np.arange(data.shape[0])
    if clim == 'auto': clim = np.nanpercentile(data,(1.5,98.5))
    one_plot = showTrend or plotAverage
    two_plot = showTrend and plotAverage

    if one_plot and not two_plot:
      if fig is None:
        fig,ax = plt.subplots(1,1)
      else:
        fig.clear()
        ax = fig.axes
    if two_plot:
      if fig is None:
        fig,ax = plt.subplots(2,1,sharex=True)
      else:
        ax = fig.axes
 
    ax = np.atleast_1d(ax)
    if showTrend:
      plt.sca(ax[1])
      plt.pcolormesh(q,x,data)
      plt.ylabel("image number, 0 being older")
      plt.xlabel(r"q ($\AA^{-1}$)")
      plt.clim( *clim )
    if plotAverage:
        ax[0].plot(q,np.nanmean(data,axis=0))
    if (plotAverage or showTrend) and title is not None:
      plt.title(title)

def volumeFraction(concentration=1,molWeight=17,density=1.347):
  """ molWeight is kDa 
      concentration in mM
      density g/ml
  """
  concentration_mg_ml = concentration*molWeight
  volume_fraction    = concentration_mg_ml/density/1e3
  return volume_fraction

def molecularMass_from_I0(I0,c,density=1.347):
    """ calculate the molecular mass in Da

        based upon:
        D Orthaber, A Bergmann and O Glatter
        J. Appl. Cryst. (2000). 33, 218-225


        Paramters
        ---------
        I0 : float
            extrapolated value (in cm-1)
        c : float
            concentration (in g/ml)
        density : float
            protein density (in g/ml)
    """
    delta_rho = 2.67e10; # [cm-2]
    delta_rho_M = delta_rho/density
    Navo = 6e23
    M = I0*Navo / (c*delta_rho_M**2)
    return M

def plotdiffs(data,select=None,err=None,absSignal=None,absSignalScale=10,
           showErr=False,cmap=plt.cm.jet,fig=None,title=None,plotDiffRef=False):
    """ Plot difference data

        Parameters
        ----------
        data: dict-like object or (q,diffs,scan)tuple
          if dict-like it has to have keys
          'q' used for one axis
          'diffs' for 2D array[numscan,nQ]
          'scan'  scan points
          if present 'err' is used for errorbars
          if present 'diffs_plus_ref' is used
        select: slice
          to select time delays (for example select=slice(None,None,2))
        absSignal: array[nQ]
          if present, ass this data to the plot (could be the average)
        absSignalScale: float
          the factor to use to divide the absSignal for, it is kept as
          different parameters because the scaling it is added to the legend
        showErr: bool
          if True, plot errorbars as well
        cmap: matplotlib color map
          color map to use
        fig:
          matplotlib fig instance to use, if None creates a new one
        title: str
          title to use for plot (if None, no title is added)
        plotDiffRef:
          plot not only the diff but also the diffs_plus_ref (if present in data)
    """
    if isinstance(data,(list,tuple)):
        q,diffs,scan = args
        diffs_abs = None
    else:
        q = data["q"]
        diffs = data["diffs"]
        scan  = data["scan"]
        err   = data.get("err",None)
        diffs_abs = data.get("diffs_plus_ref",None)

    # this selection trick done in this way allows to keep the same colors when 
    # subselecting (because I do not change the size of diffs)
    if select is not None:
        indices = range(*select.indices(scan.shape[0]))
    else:
        indices = range(len(scan))
 
    if fig is None: fig = plt.figure()
 
    lines_diff = []
    lines_abs = []
    if absSignal is not None:
        line = plt.plot(q,absSignal/absSignalScale,lw=3,
                      color='k',label="absSignal/%s"%str(absSignalScale))[0]
        lines.append(line)
    for linenum,idiff in enumerate(indices):
        color = cmap(idiff/(len(diffs)-1))
        label = timeToStr(scan[idiff])
        kw = dict( color = color, label = label )
        if err is not None and showErr:
            line = plt.errorbar(q,diffs[idiff],err[idiff],**kw)[0]
            lines_diff.append(line)
        else:
            line = plt.plot(q,diffs[idiff],**kw)[0]
            lines_diff.append(line)
            if diffs_abs is not None and plotDiffRef:
                line = plt.plot(q,diffs_abs[idiff],color=color)[0]
                lines_abs.append(line)
    if title is not None: fig.axes[0].set_title(title)
    legend = plt.legend(loc=4)
    plt.grid()
    plt.xlabel(r"q ($\AA^{-1}$)")
    # we will set up a dict mapping legend line to orig line, and enable
    # picking on the legend line
    lined = dict()
    for legline, origline in zip(legend.get_lines(), lines_diff):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline
  
    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig    = plt.gcf() 
        fig.canvas.draw()
 
    fig.canvas.mpl_connect('pick_event', onpick)
    return lines_diff,lines_abs

def updateLines(lines,data):
    for l,d in zip(lines,data):
        l.set_ydata(d)



def saveTxt(fname,q,data,headerv=None,info=None,overwrite=True,columns=''):
  """ Write data to file 'fname' in text format.
      Inputs:
        q = x vector
        data = one or 2D array (first axis is q)
        info = dictionary (saved as '# key : value') or string
        headerv = vector to be used as header or string
  """
  if os.path.isfile(fname) and not overwrite:
    log.warn("File %s exists, returning",fname)
    return
  if isinstance(info,dict):
    header = [ "# %s : %s" %(k,str(v)) for (k,v) in info.items() ]
    header = "\n".join(header); # skip first #, will be added by np
  elif isinstance(info,str):
    header = info
  else:
    header = ""
  if isinstance(headerv,str): header += "\n%s" % headerv
  if data.ndim == 1:
    x = np.vstack( (q,data) )
  elif data.ndim == 2:
    x = np.vstack( (q,data) )
    if headerv is not None:
      headerv = np.concatenate(( (data.shape[1],),headerv))
      x = np.hstack( (headerv[:,np.newaxis],x) )
  if columns != '':
    s = "#" + " ".join( [str(c).center(12) for c in columns] )
    header = header + "\n" + s if header != '' else s
  np.savetxt(fname,x.T,fmt="%+10.5e",header=header,comments='')

def reshapeToBroadcast(what,ref):
  """ expand the 1d array 'what' to allow broadbasting to match 
      multidimentional array 'ref'. The two arrays have to same the same 
      dimensions along the first axis
  """
  if what.shape == ref.shape: return what
  assert what.shape[0] == ref.shape[0],\
    "automatic reshaping requires same first dimention"
  shape  = [ref.shape[0],] + [1,]*(ref.ndim-1)
  return what.reshape(shape)

def radToQ(theta,**kw):
  """ theta is the scattering angle (theta_out - theta_in);
      kw should be have either E or wavelength
      it returns the scattering vector in the units of wavelength """
  # Energy or wavelength should be in kw
  assert "E" in kw or "wavelength" in kw,\
    "need wavelength or E to convert rad to Q"
  # but not both
  assert not ("E" in kw and "wavelength" in kw),\
    "conflicting arguments (E and wavelength)"
  if "E" in kw: kw["wavelength"] = 12.398/kw["E"]
  return 4*np.pi/kw["wavelength"]*np.sin(theta/2)

def degToQ(theta,**kw):
  theta = theta/180.*np.pi
  return radToQ(theta,**kw)
degToQ.__doc__ = radToQ.__doc__

def qToTwoTheta(q,asDeg=False,**kw):
  """ Return scattering angle from q (given E or wavelength) """
  # Energy or wavelength should be in kw
  assert "E" in kw or "wavelength" in kw,\
    "need wavelength or E to convert rad to Q"
  # but not both
  assert not ("E" in kw and "wavelength" in kw),\
    "conflicting arguments (E and wavelength)"
  if "E" in kw: kw["wavelength"] = 12.398/kw["E"]
  theta = 2*np.arcsin(q*kw["wavelength"]/4/np.pi)
  if asDeg: theta = np.rad2deg(theta)
  return theta

def attenuation_length(compound, density=None, natural_density=None,energy=None, wavelength=None):
  """ extend periodictable.xsf capabilities """
  import periodictable.xsf
  if energy is not None: wavelength = periodictable.xsf.xray_wavelength(energy)
  assert wavelength is not None, "scattering calculation needs energy or wavelength"
  if (np.isscalar(wavelength)): wavelength=np.array( [wavelength] )
  n = periodictable.xsf.index_of_refraction(compound=compound,
                           density=density, natural_density=natural_density,
                           wavelength=wavelength)
  attenuation_length = (wavelength*1e-10)/ (4*np.pi*np.imag(n))
  return np.abs(attenuation_length)

def transmission(material='Si',thickness=100e-6, density=None, natural_density=None,energy=None, wavelength=None):
  """ extend periodictable.xsf capabilities """
  att_len = attenuation_length(material,density=density,
            natural_density=natural_density,energy=energy,wavelength=wavelength)
  return np.exp(-thickness/att_len)


def _phosphorAbsorption(twotheta,mu=1770,thickness=40e-6,E=None):
  """ return 1-np.exp(-mu*thick/np.cos(np.deg2rad(theta)))
      - mu is the neperian absortption linear coefficient (m-1)
      - twotheta is the scattering angle (in degrees)
  """
  if mu =='auto':
    assert E is not None, "phosphorAbsorption with automatic mu requires E"
    mu = 1/attenuation_length("Ce",energy=E,density=0.475)
  return 1-np.exp(-mu*thickness/np.cos(np.deg2rad(twotheta)))

def phosphorCorrection(twotheta,mu=1770,thickness=40e-6,E=None,normalizeToZeroAngle=False):
  """ helper function to correct for angle dependent absorption of the phosphor screen for an
      x-ray detector.
      return the correction factor one has to *multiply* the data for.
      - mu is the neperian absortption linear coefficient (m-1) [could be 'auto' if E is given]
      - twotheta is the scattering angle (in degrees)
  """
  corr = 1/_phosphorAbsorption(twotheta,mu=mu,thickness=thickness,E=E)
  if normalizeToZeroAngle: corr = corr*_phosphorAbsorption(0,mu=mu,thickness=thickness,E=E)
  return corr




def chargeToPhoton(chargeOrCurrent,material="Si",thickness=100e-6,energy=10,e_hole_pair=3.63):
  """
    Function to convert charge (or current to number of photons (or number 
    of photons per second)
    
    Parameters
    ----------
    
    chargeOrCurrent: float or array
    material : str
       Used to calculate 
     
  """
  # calculate absortption
  A = 1-transmission(material=material,energy=energy,thickness=thickness)
  chargeOrCurrent = chargeOrCurrent/A

  e_hole_pair_energy_keV = e_hole_pair*1e-3
  n_charge_per_photon = energy/e_hole_pair_energy_keV
  # convert to Q
  charge_per_photon = n_charge_per_photon*1.60217662e-19
  nphoton = chargeOrCurrent/charge_per_photon
 
  if len(nphoton) == 1: nphoton = float(nphoton)
  return nphoton

def approx(values,approx_values):
    """ returns array where every value is replaced by the closest in approx_values

        This funciton is useful for rebinning

        Example:
        -------
        approx( np.arange(0,1,0.1), [0,0.3,0.7] )
           array([ 0. ,  0. ,  0.3,  0.3,  0.3,  0.7,  0.7,  0.7,  0.7,  0.7])

    """
    # make sure they are arrays
    values = np.asarray(values)
    approx_values = np.asarray(approx_values)
    # create outter difference
    diff = np.abs(values[:,np.newaxis] - approx_values)
    args = np.argmin(diff,axis=1)
    values = approx_values[args]
    #values = np.asarray( [ approx_values[np.argmin(np.abs(v-approx_values))] for v in values] )
    return values
  
def logToScreen():
  """ It allows printing to terminal on top of logfile """
  # define a Handler which writes INFO messages or higher to the sys.stderr
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  # set a format which is simpler for console use
  formatter = logging.Formatter('%(message)s')
  # tell the handler to use this format
  console.setFormatter(formatter)
  # add the handler to the root logger (if needed)
  if len(logging.getLogger('').handlers)==1:
    logging.getLogger('').addHandler(console)


