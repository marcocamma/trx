# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import
import logging
log = logging.getLogger(__name__)


import numpy as np
np.seterr(all='ignore')
import inspect
import os
import collections
import glob
import pathlib
from datastorage import DataStorage
from . import utils
from . import filters
from .mask import interpretMasks
import re
import fabio
import pyFAI


g_default_extension='.h5'
try:
  import matplotlib.pyplot as plt
except ImportError:
  log.warn("Can't import matplotlib !")

def _read(fname):
  """ read data from file using fabio """
  f = fabio.open(fname)
  data = f.data
  del f; # close file
  return data

def read(fnames):
  """ read data from file(s) using fabio """
  if os.path.isfile(fnames):
    data = _read(fnames)
  else:
    # read one image to know img size
    temp = _read(fnames[0])
    shape = [len(fnames),]+list(temp.shape)
    data  = np.empty(shape)
    data[0] = temp
    for i in range(1,len(fnames)): data[i] = _read(fnames[i])
  return data
    
def ai_as_dict(ai):
  """ ai is a pyFAI azimuthal intagrator"""
  methods = dir(ai)
  methods = [m for m in methods if m.find("get_") == 0]
  names   = [m[4:] for m in methods]
  values  = [getattr(ai,m)() for m in methods]
  ret = dict( zip(names,values) )
  ret["detector"] = ai.detector.get_name()
  return ret

def ai_as_str(ai):
  """ ai is a pyFAI azimuthal intagrator"""
  s=[ "# Detector        : %s" % ai.detector.name,
      "# Pixel      [um] : %.2fx%.2f" % (ai.pixel1*1e6,ai.pixel2*1e6),
      "# Distance   [mm] : %.3f" % (ai.dist*1e3),
      "# Center     [mm] : %.3f,%.3f" % (ai.poni1*1e3,ai.poni2*1e3),
      "# Center     [px] : %.3f,%.3f" % (ai.poni1/ai.pixel1,ai.poni2/ai.pixel2),
      "# Wavelength [A]  : %.5f" % (ai.wavelength*1e10),
      "# rot[1,2,3] [rad]: %.3f,%.3f,%.3f" % (ai.rot1,ai.rot2,ai.rot3) ]
  return "\n".join(s)

def dodezinger(ai, imgs, mask = None, npt_radial = 600, method = 'csr',dezinger=50):
  """ ai is a pyFAI azimuthal intagrator 
              it can be defined with pyFAI.load(ponifile)
        dezinger: None or float (used as percentile of ai.separate)
        mask: True are points to be masked out """
  if dezinger is None or dezinger <= 0: return imgs
  if imgs.ndim == 2: imgs=imgs[np.newaxis,:]
  for iimg,img in enumerate(imgs):
    _,imgs[iimg]=ai.separate(imgs[iimg],npt_rad=npt_radial,npt_azim=512,
                 unit='q_A^-1',method=method,mask=mask,percentile=dezinger)
  return np.squeeze(imgs)

def do1d(ai, imgs, mask = None, npt_radial = 600, method = 'csr',safe=True,dark=10., polCorr = 1,dezinger=None):
    """ ai is a pyFAI azimuthal intagrator 
              it can be defined with pyFAI.load(ponifile)
        dezinger: None or float (used as percentile of ai.separate)
        mask: True are points to be masked out """
    if isinstance(ai,dict): ai = getAI(ai)
    # force float to be sure of type casting for img
    if isinstance(dark,int): dark = float(dark);
    if imgs.ndim == 2: imgs = (imgs,)
    if mask is not None and not isinstance(mask,np.ndarray):
        mask = interpretMasks(mask,imgs[0].shape)
    out_i = np.empty( ( len(imgs), npt_radial) )
    out_s = np.empty( ( len(imgs), npt_radial) )
    for _i,img in enumerate(imgs):
      if dezinger is not None and dezinger > 0:
        img=dodezinger(ai,img,npt_radial=npt_radial,mask=mask,
                     dezinger=dezinger,method=method)
      q,i, sig = ai.integrate1d(img-dark, npt_radial, mask= mask, safe = safe,\
                 unit="q_A^-1", method = method, error_model = "poisson",
                 polarization_factor = polCorr)
      out_i[_i] = i
      out_s[_i] = sig
    return q,np.squeeze(out_i),np.squeeze(out_s)

def do2d(ai, imgs, mask = None, npt_radial = 600, npt_azim=360,method = 'csr',safe=True,dark=10., polCorr = 1):
    """ ai is a pyFAI azimuthal intagrator 
              it can be defined with pyFAI.load(ponifile)
        mask: True are points to be masked out """
    # force float to be sure of type casting for img
    if isinstance(dark,int): dark = float(dark);
    if imgs.ndim == 2: imgs = (imgs,)
    if mask is not None and not isinstance(mask,np.ndarray):
        mask = interpretMasks(mask,imgs[0].shape)
    out = np.empty( ( len(imgs), npt_azim,npt_radial) )
    for _i,img in enumerate(imgs):
      i2d,q,azTheta = ai.integrate2d(img-dark, npt_radial, npt_azim=npt_azim,
                      mask= mask, safe = safe,unit="q_A^-1", method = method,
                      polarization_factor = polCorr )
      out[_i] = i2d
    return q,azTheta,np.squeeze(out)

def getAI(poni=None,folder=None,**kwargs):
  """ get AzimuthalIntegrator instance:
      → if poni is a string, it is used as filename to read.
        in this case if folder is given it is used (together with all its 
        subfolder) as search path (along with ./ and home folder)
      → kwargs if present can be used to define (or override) parameters from files,
        dist,xcen,ycen,poni1,poni2,rot1,rot2,rot3,pixel,pixel1,pixel2,
        splineFile,detector,wavelength
  """
  if isinstance(poni,dict):
    kwargs = poni
    _def_names = ["poni1","poni2","pixel1","pixel2","dist","wavelength",\
                "rot1","rot2","rot3","xcen","ycen","pixel"]
    kwargs = dict( (name,value) for (name,value) in poni.items() \
             if name in _def_names )
  if isinstance(poni,pyFAI.azimuthalIntegrator.AzimuthalIntegrator):
    ai = poni
  elif isinstance(poni,str):
    # look is file exists in cwd
    if os.path.isfile(poni):
      fname = poni
    # if file does not exist look for one with that name around
    else:
      # build search paths
      folders = []
      if folder is not None:
        temp = os.path.abspath(folder)
        path = pathlib.Path(temp)
        folders = [ str(path), ]
        for p in path.parents: folders.append(str(p))
      folders.append( os.curdir )
      folders.append( os.path.expanduser("~") )
      # look for file
      for path in folders:
        fname = os.path.join(path,poni)
        if os.path.isfile(fname):
          log.info("Found poni file %s",fname)
          break
        else:
          log.debug("Could not poni file %s",fname)
    ai = pyFAI.load(fname) if os.path.isfile(fname) else None
  else:
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator()
  if ai is not None:
    for par,value in kwargs.items(): setattr(ai,par,value)
    # provide xcen and ycen for convenience (note: xcen changes poni2
    # and ycen changes poni1)
    if 'pixel' in kwargs: ai.pixel1 = kwargs['pixel']
    if 'pixel' in kwargs: ai.pixel2 = kwargs['pixel']
    if 'xcen' in kwargs: ai.poni2 = kwargs['xcen'] * ai.pixel2
    if 'ycen' in kwargs: ai.poni1 = kwargs['ycen'] * ai.pixel1
    ai.reset(); # needed in case of overridden parameters
  else:
    log.warn("Could not find poni")
  return ai

def doFolder(folder="./",files='*.edf*',nQ = 1500,force=False,mask=None,dark=10,
    qlims=None,monitor='auto',save_pyfai=False,saveChi=True,poni='pyfai.poni',
    storageFile='auto',save=True,logDict=None,dezinger=None,skip_first=0,
    last=None):
  """ calculate 1D curves from files in folder

      Parameters
      ----------
      folder : str
          folder to work on
      files : str
          regular expression to look for ccd images (use edf* for including
          gzipped giles)
      nQ : int
          number of Q-points (equispaced)
      monitor : array or (qmin,qmax)
          normalization array (or list for q range normalization)
      force : True|False
          if True, redo from beginning even if previous data are found
          if False, do only new files
      mask : can be a list of [filenames|array of booleans|mask string]
          pixels that are True are dis-regarded
      saveChi : True|False
          if False, chi files (text based for each image) are not saved
      dezinger : None or 0<float<100
          use pyfai function 'separate' to remove zingers. The value is the 
          percentile used to find the liquicd baseline, 50 (i.e. median value)
          if a good approximation. Dezinger takes ~200ms per 4M pixel image.
          Needs good center and mask
      logDict : None or dictionary(-like)
          each key is a field. if given it has to have 'file' key
      poni : informationation necessary to build an AzimuthalIntegrator:
          → an AzimuthalIntegrator instance
          → a filename that will be look for in
               1 'folder' first
               2 in ../folder
               3 in ../../folder
               ....
               n-1 in pwd
               n   in homefolder
          → a dictionary (use to bootstrap an AzimuthalIntegrator using 
              AzimuthalIntegrator(**poni)
      save_pyfai : True|False
          if True, it stores all pyfai's internal arrays (~110 MB)
      skip_first : int
          skip the first images (the first one is sometime not ideal)
      last : int
          skip evey image after 'last'
 """

  func = inspect.currentframe()
  args = inspect.getargvalues(func)
  files_reg = files
  # store argument for saving ..
  args = dict( [(arg,args.locals[arg]) for arg in args.args] )

  folder = folder.replace("//","/").rstrip("/")

  # can't store aritrary objects
  if isinstance(args['poni'],pyFAI.AzimuthalIntegrator):
    args['poni'] = ai_as_dict(args['poni'])

  if storageFile == 'auto':
    fname = "pyfai_1d" + g_default_extension
    if not os.path.isdir(folder): 
      # do not overide folder, it might be useful
      storageFile = os.path.join( ".",fname)
    else:
      storageFile = os.path.join(folder,fname)

  if os.path.isfile(storageFile) and not force:
    saved = DataStorage(storageFile)
    log.info("Found %d images in storage file"%saved.data.shape[0])
    ai = getAI(poni,folder)
    # consistency check (saved images done with same parameters ?)
    if ai is not None:
      if (saved.pyfai_info != ai_as_str(ai)) or  \
          np.any( saved.mask != interpretMasks(mask,saved.mask.shape))  or \
          (saved.args.dezinger != dezinger) : 
        log.warn("Found inconsistency between curves already saved and new ones")
        log.warn("Redoing saved ones with new parameters")
        args['force'] = True
        saved = doFolder(**args)
  else:
    saved = None

  files = utils.getFiles(folder,files)
  if logDict is not None:
    files = [f for f in files if utils.getBasename(f) in logDict['file'] ]

    # sometime one deletes images but not corresponding lines in logfiles...
    if len(files)<len(logDict['file']):
      basenames = np.asarray( [ utils.getBasename(file) for file in files] )
      idx_to_keep = np.asarray([f in basenames for f in logDict['file']] )
      for key in logDict.keys(): logDict[key] = logDict[key][idx_to_keep]
      log.warn("More files in log than actual images, truncating loginfo")
      
  files = files[skip_first:last]


  if saved is not None:
    files = [f for f in files if utils.getBasename(f) not in saved["files"]]
  log.info("Will do azimuthal integration for %d files"%(len(files)))

  files = np.asarray(files)
  basenames = np.asarray( [ utils.getBasename(file) for file in files] )

  if len(files) > 0:
    # which poni file to use:
    ai = getAI(poni,folder)
    _msg = "could not interpret poni info or find poni file"
    if ai is None: raise ValueError(_msg)

    shape = read(files[0]).shape
    mask = interpretMasks(mask,shape)
      
    data   = np.empty( (len(files),nQ) )
    err    = np.empty( (len(files),nQ) )
    pbar = utils.progressBar(len(files))
    for ifname,fname in enumerate(files):
      img = read(fname)
      q,i,e = do1d(ai,img,mask=mask,npt_radial=nQ,dark=dark,dezinger=dezinger)
      data[ifname] = i
      err[ifname]  = e
      if saveChi:
        chi_fname = utils.removeExt(fname) + ".chi"
        utils.saveTxt(chi_fname,q,np.vstack((i,e)),info=ai_as_str(ai),overwrite=True)
      pbar.update(ifname+1)
    pbar.finish()
    if saved is not None:
      files = np.concatenate( (saved.orig.files  ,basenames ) )
      data  = np.concatenate( (saved.orig.data ,data  ) )
      err   = np.concatenate( (saved.orig.err  ,err   ) )
    else:
      files = basenames
    twotheta_rad = utils.qToTwoTheta(q,wavelength=ai.wavelength)
    twotheta_deg = utils.qToTwoTheta(q,wavelength=ai.wavelength,asDeg=True)
    orig = dict(data=data.copy(),err=err.copy(),q=q.copy(),
           twotheta_deg=twotheta_deg,twotheta_rad=twotheta_rad,files=files)
    ret = dict(folder=folder,files=files,orig = orig,pyfai=ai_as_dict(ai),
          pyfai_info=ai_as_str(ai),mask=mask,args=args)
    if not save_pyfai:
      ret['pyfai']['chia'] = None
      ret['pyfai']['dssa'] = None
      ret['pyfai']['q']    = None
      ret['pyfai']['ttha'] = None
 
    ret = DataStorage(ret)

  else:
    ret = saved

  if ret is None: return None

  if qlims is not None:
    idx = (ret.orig.q>=qlims[0]) & (ret.orig.q<=qlims[1])
  else:
    idx = np.ones_like(ret.orig.q,dtype=bool)

  ret.data = ret.orig.data[:,idx]
  ret.err  = ret.orig.err[:,idx]
  ret.q    = ret.orig.q[idx]
  ret.twotheta_rad = ret.orig.twotheta_rad
  ret.twotheta_deg = ret.orig.twotheta_deg

  if monitor == 'auto':
    monitor = ret.data.mean(1)
  elif isinstance(monitor,(tuple,list)):
    idx_norm = (ret.q >= monitor[0]) & (ret.q <= monitor[1])
    monitor = ret.data[:,idx_norm].mean(1)
  ret["data_norm"] = ret.data/monitor[:,np.newaxis]
  ret["err_norm"] = ret.err/monitor[:,np.newaxis]
  ret["monitor"] = monitor[:,np.newaxis]

  # add info from logDict if provided
  if logDict is not None: ret['log']=logDict
    # sometime saving is not necessary (if one has to do it after subtracting background
  if storageFile is not None and save: ret.save(storageFile)


  return ret


def removeBackground(data,qlims=(0,10),max_iter=30,background_regions=[],force=False,
      storageFile=None,save=True,**removeBkg):
  """ similar function to the zray.utils one, this works on dataset created by
      doFolder """
  idx = utils.findSlice(data.orig.q,qlims)
  # see if there are some to do ...
  if force:
    idx_start = 0
  else:
    idx_start = len(data.data)
  if idx_start < len(data.orig.data):
    _q,_data = utils.removeBackground(data.orig.q[idx],data.orig.data[:,idx],
               max_iter=max_iter,background_regions=background_regions,**removeBkg)
    data.q = _q
    data.data  = np.concatenate( (data.data,_data  ) )
    data.err   = np.concatenate( (data.err ,data.err[idx_start,idx]   ) )
  if save: data.save(storageFile); # if None uses .filename
  return data



def _calc_R(x,y, xc, yc):
  """ calculate the distance of each 2D points from the center (xc, yc) """
  return np.sqrt((x-xc)**2 + (y-yc)**2)

def _chi2(c, x, y):
  """ calculate the algebraic distance between the data points and the mean
      circle centered at c=(xc, yc) """
  Ri = _calc_R(x, y, *c)
  return Ri - Ri.mean()

def leastsq_circle(x,y):
  from scipy import optimize
  # coordinates of the barycenter
  center_estimate = np.nanmean(x), np.nanmean(y)
  center, ier = optimize.leastsq(_chi2, center_estimate, args=(x,y))
  xc, yc = center
  Ri       = _calc_R(x, y, *center)
  R        = Ri.mean()
  residu   = np.sum((Ri - R)**2)
  return xc, yc, R

def find_center(img,psize=100e-6,dist=0.1,wavelength=0.8e-10,center=None,reference=None,**kwargs):
  """ center is the initial centr (can be None)
      reference is a reference position to be plot in 2D plots """
  plt.ion()
  kw = dict( pixel1 = psize, pixel2 = psize, dist = dist,wavelength=wavelength )
  kw.update(kwargs)
  ai =  pyFAI.azimuthalIntegrator.AzimuthalIntegrator(**kw)
  fig_img,ax_img = plt.subplots(1,1)
  fig_pyfai,ax_pyfai = plt.subplots(1,1)
  fig_pyfai = plt.figure(2)
  temp= ax_img.imshow(img)
  plt.sca(ax_img); # set figure to use for mouse interaction
  temp.set_clim( *np.percentile(img,(2,95) ) )
  ans = ""
  print("Enter 'end' when done")
  while ans != "end":
    if center is None:
      print("Click on beam center:")
      plt.sca(ax_img); # set figure to use for mouse interaction
      center = plt.ginput()[0]
    print("Selected center:",center)
    ai.set_poni1(center[1]*psize)
    ai.set_poni2(center[0]*psize)
    q,az,i = do2d(ai,img)
    mesh = ax_pyfai.pcolormesh(q,az,i)
    mesh.set_clim( *np.percentile(i,(2,95) ) )
    ax_pyfai.set_title(str(center))
    if reference is not None: ax_pyfai.axvline(reference)
    plt.pause(0.01)
    plt.draw()
    plt.draw_all()
    ans=input("Enter to continue with clinking or enter xc,yc values ")
    if ans == '':
      center = None
    else:
      try:
        center = list(map(float,ans.split(",")))
      except Exception as e:
        center = None
    if center == []: center = None
  print("Final values: (in pixels) %.3f %.3f"%(center[0],center[1]))
  return ai

def average(fileOrFolder,delays=slice(None),scale=1,norm=None,returnAll=False,plot=False,
  showTrend=False):
  data = DataStorage(fileOrFolder)
  if isinstance(delays,slice):
    idx = np.arange(data.delays.shape[0])[delays]
  elif isinstance(delays,(int,float)):
    idx = data.delays == float(delays)
  else:
    idx = data.delays < 0
  if idx.sum() == 0:
    print("No data with the current filter")
    return None
  i   = data.data[idx]
  q   = data.q
  if isinstance(norm,(tuple,list)):
    idx  = ( q>norm[0] ) & (q<norm[1])
    norm = np.nanmean(i[:,idx],axis=1)
    i = i/norm[:,np.newaxis]
  if isinstance(norm,np.ndarray):
    i = i/norm[:,np.newaxis]
  title = "%s %s" % (fileOrFolder,str(delays))
  utils.plotdata(q,i*scale,showTrend=showTrend,plot=plot,title=title)
  if returnAll:
    return q,i.mean(axis=0)*scale,i
  else:
    return q,i.mean(axis=0)*scale

#### Utilities for chi files ####
def chiRead(fname,scale=1):
  q,i = np.loadtxt(fname,unpack=True,usecols=(0,1))
  return q,i*scale

def chiPlot(fname,useTheta=False,E=12.4):
  q,i = chiRead(fname)
  lam = 12.4/E
  theta = 2*180/3.14*np.arcsin(q*lam/4/3.14)
  x = theta if useTheta else q
  plt.plot(x,i,label=fname)



def chiAverage(folder,basename="",scale=1,norm=None,returnAll=False,plot=False,showTrend=False,clim='auto'):
  files = glob.glob(os.path.joint(folder,"%s*chi"%basename))
  files.sort()
  print(files)
  if len(files) == 0:
    print("No file found (basename %s)" % basename)
    return None
  q,_ = chiRead(files[0])
  i   = np.asarray( [ chiRead(f)[1] for f in files ] )
  if isinstance(norm,(tuple,list)):
    idx  = ( q>norm[0] ) & (q<norm[1])
    norm = np.nanmean(i[:,idx],axis=1)
    i = i/norm[:,np.newaxis]
  elif isinstance(norm,np.ndarray):
    i = i/norm[:,np.newaxis]
  title = "%s %s" % (folder,basename)
  utils.plotdata(q,i,plot=plot,showTrend=showTrend,title=title,clim=clim)
  if (showTrend and plot): plt.subplot(1,2,1)
  if showTrend:
    plt.pcolormesh(np.arange(i.shape[0]),q,i.T)
    plt.xlabel("image number, 0 being older")
    plt.ylabel(r"q ($\AA^{-1}$)")
  if (showTrend and plot): plt.subplot(1,2,2)
  if plot:
    plt.plot(q,i.mean(axis=0)*scale)
  if (plot or showTrend):
    plt.title(folder+"/"+basename)
  if returnAll:
    return q,i.mean(axis=0)*scale,i
  else:
    return q,i.mean(axis=0)*scale
