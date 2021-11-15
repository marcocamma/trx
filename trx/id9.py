# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)

import time
import os
import collections
import numpy as np
import copy
from . import azav
from . import dataReduction
from . import utils
from . import filters
from datastorage import DataStorage

default_extension = ".h5"

def _conv(x):
    try:
        x = float(x)
    except:
        x = np.nan
    return x

def _readDiagnostic(fname,retry=3):
  ntry = 0
  while ntry<retry:
    try:
      data = np.genfromtxt(fname,usecols=(2,3),\
             dtype=None,converters={3: lambda x: _conv(x)},
             names = ['fname','delay'])
      return data
    except Exception as e:
      log.warn("Could not read diagnostic file, retrying soon,error was %s"%e)
      ntry += 1
      time.sleep(0.2)
  # it should not arrive here
  raise ValueError("Could not read diagnostic file after %d attempts"%retry)

def readDiagnostic(fname):
  """ return an ordered dict dictionary of filename; for each key a rounded
      value of delay is associated """
  if os.path.isdir(fname): fname += "/diagnostics.log"
  # try to read diagnostic couple of times
  data = _readDiagnostic(fname,retry=4)
  files = data['fname'].astype(str)
  delays = data['delay']
  # skip lines that cannot be interpreted as float (like done, etc)
  idx_ok = np.isfinite( delays )
  files = files[idx_ok]
  files = np.asarray( [utils.getBasename(f) for f in files]) 
  delays = delays[idx_ok]
  delays = np.round(delays.astype(float),12)
  return dict( file = files, scan = delays) 

def _findDark(line):
  _,value = line.split(":")
  return float(value)

def _delayToNum(delay):
  if delay == 'off':
    delay = -10.0 # .0 is necessary to force float when converting arrays
  else:
    delay=utils.strToTime(delay)
  return float(delay)

def timesToInfo(times):
    last = times[-1][:-3]
    first = times[0][:-3]
    info = "%s-%s" % (first,last)

    delta_h = int(last.split(":")[0])-int(first.split(":")[0])
    delta_m = int(last.split(":")[1])-int(first.split(":")[1])
    if delta_m < 0:
        delta_h -= 1
        delta_m += 60
    if delta_h < 0: delta_h += 24
    dt = delta_h*60+delta_m
    if dt < 60:
        info += "\n%d mins" % dt
    else:
        info += "\n%dh %dm (%d mins)"%(delta_h,delta_m,dt)
    return info

def readReprate(fname):
    try:
        with open(fname,"r") as f:
            lines = f.readlines()
    except OSError:
        return 0
    line = [ temp for temp in lines if temp.find("time between pulses")>0 ][0]

    waittime = line.split(":")[1].split("/")[0]
    reprate = 1/float(waittime)
    return reprate


def findLogFile(folder):
  files = utils.getFiles(folder,basename='*.log')

  # remove diagnostic if present
  diag = os.path.join(folder,"diagnostics.log")
  if diag in files:
    files.remove(diag)

  logfile = files[0]
  if len(files)>1: log.warn("Found more than one *.log file that is not diagnostics.log: %s"%files)
  return logfile

def readLogFile(fnameOrFolder,subtractDark=False,skip_first=0,
    asDataStorage=True,last=None,srcur_min=30):
    """ read id9 style logfile; 
        last before data will be used as keys ... 
        only srcur>srcur_min will be kept
        subtractDark is not needed for data collected with waxscollect
    """
    if os.path.isdir(fnameOrFolder):
      fname = findLogFile(fnameOrFolder)
    else:
      fname = fnameOrFolder
    log.info("Reading id9 logfile: %s"%fname)

    data = utils.files.readLogFile(fname,skip_first=skip_first,last=last,\
           output = "array",converters=dict(delay=_delayToNum))

 
    # work on darks if needed
    if subtractDark:
        ## find darks
        with open(fname,"r") as f: lines = f.readlines()
        lines = [line.strip() for line in lines]
        # look only for comment lines
        lines = [ line for line in lines if line[0] == "#" ]
        for line in lines:
            if line.find("pd1 dark/sec")>=0: darks['pd1ic'] = _findDark(line)
            if line.find("pd2 dark/sec")>=0: darks['pd2ic'] = _findDark(line)
            if line.find("pd3 dark/sec")>=0: darks['pd3ic'] = _findDark(line)

        ## subtract darks
        for diode in ['pd1ic','pd2ic','pd3ic','pd4ic']:
            if diode in darks: data[diode]=data[diode]-darks[diode]*data['timeic']

    # srcur filter
    if "currentmA" in data.dtype.names:
        idx_cur = data['currentmA'] > srcur_min
        if (idx_cur.sum() < idx_cur.shape[0]*0.5):
            log.warn("Minimum srcur filter has kept only %.1f%%"%(idx_cur.sum()/idx_cur.shape[0]*100))
            log.warn("Minimum srcur: %.2f, median(srcur): %.2f"%(srcur_min,np.nanmedian(data["currentmA"]))) 
        data = data[idx_cur]
    else:
        log.warn("Could not find currentmA in logfile, skipping filtering")


    info = DataStorage()

    # usually folders are named sample/run
    if os.path.isdir(fnameOrFolder):
        folder = fnameOrFolder
    else:
        folder = os.path.dirname(fnameOrFolder)
    dirs = folder.split(os.path.sep)
    ylabel = ".".join(dirs[-2:])
    info.name = ".".join(dirs[-2:])


    try:
        reprate= readReprate(fname)
        info.reprate = reprate
        ylabel += " %.2f Hz"%reprate
    except:
        log.warn("Could not read time duration info")

    try:
        time_info = timesToInfo(data['time'])
        info.duration = time_info
        ylabel += "\n" + time_info
    except:
        log.warn("Could not read time duration info")

    info.ylabel = ylabel

    if asDataStorage:
        data = DataStorage( dict((name,data[name]) for name in data.dtype.names ) )


    return data,info


def doFolder_azav(folder,nQ=1500,files='*.edf*',force=False,mask=None,
  saveChi=True,poni='pyfai.poni',storageFile='auto',dark=9.9,dezinger=None,
  qlims=None,monitor='auto',skip_first=0,last=None,srcur_min=80,detector=None, azimuth_range = None):
  """ very small wrapper around azav.doFolder, essentially just reading
      the id9 logfile or diagnostics.log
      monitor  : normalization vector that can be given as
               1. numpy array
               2. a list (interpreted as q-range normalization)
               3. a string to look for as key in the log, e.g.
                  monitor="pd2ic" would reult in using
                  azavStorage.log.pd2ic
      detector : if not None, file names in logfiles will have an _detname appended
 """

  try:
    loginfo,extra_info = readLogFile(folder,skip_first=skip_first,last=last,
              srcur_min=srcur_min)
  except Exception as e:
    log.warn("Could not read log file, trying to read diagnostics.log")
    loginfo = readDiagnostic(folder)
  if detector is not None:
    loginfo.file = np.asarray( [f+"_" + detector for f in loginfo.file ] )
    files = files.replace(".edf","_%s.edf"%detector)
  if storageFile == 'auto' : storageFile = folder +  "/" + "pyfai_1d" + default_extension

  if monitor != "auto" and isinstance(monitor,str): monitor = loginfo[monitor]

  data = azav.doFolder(folder,files=files,nQ=nQ,force=force,mask=mask,
    saveChi=saveChi,poni=poni,storageFile=storageFile,logDict=loginfo,
    dark=dark,save=False,dezinger=dezinger,qlims=qlims,monitor=monitor, azimuth_range = azimuth_range)
  data.save(storageFile)
  data.info = extra_info
  return data


def doFolder_dataRed(azavStorage,funcForAveraging=np.nanmean,
                     outStorageFile='auto',reference='min',chi2_0_max='auto',
                     saveTxt=True,first=None,last=None,idx=None,split_angle=False):
  """ azavStorage if a DataStorage instance or the filename to read 
  """


  if isinstance(azavStorage,DataStorage):
    azav = azavStorage
    folder = azavStorage.folder
  elif os.path.isfile(azavStorage):
    folder = os.path.dirname(azavStorage)
    azav = DataStorage(azavStorage)
  else:
    # assume is just a folder name
    folder = azavStorage
    azavStorage  = folder +  "/pyfai_1d" + default_extension
    azav = DataStorage(azavStorage)


  if split_angle:
      angles = np.unique(azav.log.angle)
      diffs = []
      for angle in angles:
          idx = azav.log.angle == angle
          diffs.append(
                  doFolder_dataRed(azav,funcForAveraging=funcForAveraging,
                      outStorageFile=None,reference=reference,
                      chi2_0_max=chi2_0_max,saveTxt=False,
                      idx=idx,split_angle=False)
                  )
      ret = DataStorage(angles=angles,diffs=diffs)
      if outStorageFile == 'auto':
        if not os.path.isdir(folder): folder = "./"
        outStorageFile = folder + "/diffs" + default_extension
      if outStorageFile is not None:
        ret.save(outStorageFile)
      return ret

  azav = copy.deepcopy(azav)

  if last is not None or first is not None and idx is None:
      idx = slice(first,last)

  if idx is not None:
      azav.log.delay = azav.log.delay[idx]
      azav.data_norm = azav.data_norm[idx]
      azav.err_norm = azav.err_norm[idx]


  # laser off is saved as -10s, if using the automatic "min"
  # preventing from using the off images
  # use reference=-10 if this is what you want
  if reference == "min":
      reference = azav.log.delay[azav.log.delay!= -10].min()

  # calculate differences
  tr = dataReduction.calcTimeResolvedSignal(azav.log.delay,azav.data_norm,
          err=azav.err_norm,q=azav.q,reference=reference,
          funcForAveraging=funcForAveraging,chi2_0_max=chi2_0_max)

  tr.folder = folder
  tr.twotheta_rad = azav.twotheta_rad
  tr.twotheta_deg = azav.twotheta_deg
  tr.info = azav.pyfai_info

  if outStorageFile == 'auto':
    if not os.path.isdir(folder): folder = "./"
    outStorageFile = folder + "/diffs" + default_extension
  tr.filename = outStorageFile

  # save txt and npz file
  if saveTxt: dataReduction.saveTxt(folder,tr,info=azav.pyfai_info)

  if outStorageFile is not None:
    tr.save(outStorageFile)

  return tr

def doFolder(folder,azav_kw = dict(), datared_kw = dict(),online=True, retryMax=20,force=False):
  import matplotlib.pyplot as plt
  if folder == "./": folder = os.path.abspath(folder)
  fig = plt.figure()
  lastNum = None
  keepGoing = True
  lines   = None
  retryNum = 0
  azav_kw['force']=force
  if online: print("Press Ctrl+C to stop")
  while keepGoing and retryNum < retryMax:
    try:
      azav = doFolder_azav(folder,**azav_kw)
      # check if there are new data
      if lastNum is None or lastNum<azav.data.shape[0]:
        tr = doFolder_dataRed(azav,**datared_kw)
        if lines is None or len(lines) != tr.diffs.shape[0]:
          lines,_ = utils.plotdiffs(tr,fig=fig,title=folder)
        else:
          utils.updateLines(lines,tr.diffs)
        plt.draw()
        lastNum = azav.data.shape[0]
        retryNum = 0
      else:
        retryNum += 1
      if online: plt.pause(30)
      if force: azav_kw['force']=False; # does not make sense to have always True ...
    except KeyboardInterrupt:
      keepGoing = False
    if not online: keepGoing = False
  return azav,tr


def readMotorDump(fnameOrFolder,asDataStorage=True,\
    default_fname="motor_position_after_data_collection.txt"):
  """ 
      Read waxecollect style motor dump
      if fnameOrFolder is a folder, default_fname is read
      if asDataStorage is False:
        return recArray with fields name,user,dial
      else: return dictory like object (each motor is a key)
  """
  if os.path.isfile(fnameOrFolder):
    fname = fnameOrFolder
  else:
    fname = os.path.join(fnameOrFolder,default_fname)
  data = np.genfromtxt(fname,names=True,dtype=("<U15",float,float))
  # remove interleaved headers
  idx_to_remove = data['name'] == 'name'
  data = data[~idx_to_remove]
#  for i in range(data.shape[0]): data['name'][i] = data['name'][i].decode('ascii')
  if asDataStorage:
    motor_pos = collections.namedtuple('motor_pos',['user','dial'])
    ret = dict()
    for imotor,motor in enumerate(data['name']):
      ret[motor] = motor_pos(dial=data['dial'][imotor],user=data['user'][imotor])
    data = DataStorage(ret)
  return data
