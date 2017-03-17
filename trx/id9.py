import logging
log = logging.getLogger(__name__)

import os
import collections
import numpy as np
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
  if delay.decode('ascii') == 'off':
    delay = -10
  else:
    delay=utils.strToTime(delay)
  return delay

def findLogFile(folder):
  files = utils.getFiles(folder,basename='*.log')
  files.remove(os.path.join(folder,"diagnostics.log"))
  logfile = files[0]
  if len(files)>1: log.warn("Found more than one *.log file that is not diagnostics.log: %s"%files)
  return logfile

def readLogFile(fnameOrFolder,subtractDark=False,skip_first=0,asDataStorage=True,last=None):
  """ read id9 style logfile """
  if os.path.isdir(fnameOrFolder):
    fname = findLogFile(fnameOrFolder)
  else:
    fname = fnameOrFolder
  f = open(fname,"r")
  lines = f.readlines()
  f.close()
  lines = [line.strip() for line in lines]
  darks = {}
  for line in lines:
    if line.find("pd1 dark/sec")>=0: darks['pd1ic'] = _findDark(line)
    if line.find("pd2 dark/sec")>=0: darks['pd2ic'] = _findDark(line)
    if line.find("pd3 dark/sec")>=0: darks['pd3ic'] = _findDark(line)
    if line.find("pd4 dark/sec")>=0: darks['pd4ic'] = _findDark(line)
  for iline,line in enumerate(lines):
    if line.lstrip()[0] != "#": break
  data=np.genfromtxt(fname,skip_header=iline-1,names=True,comments="%",dtype=None,converters = {'delay': lambda s: _delayToNum(s)})
  if subtractDark:
    for diode in ['pd1ic','pd2ic','pd3ic','pd4ic']:
      if diode in darks: data[diode]=data[diode]-darks[diode]*data['timeic']
  data = data[skip_first:last]
  if asDataStorage:
    # rstrip("_") is used to clean up last _ that appera for some reason in file_
    data = DataStorage( dict((name.rstrip("_"),data[name]) for name in data.dtype.names ) )
  data.file = data.file.astype(str)
  return data



def doFolder_azav(folder,nQ=1500,files='*.edf*',force=False,mask=None,
  saveChi=True,poni='pyfai.poni',storageFile='auto',dark=9.9,dezinger=None,
  qlims=(0,10),removeBack=False,removeBack_kw=dict(),skip_first=0,
  last=None):
  """ very small wrapper around azav.doFolder, essentially just reading
      the id9 logfile or diagnostics.log """

  try:
    loginfo = readLogFile(folder,skip_first=skip_first,last=last)
  except Exception as e:
    log.warn("Could not read log file, trying to read diagnostics.log")
    loginfo = readDiagnostic(folder)
  if storageFile == 'auto' : storageFile = folder +  "/" + "pyfai_1d" + default_extension

  data = azav.doFolder(folder,files=files,nQ=nQ,force=force,mask=mask,
    saveChi=saveChi,poni=poni,storageFile=storageFile,logDict=loginfo,dark=dark,save=False,dezinger=dezinger)
  #try:
  #  if removeBack is not None:
  #    _,data.data = azav.removeBackground(data,qlims=qlims,**removeBack_kw)
  #except Exception as e:
  #  log.error("Could not remove background, error was %s"%(str(e)))

#  idx = utils.findSlice(data.q,qlims)
#  n   = np.nanmean(data.data[:,idx],axis=1)
#  data.norm_range = qlims
#  data.norm = n
#  n   = utils.reshapeToBroadcast(n,data.data)
#  data.data_norm = data.data/n

  data.save(storageFile)


  return data



def doFolder_dataRed(azavStorage,monitor=None,funcForAveraging=np.nanmean,
                     qlims=None,outStorageFile='auto',reference='min'):
  """ azavStorage if a DataStorage instance or the filename to read 
      monitor  : normalization vector that can be given as
               1. numpy array
               2. a list (interpreted as q-range normalization)
               3. a string to look for as key in the log, e.g.
                  monitor="pd2ic" would reult in using
                  azavStorage.log.pd2ic

  """

  if isinstance(azavStorage,datastorage.datastorage.DataStorage):
    data = azavStorage
    folder = azavStorage.folder
  elif os.path.isfile(azavStorage):
    folder = os.path.dirname(azavStorage)
    data = DataStorage(azavStorage)
  else:
    # assume is just a folder name
    folder = azavStorage
    azavStorage  = folder +  "/pyfai_1d" + default_extension
    data = DataStorage(azavStorage)

  #assert data.q.shape[0] == data.data.shape[1] == data.err.shape[1]
  if qlims is not None:
    idx = (data.q>qlims[0]) & (data.q<qlims[1])
    data.data = data.data[:,idx]
    data.err  = data.err[:,idx]
    data.q    = data.q[idx]

  if isinstance(monitor,str): monitor = data['log'][monitor]

  # calculate differences
  diffs = dataReduction.calcTimeResolvedSignal(data.log.delay,data.data,
          err=data.err,q=data.q,reference=reference,monitor=monitor,
          funcForAveraging=funcForAveraging)

  # save txt and npz file
  dataReduction.saveTxt(folder,diffs,info=data.pyfai_info)
  if outStorageFile == 'auto':
    outStorageFile = folder + "/diffs" + default_extension
  diffs.save(outStorageFile)

  return data,diffs

def doFolder(folder,azav_kw = dict(), datared_kw = dict(),online=True, retryMax=20,force=False):
  import matplotlib.pyplot as plt
  if folder == "./": folder = os.path.abspath(folder)
  fig = plt.figure()
  lastNum = None
  keepGoing = True
  lines   = None
  retryNum = 0
  if online: print("Press Ctrl+C to stop")
  while keepGoing and retryNum < retryMax:
    try:
      data = doFolder_azav(folder,**azav_kw)
      # check if there are new data
      if lastNum is None or lastNum<data.data.shape[0]:
        data,diffs = doFolder_dataRed(data,**datared_kw)
        if lines is None or len(lines) != diffs.data.shape[0]:
          lines,_ = utils.plotdiffs(diffs,fig=fig,title=folder)
        else:
          utils.updateLines(lines,diffs.data)
        plt.draw()
        lastNum = data.data.shape[0]
        retryNum = 0
      else:
        retryNum += 1
      plt.pause(30)
      if force: force=False; # does not make sense to have always True ...
    except KeyboardInterrupt:
      keepGoing = False
    if not online: keepGoing = False
  return data,diffs
