# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

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

def readLogFile(fnameOrFolder,subtractDark=False,skip_first=0,
    asDataStorage=True,last=None,srcur_min=30):
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
  idx_cur = data['currentmA'] > srcur_min
  data = data[idx_cur]
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
  qlims=None,monitor='auto',skip_first=0,last=None,srcur_min=80):
  """ very small wrapper around azav.doFolder, essentially just reading
      the id9 logfile or diagnostics.log
      monitor  : normalization vector that can be given as
               1. numpy array
               2. a list (interpreted as q-range normalization)
               3. a string to look for as key in the log, e.g.
                  monitor="pd2ic" would reult in using
                  azavStorage.log.pd2ic


 """

  try:
    loginfo = readLogFile(folder,skip_first=skip_first,last=last,
              srcur_min=srcur_min)
  except Exception as e:
    log.warn("Could not read log file, trying to read diagnostics.log")
    loginfo = readDiagnostic(folder)
  if storageFile == 'auto' : storageFile = folder +  "/" + "pyfai_1d" + default_extension

  if monitor != "auto" and isinstance(monitor,str): monitor = loginfo[monitor]

  data = azav.doFolder(folder,files=files,nQ=nQ,force=force,mask=mask,
    saveChi=saveChi,poni=poni,storageFile=storageFile,logDict=loginfo,
    dark=dark,save=False,dezinger=dezinger,qlims=qlims,monitor=monitor)
  data.save(storageFile)
  return data



def doFolder_dataRed(azavStorage,funcForAveraging=np.nanmean,
                     outStorageFile='auto',reference='min',chi2_0_max='auto',
                     saveTxt=True):
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


  # calculate differences
  tr = dataReduction.calcTimeResolvedSignal(azav.log.delay,azav.data_norm,
          err=azav.err_norm,q=azav.q,reference=reference,
          funcForAveraging=funcForAveraging,chi2_0_max=chi2_0_max)

  tr.folder = folder
  if outStorageFile == 'auto':
    if not os.path.isdir(folder): folder = "./"
    outStorageFile = folder + "/diffs" + default_extension
  tr.filename = outStorageFile

  # save txt and npz file
  if saveTxt: dataReduction.saveTxt(folder,tr,info=azav.pyfai_info)

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
  
