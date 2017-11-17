# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import os
import glob
import pathlib
import re
import numpy as np
from datastorage import DataStorage

def getFiles(folder,basename="*.edf*",nFiles=None):
  files = glob.glob(os.path.join(folder,basename))
  files.sort()
  if nFiles is not None: files = files[:nFiles]
  return files

def getEdfFiles(folder,**kw):
  return getFiles(folder,basename="*.edf*",**kw)

def removeExt(fname):
  """ special remove extension meant to work with compressed files.edf and .edf.gz files """
  if fname[-3:] == ".gz": fname = fname[:-3]
  return os.path.splitext(fname)[0]

def getBasename(fname):
  return os.path.basename(removeExt(fname));

def readLogFile(fname,skip_first=0,last=None,converters=None,
    output="datastorage"):
  """ read generic log file efficiently
      lines starting with "#" will be skipped
      last line starting with # will be used to find the keys
      converters is used convert a certain field. (see np.genfromtxt)
      
      output is a string, if 'datastorage' or 'dict' data is converted
      else it is left as recarray
  """
  # makes 'output' case insentive
  if isinstance(output,str): output = output.lower()

  with open(fname,"r") as f: lines = f.readlines()
  lines = [line.strip() for line in lines]

  # find last line starting with "#"
  for iline,line in enumerate(lines):
      if line.lstrip()[0] != "#": break

  # extract names (numpy can do it but gets confused with "# dd#
  # as it does not like the space ...
  names = lines[iline-1][1:].split()

  data=np.genfromtxt(fname,skip_header=iline,names=names,dtype=None,
       converters = converters, excludelist = [] )

  # skip firsts/lasts
  data = data[skip_first:last]
  
  # force previously found names, numpy changes file to file_
  names = [ name.strip("_") for name in data.dtype.names ]
  data.dtype.names = names

  # convert to string columns that can be
  dtype = data.dtype.descr
  newtype = []
  for (name,type_str) in dtype:
      name = name.strip("_"); # numpy changes file to file_
      type_str = type_str.replace("|S","<U")
      newtype.append( (name,type_str) )
  data = data.astype(newtype) 
  

  if output.lower() == "dict":
      # convert to dict
      data = dict((name,data[name]) for name in data.dtype.names )
  elif output.lower() == "datastorage":
      data = dict((name,data[name]) for name in data.dtype.names )
      data = DataStorage( data )

  return data

  
