# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import os
import glob
import pathlib
import re

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


