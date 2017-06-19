# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import os
import numpy as np
np.seterr(all='ignore')

try:
  import progressbar as pb
  _has_progress_bar = True
except ImportError:
  _has_progress_bar = False
  log.warn("Reccomended package: progressbar is missing")



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


