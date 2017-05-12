# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)

try:
  import lmfit
except ImportError as e:
  log.error("To use this submodule, please install `lmfit`")
import numpy as np
import logging as log

def fitPeak(x,y,err=1,autorange=False):
  pv = lmfit.models.PseudoVoigtModel()
  if isinstance(err,np.ndarray):
    if np.all(err==0):
      err = 1
      log.warn("Asked to fit peak but all errors are zero, forcing them to 1")
    elif np.isfinite(err).sum() != len(err):
      idx = np.isfinite(err)
      x = x[idx]
      y = y[idx]
      err = err[idx]
      log.warn("Asked to fit peak but some errorbars are infinite or nans,\
                excluding those points")
  if autorange:
    # find fwhm
    idx = np.ravel(np.argwhere( y<y.max()/2 ))
    # find first crossing
    p1 = idx[idx<np.argmax(y)][-1]
    p2 = idx[idx>np.argmax(y)][0]
    c = int( (p1+p2)/2 )
    dp = int( np.abs(p1-p2) )
    idx = slice(c-dp,c+dp)
    x = x[idx]
    y = y[idx]
  pars = pv.guess(y,x=x)
  ret  = pv.fit(y,x=x,weights=1/err,params=pars)
  return ret
