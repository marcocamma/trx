""" 
  module that contains filters and outliers removal procedures 
  most of them return the data array and a dictionary with additional info
  (parameters, statistics, etc)
"""
from __future__ import print_function,division
from . import utils
import logging
import statsmodels.robust
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import numpy as np
np.seterr(all='ignore')

def applyFilter(data,boolArray):
  for key in data.keys():
    if isinstance(data[key],np.ndarray) and \
       (data[key].shape[0]==boolArray.shape[0]):
      data[key] = data[key][boolArray]
    elif isinstance(data[key],dict) and key != 'orig':
      data[key]=applyFilter(data[key],boolArray)
  return data

def removeZingers(curves,errs=None,norm='auto',threshold=10,useDerivative=False):
  """ curves will be normalized internally 
      if errs is None, calculate mad based noise 
      useDerivative for data with trends ..
  """
  # normalize
  if norm == 'auto':
    norm = np.nanmean(curves,axis=1)
    norm = utils.reshapeToBroadcast(norm,curves)

  if useDerivative:
    data = np.gradient(curves/norn,axis=0)
  else:
    data = curves/norm

  median = np.median(data,axis=0)

  # calculate or normalize error
  if errs is None:
    errs   = statsmodels.robust.mad(data,axis=0)
  else:
    errs   = errs/norm

  diff   = np.abs(data-median)/errs
  idx    = diff > threshold
  log.debug("Removed %d zingers from %d curves"%(idx.sum(),len(curves)))
  print("Removed %d zingers from %d curves"%(idx.sum(),len(curves)))
  if idx.sum()>0:
    curves[idx]=np.nan
    #curves = np.ma.MaskedArray(data=curves,mask=idx)
  return curves

def filterOutlier(curves,errs=None,norm=None,threshold=10):
  # normalize
  if norm == 'auto':
    norm = np.nanmean(curves,axis=1)
    norm = utils.reshapeToBroadcast(n,curves)
  elif norm is None:
    norm = 1

  curves = curves/norm
  if errs is None:
    errs   = statsmodels.robust.mad(curves,axis=0)
  else:
    errs   = errs/norm

  median = np.median(curves)
  diff   = np.abs(curves-median)/errs
  chi2   = np.sum(diff**2)/len(curves)
  idx    = chi2 < threshold
  return curves[idx]

def chi2Filter(diffs,threshold=10):
  """ Contrary to removeZingers, this removes entire curves """
  idx_mask = []
  for iscan in range(len(diffs.diffsInScanPoint)):
    idx = diffs.chi2_0[iscan] > threshold
    # expand along other axis (q ...)
    #idx = utils.reshapeToBroadcast(idx,data.diffsInScanPoint[iscan])
    idx_mask.append(idx)
    log.debug("Chi2 mask, scanpoint: %s, curves filtereout out %d/%d (%.2f%%)"%\
              (data.scan[iscan],idx.sum(),len(idx),idx.sum()/len(idx)*100) )
    print("Chi2 mask, scanpoint: %s, curves filtereout out %d/%d (%.2f%%)"%\
              (data.scan[iscan],idx.sum(),len(idx),idx.sum()/len(idx)*100) )

  if "masks" not in data: data['masks'] = dict()
  if "masks_pars" not in data: data['masks_pars'] = dict()
  data['masks']['chi2'] = idx_mask
  data['masks_pars']['chi2_threshold'] = threshold
  return data

