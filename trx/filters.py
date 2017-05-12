# -*- coding: utf-8 -*-
""" 
  module that contains filters and outliers removal procedures 
  most of them return the data array and a dictionary with additional info
  (parameters, statistics, etc)
"""
from __future__ import print_function,division,absolute_import
from . import utils
import copy
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

def applyFilters(data,funcForAveraging=np.nanmean):
  # make copy in this way tr1 = trx.filters.applyFilters(tr) does not modity tr
  data = copy.deepcopy(data)
  if not "filters" in data: return data
  if not "unfiltered" in data: data.unfiltered = \
      dict( diffs_in_scan = data.diffs_in_scan,
            chi2_0=data.chi2_0,
            diff=data.diffs )
  data.diffs_in_scan = data.unfiltered.diffs_in_scan
  filters = data.filters.keys()
  for filt_name in filters:
    filt = data.filters[filt_name]
    # understand what kind of filter (q-by-q or for every image)
    if filt[0].ndim == 1:
      for nscan in range(len(data.diffs_in_scan)):
        data.diffs_in_scan[nscan] = data.diffs_in_scan[nscan][~filt[nscan]]
        data.diffs[nscan] = funcForAveraging( data.diffs_in_scan[nscan],axis=0)
    elif filt[0].ndim == 2: # q-by-q kind of filter
      for nscan in range(len(data.diffs_in_scan)):
        data.diffs_in_scan[nscan][~filt[nscan]] = np.nan
        data.diffs[nscan] = funcForAveraging( data.diffs_in_scan[nscan],axis=0)
  data.diffs_plus_ref = data.diffs+data.ref_average
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

def chi2Filter(data,threshold='auto'):
  """ Contrary to removeZingers, this removes entire curves """
  if threshold == "auto":
    threshold=np.percentile(np.concatenate(data.chi2_0),95)
  idx_mask = []
  for iscan in range(len(data.diffs_in_scan)):
    idx = data.chi2_0[iscan] > threshold
    # expand along other axis (q ...)
    #idx = utils.reshapeToBroadcast(idx,data.diffsInScanPoint[iscan])
    idx_mask.append(idx)
    log.info("Chi2 mask, scanpoint: %s, curves filtereout out %d/%d (%.2f%%)"%\
              (data.diffs[iscan],idx.sum(),len(idx),idx.sum()/len(idx)*100) )

  if "filters" not in data: data.filters = dict()
  if "filters_pars" not in data: data.filters_pars = dict()
  data.filters.chi2 = idx_mask
  data.filters_pars.chi2_threshold = threshold
  return data

