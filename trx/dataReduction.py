# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)

import numpy as np
np.seterr(all='ignore')
from . import utils
from . import filters
from datastorage import DataStorage
import os

def subtractReferences(i,idx_ref, useRatio = False):
  """ given data in i (first index is shot num) and the indeces of the 
      references (idx_ref, array of integers) it interpolates the closest
      reference data for each shot and subtracts it (or divides it, depending
      on useRatio = [True|False]; 
      Note: it works in place (i.e. it modifies i) """
  iref=np.empty_like(i)
  idx_ref = np.squeeze(idx_ref)
  idx_ref = np.atleast_1d(idx_ref)
  # sometime there is just one reference (e.g. sample scans)
  if idx_ref.shape[0] == 1:
    if useRatio:
      return i/i[idx_ref]
    else:
      return i-i[idx_ref]
  # references before first ref are "first ref"
  iref[:idx_ref[0]] = i[idx_ref[0]]
  # references after last ref are "last ref"
  iref[idx_ref[-1]:] = i[idx_ref[-1]]
  _ref = 0
  for _i in range(idx_ref[0],idx_ref[-1]):
    if _i in idx_ref: continue
    idx_ref_before = idx_ref[_ref]
    idx_ref_after  = idx_ref[_ref+1]
    ref_before = i[idx_ref_before]
    ref_after  = i[idx_ref_after]
    weight_before = float(_i-idx_ref_before)/(idx_ref_after-idx_ref_before)
    weight_after  = 1-weight_before
    # normal reference for an on chi, the weighted average
    iref[_i]      = weight_before*ref_before + weight_after*ref_after
    if _i>=idx_ref_after-1: _ref += 1
    log.debug("For image %d : %d-%d"%(_i,idx_ref_before,idx_ref_after))
  # take care of the reference for the references ...
  if len(idx_ref) >  2:
    iref[idx_ref[0]] = i[idx_ref[1]]
    iref[idx_ref[-1]] = i[idx_ref[-2]]
    for _i in range(1,len(idx_ref)-1):
      idx_ref_before = idx_ref[_i-1]
      idx_ref_after  = idx_ref[_i+1]
      ref_before = i[idx_ref_before]
      ref_after  = i[idx_ref_after]
      weight_before = float(idx_ref[_i]-idx_ref_before)/(idx_ref_after-idx_ref_before)
      weight_after  = 1-weight_before
      # normal reference for an on chi, the weighted average
      iref[idx_ref[_i]]    = weight_before*ref_before + weight_after*ref_after
      log.debug("For reference image %d : %d-%d"%(idx_ref[_i],idx_ref_before,idx_ref_after))
  else:
    #print(idx_ref)
    #print(iref[idx_ref])
    iref[idx_ref]=i[idx_ref[0]]
    #print(iref[idx_ref])
  if useRatio:
    i /= iref
  else:
    i -= iref
  return i

def averageScanPoints(scan,data,errAbs=None,isRef=None,lpower=None,
    useRatio=False,funcForAveraging=np.nanmean,chi2_0_max='auto'):
  """ Average data for equivalent values in 'scan' array

      given scanpoints in 'scan' and corresponding data in 'data'
      average all data corresponding the exactly the same scanpoint.
      If the values in scan are coming from a readback, rounding might be
      necessary.

      Parameters
      ----------
      scan : array(N)
          array of scan points
      data : array(N,M)
          array of data to average, first axis correspond to scan index
      errAbs : None or array as data
          errbar for each data point. if None take the standard deviation 
          over images in given scan point
      isRef : None or array(N)
          if None no reference is subtracted. if array, True indicate that 
          a particular image is a reference one
      lpower : None or array(N)
          if not None, time resolved difference or ratio is normalized by it
      useRatio : bool
          use True if you want to calculate ratio ( I_{on}/I_{ref} ) instead
          of I_{on} - I_{off}
      funcForAveraging: function accepting axis=int keyword argument
          is usually np.nanmean or np.nanmedian.
      chi2_0_max = None, "auto" or float
          simple chi2_0 threshold filter. use trx.filters for more advanced
          ones. If auto, define max as 95% percentle. if None it is not applied

      Returns
      -------
      DataStorage instance with all info
"""
  args = dict( isRef = isRef, lpower = lpower, useRatio = useRatio )
  data = data.astype(np.float)
  average = np.mean(data,axis=0)
  median  = np.median(data,axis=0)

  if isRef is None: isRef = np.zeros( data.shape[0], dtype=bool )
  assert data.shape[0] == isRef.shape[0], \
    "Size mismatch, data is %d, isRef %d"%(data.shape[0],isRef.shape[0])

  # subtract reference only is there is at least one
  if isRef.sum()>0:
    # create a copy (subtractReferences works in place)
    diff_all = subtractReferences(data.copy(),np.argwhere(isRef),
               useRatio=useRatio)
    ref_average = funcForAveraging(data[isRef],axis=0)
  else:
    diff_all = data
    ref_average = np.zeros_like(average)

  # normalize signal for laser intensity if provided
  if lpower is not None:
    lpower = utils.reshapeToBroadcast(lpower,data)
    if useRatio is False:
      diff_all /= lpower
    else:
      diff_all = (diff_all-1)/lpower+1

  scan_pos = np.unique(scan)
  shape_out = [len(scan_pos),] + list(diff_all.shape[1:])
  diffs     = np.empty(shape_out)
  diff_err  = np.empty(shape_out)
  diffs_in_scan = []
  chi2_0 = []
  for i,t in enumerate(scan_pos):
    shot_idx = (scan == t)
    if shot_idx.sum() == 0:
      log.warn("No data to average for scan point %s"%str(t))

    # select data for the scan point
    diff_for_scan = diff_all[shot_idx]
    if errAbs is not None:
      noise  = np.nanmean(errAbs[shot_idx],axis = 0)
    else:
      noise = np.nanstd(diff_for_scan, axis = 0)

    # if it is the reference take only every second ...
    if np.all( shot_idx == isRef ):
      diff_for_scan = diff_for_scan[::2]

    diffs_in_scan.append( diff_for_scan )

    # calculate average
    diffs[i] = funcForAveraging(diff_for_scan,axis=0)

    # calculate chi2 of different repetitions
    chi2 = np.power( (diff_for_scan - diffs[i])/noise,2)
    # sum over all axis but first
    for _ in range(diff_for_scan.ndim-1):
      chi2 = np.nansum( chi2, axis=-1 )

    # store chi2_0
    chi2_0.append( chi2/diffs[i].size )

    # store error of mean
    diff_err[i] = noise/np.sqrt(shot_idx.sum())
  ret = dict(scan=scan_pos,diffs=diffs,err=diff_err,
        chi2_0=chi2_0,diffs_in_scan=diffs_in_scan,
        ref_average = ref_average, diffs_plus_ref=diffs+ref_average,
        average=average,median=median,args=args)
  ret = DataStorage(ret)
  if chi2_0_max is not None:
    ret = filters.chi2Filter(ret,threshold=chi2_0_max)
    ret = filters.applyFilters(ret)
  return ret


def calcTimeResolvedSignal(scan,data,err=None,reference="min",q=None,
    monitor=None,saveTxt=True,folder=os.curdir,**kw):
  """
    reference: can be 'min', 'max', a float|integer or an array of booleans
    data     : >= 2dim array (first index is image number)
    q        : is needed if monitor is a tuple|list
    saveTxt  : will save txt outputfiles (diff_av_*) in folder
    other keywords are passed to averageScanPoints
  """
  if reference == "min":
    isRef = (scan == scan.min())
  elif reference == "max":
    isRef = (scan == scan.max())
  elif isinstance(reference,(float,int)):
    isRef = (scan == reference)
  else:
    isRef = reference
  # normalize if needed
  if monitor is not None:
    if isinstance(monitor,(tuple,list)):
      assert q is not None, "q is None and can't work with q-range scaling"
      assert data.ndim == 2, "currently q-range scaling works with 2dim data"
      idx = (q>= monitor[0]) & (q<= monitor[1])
      monitor = np.nanmedian(data[:,idx],axis=1)
    monitor = utils.reshapeToBroadcast(monitor,data)
    data = data/monitor
    if err is not None: err  = err/monitor
  ret = averageScanPoints(scan,data,errAbs=err,isRef=isRef,**kw)
  if q is not None: ret["q"] = q
  return ret

def saveTxt(folder,data,delayToStr=True,basename='auto',info="",**kw):
  """ data must be a DataStorage instance """
  # folder ends usually with sample/run so use the last two subfolders
  # the abspath is needed in case we analyze the "./"
  folder = os.path.abspath(folder);
  if basename == 'auto':
      sep = os.path.sep
      basename = "_".join(folder.rstrip(sep).split(sep)[-2:]) + "_"
  q = data.q if "q" in data else np.arange(data.diffs.shape[-1])
  # save one file with all average diffs
  fname = os.path.join(folder,"%sdiff_av_matrix.txt" %basename)
  utils.saveTxt(fname,q,data.diffs,headerv=data.scan,**kw)
  fname = os.path.join(folder,"%sdiff_plus_ref_av_matrix.txt" %basename)
  utils.saveTxt(fname,q,data.diffs_plus_ref,headerv=data.scan,**kw)
  # save error bars in the matrix form
  fname = os.path.join(folder,"%sdiff_av_matrix_err.txt" % basename)
  utils.saveTxt(fname,q,data.err,headerv=data.scan,**kw)

  for iscan,scan in enumerate(data.scan):
    scan = utils.timeToStr(scan) if delayToStr else "%+10.5e" % scan

    # try retreiving info on chi2
    try:
      chi2_0  = data.chi2_0[iscan]
      info_delay = [ "# rep_num : chi2_0 , discarded by chi2masking ?", ]
      for irep,value in enumerate(chi2_0):
        info_delay.append( "# %d : %.3f" % (irep,value))
        if 'chi2' in data.masks: info_delay[-1] += " %s"%str(data.masks['chi2'][iscan][irep])
      info_delay = "\n".join(info_delay)
      if info != '': info_delay = "%s\n%s" % (info,info_delay)
    except AttributeError:
      info_delay = info

    # save one file per timedelay with average diff (and err)
    fname = os.path.join(folder,"%sdiff_av_%s.txt" %(basename,scan))
#    if 'mask' in data:
#      tosave = np.vstack( (data.diffs[iscan],data.err[iscan],
#               data.dataUnmasked[iscan],data.errUnmasked[iscan] ) )
#      columns = 'q diffmask errmask diffnomask errnomask'.split()
#    else:
    tosave = np.vstack( (data.diffs[iscan],data.err[iscan] ) )
    columns = 'q diff err'.split()
    utils.saveTxt(fname,q,tosave,info=info_delay,columns=columns)

    # save one file per timedelay with all diffs for given delay
    fname = os.path.join(folder,"%sdiffs_%s.txt" % (basename,scan))
    utils.saveTxt(fname,q,data.diffs_in_scan[iscan],info=info_delay,**kw)
