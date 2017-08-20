# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import numpy as np
np.seterr(all='ignore')

def findSlice(array,lims):
  start = np.ravel(np.argwhere(array>lims[0]))[0]
  stop  = np.ravel(np.argwhere(array<lims[1]))[-1]
  return slice(int(start),int(stop))

def approx(values,approx_values):
    """ returns array where every value is replaced by the closest in approx_values

        This funciton is useful for rebinning; careful, can be slow with many bins...

        Example:
        -------
        approx( np.arange(0,1,0.1), [0,0.3,0.7] )
           array([ 0. ,  0. ,  0.3,  0.3,  0.3,  0.7,  0.7,  0.7,  0.7,  0.7])

    """
    # make sure they are arrays
    values = np.asarray(values)
    approx_values = np.asarray(approx_values)
    # create outter difference
    diff = np.abs(values[:,np.newaxis] - approx_values)
    args = np.argmin(diff,axis=1)
    values = approx_values[args]
    #values = np.asarray( [ approx_values[np.argmin(np.abs(v-approx_values))] for v in values] )
    return values


def rebin(values,bins):
    """ returns array where every value is replaced by the closest in approx_values

        This funciton is useful for rebinning

        Example:
        -------
        approx( np.arange(0,1,0.1), [0,0.3,0.7] )
           array([ 0. ,  0. ,  0.3,  0.3,  0.3,  0.7,  0.7,  0.7,  0.7,  0.7])

    """
    # make sure they are arrays
    bins = np.asarray(bins)
    idx = np.digitize(values,bins)
    idx[idx > bins.shape[0]-1] = bins.shape[0]-1
    return (bins[idx]+bins[idx-1])/2

def reshapeToBroadcast(what,ref):
  """ expand the 1d array 'what' to allow broadbasting to match 
      multidimentional array 'ref'. The two arrays have to same the same 
      dimensions along the first axis
  """
  if what.shape == ref.shape: return what
  assert what.shape[0] == ref.shape[0],\
    "automatic reshaping requires same first dimention"
  shape  = [ref.shape[0],] + [1,]*(ref.ndim-1)
  return what.reshape(shape)

def removeBackground(x,data,xlims=None,max_iter=100,background_regions=[],**kw):
  from dualtree import dualtree
  if data.ndim == 1: data = data[np.newaxis,:]
  if xlims is not None:
    idx = findSlice(x,xlims)
    x = x[idx]
    data = data[:,idx].copy()
  else:
    data = data.copy(); # create local copy
  # has to be a list of lists ..
  if background_regions != [] and isinstance(background_regions[0],numbers.Real):
    background_regions = [background_regions,]
  background_regions = [findSlice(x,brange) for brange in background_regions]
  for i in range(len(data)):
    data[i] = data[i] - dualtree.baseline(data[i],max_iter=max_iter,
                        background_regions=background_regions,**kw)
  return x,np.squeeze(data)
