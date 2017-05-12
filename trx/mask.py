# -*- coding: utf-8 -*-
""" 
    Module to handle masks: it defines:

      - Mask: a mask object with command line methods: addPolygon, etc
      - makeMaskGui: a GUI based way of creating masks
      - maskBorder: to mask the borders of an array
      - maskCenterLines: to mask the central lines (good for multi-panel det)
      - interpretMask: interpret mask element (filename,y>500,array)
      - interpretMasks: add up list of mask elements
"""
from __future__ import print_function,division,absolute_import
import sys
if sys.version_info.major == 2: input=raw_input
import logging
log = logging.getLogger(__name__)

import os
import re
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import fabio

def read(fname):
  """ read data from file using fabio """
  f = fabio.open(fname)
  data = f.data
  del f; # close file
  return data


maskComponent = collections.namedtuple('maskComponent',['operation','geometry','vertices'])

def _rectangleToMask(X,Y,vertices):
    ( (x1,y1), (x2,y2) ) = vertices
    if x1>x2: x1,x2=x2,x1
    if y1>y2: y1,y2=y2,y1
    return (X>x1) & (X<x2) & ( Y>y1) & (Y<y2)

def _circleToMask(X,Y,vertices):
    c,p = vertices
    r = np.sqrt( (p[0]-c[0])**2 + (p[1]-c[1])**2 )
    d = np.sqrt((X-c[0])**2+(Y-c[1])**2)
    return d<r

def _polygonToMask(X,Y,vertices):
    points = np.vstack((X.flatten(),Y.flatten())).T
    path = Path(vertices)
    grid = path.contains_points(points)
    return grid.reshape(X.shape)

class Mask(object):
  """ class for making masks. True are pixels masked OUT.

      This class provides methods for adding/subtracting components in the
      for of rectangles, circles, polygons. Each addition/subtraction is 
      stored as 'operations' that are then applied on request.
      
      Parameters
      ----------
      img: {array,filename,shape of array}
          img to use for calculation. Since only the shape is used a shape-like
          tuple can be provided


  """
  def __init__(self,img=None):
    self.comp = []
    if img is not None:
      self.shape = img.shape
    else:
      self.shape = None
    if isinstance(img,str): img = read(img)
    # if img is not array at this point assume it is a shape-tuple
    if not isinstance(img,np.ndarray): img = np.zeros( img, dtype=bool )
    self.img  = img
    self.mask = None
    self._cache = None

  def _define_component(self,operation,geometry,*vertices):
    #print("define_comp",type(vertices),vertices)
    if geometry == 'circle' and len(vertices) == 3:
      xcen,ycen,radius = vertices
      vertices = ( (xcen,ycen), (xcen+radius,ycen) )
    if geometry == 'rectangle' and len(vertices) == 4:
      vertices = ( (vertices[0],vertices[1]),(vertices[2],vertices[3]) )
    # make sure vertices tuples
    if isinstance(vertices,list):
      vertices = [ (v[0],v[1]) for v in vertices ]
      vertices = tuple(vertices)
    a = dict( vertices = None )
    self.comp.append( maskComponent(operation=operation,vertices=vertices,geometry=geometry) )
    
  
  def addCircle(self,*vertices): self._define_component( 'add', 'circle', *vertices )
  def subtractCircle(self,*vertices): self._define_component( 'subtract', 'circle', *vertices )

  def addRectangle(self,*vertices): self._define_component( 'add','rectangle', *vertices)
  def subtractRectangle(self,*vertices): self._define_component( 'subtract','rectangle',*vertices)
  
 
  def addPolygon(self,*vertices): self._define_component( 'add','polygon',*vertices)
  def subtractPolygon(self,*vertices): self._define_component( 'subtract','polygon',*vertices)


  def getMask(self,shape=None):
    if shape is None and self.img is not None: shape = self.img.shape
    if shape is None and self.img is     None: shape = self._cache['shape']
    
    if self._cache is None: self._cache = dict( shape = shape )
    # reset cache if shape does not match
    if shape != self._cache['shape']:
      self._cache = dict( shape = shape )
    X,Y = np.meshgrid ( range(shape[1]),range(shape[0]) )
    for component in self.comp:
      if component not in self._cache:
        if component.geometry == 'circle':
          mask = _circleToMask( X,Y,component.vertices )
        elif component.geometry == 'rectangle':
          mask = _rectangleToMask( X,Y,component.vertices )
        elif component.geometry == 'polygon':
          mask = _polygonToMask( X,Y,component.vertices )
        else:
          raise ValueError("Mask type %s not recongnized"%component.geometry)
        self._cache[component] = mask
    mask = np.zeros(shape,dtype=np.bool)
    for comp in self.comp:
      m = self._cache[ comp ]
      if (comp.operation == "add"):
        mask[m] = True
      else:
        mask[m] = False
    self.mask = mask
    return mask

  def getMatplotlibMask(self,shape=None):
    mask = self.getMask(shape=shape)
    # convert
    mpl_mask = np.zeros( (mask.shape[0],mask.shape[1],4) )
    mpl_mask[:,:,:3] = 0.5; # gray color
    mpl_mask[:,:,3] = mask/2; # give some transparency
    return mpl_mask

  def save(self,fname,inverted=False):
    import fabio
    if self.mask is None: self.getMask()
    mask = self.mask
    if (inverted): mask = ~mask
    if os.path.splitext(fname)[1] == ".npy":
      np.save(fname,mask)
    else:
      i=fabio.edfimage.edfimage(mask.astype(np.uint8)); # edf does not support bool
      i.save(fname)

def snap(point,shape,snapRange=20):
  """ snap 'point' if within 'snapRange' from the border defined by 'shape' """
  snapped = list(point)
  if snapped[0] < snapRange: snapped[0] = 0
  if snapped[0] > shape[1]-snapRange: snapped[0] = shape[1]
  if snapped[1] < snapRange: snapped[1] = 0
  if snapped[1] > shape[0]-snapRange: snapped[1] = shape[0]
  return tuple(snapped)

def getPoints(N=1,shape=(100,100),snapRange=0):
  if N<1: print('Right click cancels last point, middle click ends the polygon')
  c = plt.ginput(N)
  c = [ snap(point,shape,snapRange=snapRange) for point in c ]
  if len(c) == 1: c = c[0]
  return c

def makeMaskGui(img,snapRange=60,clim='auto'):
  """ interactive, click based approach do define a mask.
      
      Parameters
      ----------
      snapRange : int
          controls border snapping (in pixels) use <= 0 to disable;
      clim: 'auto' or list(min,max)
          controls color scale os image, if 'auto' uses 2%-98% percentile

      Returns
      -------
      Mask
          instance of the Mask class that allows to modify or save the mask

  """
  if isinstance(img,str): img = read(img)
  mask = Mask(img)
  if clim == "auto": clim = np.percentile(img,(2,98))
  ans='ok'
  while (ans != 'done'):
    plt.imshow(img)
    plt.clim(clim)
    plt.imshow(mask.getMatplotlibMask())
    plt.pause(0.01)
    ans = input("What's next p/P/c/C/r/R/done? (capitals = subtract)")
    if ans == "c":
      print("Adding circle, click on center then another point to define radius")
      vertices = getPoints(N=2,shape=img.shape,snapRange=snapRange)
      mask.addCircle(*vertices)
    if ans == "C":
      print("Subtracting circle, click on center then another point to define radius")
      vertices = getPoints(N=2,shape=img.shape,snapRange=snapRange)
      mask.subtractCircle(*vertices)
    if ans == "r":
      print("Adding rectangle, click on one corner and then on the opposite one")
      vertices = getPoints(N=2,shape=img.shape,snapRange=snapRange)
      mask.addRectangle(*vertices)
    if ans == "R":
      print("Subtracting rectangle, click on one corner and then on the opposite one")
      vertices = getPoints(N=2,shape=img.shape,snapRange=snapRange)
      mask.subtractRectangle(*vertices)
    if ans == 'p':
      print("Adding polygon")
      vertices = getPoints(N=-1,shape=img.shape,snapRange=snapRange)
      mask.addPolygon(*vertices)
    if ans == 'P':
      print("Subtracting polygon")
      vertices = getPoints(N=-1,shape=img.shape,snapRange=snapRange)
      mask.subtractPolygon(*vertices)

    plt.imshow(mask.getMatplotlibMask())
    plt.pause(0.01)
  fname = input("Enter a valid filename (ext .edf or .npy) if you want to save the mask (empty otherwise)")
  try:
    if fname != '':
      ext = os.path.splitext(fname)[1]
      if ext == '.edf':
        mask.save(fname)
      elif ext == '.npy':
        np.save(fname,mask.getMask())
  except Exception as e:
    log.error("Error in saving mask")
    log.error(e)
  finally:
    return mask
    
def maskBorder(width,shape):
  """ mask the border of an array for a given width
      
      Parameters
      ----------
      width : int
          the width of the region to mask (>0)

      Returns
      -------
      boolean (False/True) array
          True are the pixels masked out
  """
  assert isinstance(width,int), "width has to be integer"
  assert width>0, "width has to be positive"
  mask = np.zeros(shape,dtype=bool)
  mask[ :width  ,   :     ] = True
  mask[ -width: ,   :     ] = True
  mask[    :    , :width  ] = True
  mask[    :    , -width: ] = True
  return mask

def maskCenterLines(width,shape):
  """ mask a cross going trough the center of the array for a given width
      
      Parameters
      ----------
      width : int
          the width of the region to mask (>0)

      Returns
      -------
      boolean (False/True) array
          True are the pixels masked out
  """
  assert isinstance(width,int), "width has to be integer"
  assert width>0, "width has to be positive"
  mask = np.zeros(shape,dtype=bool)
  if isinstance(width,int): width = (width,width)
  c0 = int(shape[0]/2)
  c1 = int(shape[1]/2)
  w0 = int(width[0]/2)
  w1 = int(width[1]/2)
  mask[ c0-w0:c0+w0  ,       :      ] = True
  mask[      :       ,  c1-w1:c1+w1 ] = True
  return mask

g_mask_str = re.compile("(\w)\s*(<|>)\s*(\d+)")

def interpretMask(mask,shape=None):
  """ Interpret 'mask' as a mask

      Parameters
      ----------
      mask : filename or array or string like y>500

      shape : array or tuple
          needed to interpret y>500

      Returns
      -------
      boolean (False/True) array
          True are the pixels masked out
  """
  maskout = None
  ## simplest case, an existing file
  if isinstance(mask,str) and os.path.isfile(mask):
    maskout = read(mask).astype(np.bool)
  ## mask string
  elif isinstance(mask,str) and not os.path.isfile(mask):
    if isinstance(shape,np.ndarray) : shape = shape.shape
    err_msg = ValueError("The string '%s' could not be interpreted as simple\
              mask; it should be something like x>10"%mask)
    assert shape is not None, "_interpretMask needs a shape to interpret a string"
    # interpret string
    maskout = np.zeros(shape,dtype=bool)
    match = g_mask_str.match(mask)
    if match is None: raise err_msg
    (axis,sign,lim) = match.groups()
    if axis not in ("x","y"): raise err_msg
    if sign not in (">","<"): raise err_msg
    lim = int(lim)
    idx = slice(lim,None) if sign == ">" else slice(None,lim)
    if axis == 'y':
      maskout[idx,:] = True
    else:
      maskout[:,idx] = True
  elif isinstance(mask,np.ndarray):
    maskout = mask.astype(np.bool)
  elif mask is None:
    assert shape is not None, "_interpretMask needs a shape to interpret a string"
    maskout = np.zeros(shape,dtype=bool)
  else:
    maskout = None
    raise ValueError("Could not interpret %s as mask input"%mask)
  
  if shape is not None and maskout.shape != shape:
    raise ValueError("The mask shape %s does not match the shape given as\
      argument %s"%(maskout.shape,shape))
  return maskout

def interpretMasks(masks,shape=None):
  """ Interpret a single or a list of mask elements

      Every element can be an array, a filename to read, a 'mask string'
      (y>500). 

      Parameters
      ----------
      masks : a 'mask element' or a list of mask elements

      shape : array or tuple
          needed to interpret y>500

      Returns
      -------
      boolean (False/True) array
          True are the pixels masked out
 
  """
  if isinstance(masks,np.ndarray): return masks.astype(bool)
  # make iterable
  if not isinstance( masks, (list,tuple,np.ndarray) ): masks = (masks,)
  masks = [interpretMask(mask,shape) for mask in masks]
  # put them all together
  mask = masks[0]
  for m in masks[1:]:
    mask = np.logical_or(mask,m)
  return mask


def test(shape=(1000,2000)):
  """ Make a simple mask programmatically """
  mask = Mask()
  mask.addCircle(400,300,250)
  mask.subtractCircle(400,300,150)
  mask.addRectangle(350,250,1500,700)
  plt.imshow( mask.getMask(shape) )
  return mask



if __name__ == "__main__":
  test()
  plt.show()
  ans=input("Enter to finish") 
