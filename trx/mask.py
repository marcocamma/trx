from __future__ import print_function
import sys
if sys.version_info.major == 2: input=raw_input
import logging
log = logging.getLogger(__name__)

import os
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

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

class MyMask(object):
  def __init__(self,img=None):
    self.comp = []
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
    mask = self.mask
    if (inverted): mask = ~mask
    i=fabio.edfimage.edfimage(mask.astype(np.uint8)); # edf does not support bool
    i.save(fname)

def snap(point,shape,snapRange=20):
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

def makeMaskGui(img,snapRange=60):
  """ snapRange controls border snapping (in pixels, use <= 0 to disable """
  mask = MyMask(img)
  ans='ok'
  while (ans != 'done'):
    plt.imshow(img)
    plt.clim(np.percentile(img,(2,98)))
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
  mask = np.zeros(shape,dtype=bool)
  mask[ :width  ,   :     ] = True
  mask[ -width: ,   :     ] = True
  mask[    :    , :width  ] = True
  mask[    :    , -width: ] = True
  return mask

def maskCenterLines(width,shape):
  mask = np.zeros(shape,dtype=bool)
  if isinstance(width,int): width = (width,width)
  c0 = int(shape[0]/2)
  c1 = int(shape[1]/2)
  w0 = int(width[0]/2)
  w1 = int(width[1]/2)
  mask[ c0-w0:c0+w0  ,       :      ] = True
  mask[      :       ,  c1-w1:c1+w1 ] = True
  return mask


def test(shape=(1000,2000)):
  mask = MyMask()
  mask.addCircle(400,300,250)
  mask.subtractCircle(400,300,150)
  mask.addRectangle(350,250,1500,700)
  plt.imshow( mask.getMask(shape) )
  return mask



if __name__ == "__main__":
  test()
  plt.show()
  ans=input("Enter to finish") 
