# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import
import collections
import itertools
import numpy as np
from numpy import sin,cos

class Triclinic(object):
  def __init__(self,a=1,b=1,c=1,alpha=90,beta=90,gamma=90):
    self.a = a
    self.b = b
    self.c = c
    alpha  = alpha*np.pi/180
    beta   = beta*np.pi/180
    gamma  = gamma*np.pi/180
    self.alpha = alpha
    self.beta  = beta 
    self.gamma = gamma

    self._s11 = b**2 * c**2 * sin(alpha)**2
    self._s22 = a**2 * c**2 * sin(beta)**2
    self._s33 = a**2 * b**2 * sin(gamma)**2
    self._s12 = a*b*c**2*(cos(alpha) * cos(beta) - cos(gamma))
    self._s23 = a**2*b*c*(cos(beta) * cos(gamma) - cos(alpha))
    self._s13 = a*b**2*c*(cos(gamma) * cos(alpha) - cos(beta))
    self.V    = (a*b*c)*np.sqrt(1-cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma))

  def __call__(self,h,k,l):  return self.q(h,k,l)

  def d(self,h,k,l):
    temp = self._s11*h**2 + \
           self._s22*k**2 + \
           self._s33*l**2 + \
           2*self._s12*h*k+ \
           2*self._s23*k*l+ \
           2*self._s13*h*l
    d    = self.V/np.sqrt(temp)
    return d

  def Q(self,h,k,l):
    return 2*np.pi/self.d(h,k,l)

  def reflection_list(self,maxQ=3,lim=10):
    ret=dict()
    # prepare hkl
    i = range(-lim,lim+1)
    prod = itertools.product( i,i,i )
    hkl = np.asarray( list( itertools.product( i,i,i ) )  )
    h,k,l = hkl.T
    q = self.Q(h,k,l)

    idx = q<maxQ;
    q = q[idx]
    hkl = hkl[idx]
    q = np.round(q,12)
    qunique = np.unique(q)
    ret = []
    for qi in qunique:
      reflec = hkl[ q == qi ]
      ret.append( (qi,tuple(np.abs(reflec)[0]),len(reflec),reflec) )
    return qunique,ret
   
#    for h in range(-lim,lim+1):
#      for j in range(-lim,lim+1):


class Orthorombic(Triclinic):
  def __init__(self,a=1,b=1,c=1):
    Triclinic.__init__(self,a=a,b=b,c=c,alpha=90,beta=90,gamma=90)

class Cubic(Orthorombic):
  def __init__(self,a=1):
    Orthorombic.__init__(self,a=a,b=a,c=a)


class Monoclinic(object):
  def __init__(self,a=1,b=1,c=1,beta=90.):
    Triclinic.__init__(self,a=a,b=b,c=c,alpha=90,beta=beta,gamma=90)


def plotReflections(cell_instance,ax=None,line_kw=dict(),text_kw=dict()):
  import matplotlib.pyplot as plt
  from matplotlib import lines
  import matplotlib.transforms as transforms
  _,refl_info = cell_instance.reflection_list()
  if ax is None: ax = plt.gca()

  # the x coords of this transformation are data, and the
  # y coord are axes
  trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
  txt_kw = dict( horizontalalignment='center', rotation=45)
  txt_kw.update(**text_kw)
  for reflection in refl_info[1:]:
    q,hkl,n,_ = reflection
    line = lines.Line2D( [q,q],[1,1.1],transform=trans,**line_kw)
    line.set_clip_on(False)
    ax.add_line(line)
    ax.text(q,1.15,str(hkl),transform=trans,**txt_kw)

ti3o5_lambda = Triclinic(a = 9.83776, b = 3.78674, c = 9.97069, beta = 91.2567)
ti3o5_beta   = Triclinic(a = 9.7382 , b = 3.8005 , c = 9.4333 , beta = 91.496)
#ti3o5_beta   = Monoclinic(a = 9.7382 , b = 3.8005 , c = 9.4333 , beta = 91.496)
ti3o5_alpha  = Triclinic(a = 9.8372,  b = 3.7921,  c = 9.9717)
ti3o5_alpha1  = Orthorombic(a = 9.8372,  b = 3.7921,  c = 9.9717)

si = Cubic(a=5.431020504)
