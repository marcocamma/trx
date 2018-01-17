import numpy as np
import matplotlib.pyplot as plt
import os
import fabio
from .mask import interpretMasks

from datastorage import DataStorage

def _calc_R(x,y, xc, yc):
  """ calculate the distance of each 2D points from the center (xc, yc) """
  return np.sqrt((x-xc)**2 + (y-yc)**2)

def _chi2(c, x, y):
  """ calculate the algebraic distance between the data points and the mean
      circle centered at c=(xc, yc) """
  Ri = _calc_R(x, y, *c)
  return Ri - Ri.mean()

def leastsq_circle(x,y):
  from scipy import optimize
  # coordinates of the barycenter
  center_estimate = np.nanmean(x), np.nanmean(y)
  center, ier = optimize.leastsq(_chi2, center_estimate, args=(x,y))
  xc, yc = center
  Ri       = _calc_R(x, y, *center)
  R        = Ri.mean()
  residu   = np.sum((Ri - R)**2)
  return xc, yc, R

def find_center_using_circle(img,X=None,Y=None,mask=None,percentile=(90,99),\
    plot=False):
    """ Find beam center position fitting a ring (usually liquid peak)

        Parameters
        ==========
        img: array or string
          image to use, if string, reads it with fabio
        X,Y: None or arrays
          position of center of pixels, if given they will have to have
          same shape as img, if None, they will be created with meshgrid
        mask: bool array
          Pixels with True are masked out in calculations
        percentile: tuple
          range of intensity to use (in percentile values)
    """
    try:
        f = fabio.open(img)
        img = f.data
        del f
    except:
        pass
    # make sure that is float (and create a local copy) 
    img = img.astype(float)
    if mask is not None and not isinstance(mask,np.ndarray):
        mask = interpretMasks(mask,img.shape)
    if mask is not None:
        img[mask] = np.nan
    zmin,zmax = np.nanpercentile(img.ravel(),percentile[:2])
    shape     = img.shape
    idx       = (img>=zmin) & (img<=zmax)
    if X is None or Y is None:
        _use_imshow = True
        X,Y = np.meshgrid ( range(shape[1]),range(shape[0]) )
    else:
        _use_imshow = False
    xfit = X[idx].ravel()
    yfit = Y[idx].ravel()
    xc,yc,R = leastsq_circle(xfit,yfit)
    if plot:
        ax = plt.gca()
        cmin = np.nanmin(img)
        if _use_imshow:
            plt.imshow(img,clim=(cmin,zmin),cmap=plt.cm.gray)
            # RGBA
            img = np.zeros( (img.shape[0],img.shape[1],4) )
            img[idx,0]=1
            img[idx,3]=1
            plt.imshow(img)
        else:
            plt.pcolormesh(X,Y,img,cmap=plt.cm.gray,vmin=cmin,vmax=zmin)
            img = np.ma.masked_array(img,idx)
            plt.pcolormesh(X,Y,img)
        circle = plt.Circle( (xc,yc),radius=R,lw=5,color='green',fill=False)
        ax.add_artist(circle)
        plt.plot(xc,yc,"o",color="green",markersize=5)
    return DataStorage( xc=xc, yc=yc, R=R, x = xfit, y = yfit )



def find_center_using_clicks(img,X=None,Y=None,clim='auto'):
    """ Find beam center position fitting points (selected by clicks) on a ring

        Parameters
        ==========
        img: array or string
          image to use, if string, reads it with fabio
        X,Y: None or arrays
          position of center of pixels, if given they will have to have
          same shape as img, if None, they will be created with meshgrid
        clim: tuple|'auto'
          for color scale
    """
    try:
        f = fabio.open(img)
        img = f.data
        del f
    except:
        pass
    # make sure that is float (and create a local copy) 
    img = img.astype(float)
    if clim == 'auto':
      clim = np.nanpercentile(img.ravel(),(90,100))
    shape     = img.shape
    if X is None or Y is None:
        X,Y = np.meshgrid ( range(shape[1]),range(shape[0]) )
    ans = 'ok'
    while (ans != 'done'):
        ax = plt.gca()
        ax.pcolormesh(X,Y,img,cmap=plt.cm.gray,vmin=clim[0],vmax=clim[1])
        print("Select points on a ring, middle-click to stop")
        coords = plt.ginput(-1)
        coords = np.asarray(coords).T
        xc,yc,R = leastsq_circle( coords[0],coords[1] )
        circle = plt.Circle( (xc,yc),radius=R,lw=5,color='green',fill=False)
        ax.add_artist(circle)
        print("Found circle at (%.4f,%.4f), R = %.4f"%(xc,yc,R))
        ax.plot(xc,yc,"o",color="green",markersize=5)
        plt.draw()
        ans = input("type 'done' to finish, anything else to try again")
    return DataStorage( xc=xc, yc=yc, R=R )


