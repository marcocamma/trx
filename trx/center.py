"""
Module implementing different ways to find beamcenter
"""
import numpy as np
import matplotlib.pyplot as plt
import os


import logging
log = logging.getLogger(__name__)

from scipy import optimize
from skimage import feature

import fabio
from .mask import interpretMasks
from .utils import find_hist_ranges
from . import utils
from . import azav
from datastorage import DataStorage

def _prepare_img(img):
    """ if argument is string, read it with fabio; it converts the image
    to float"""
    if isinstance(img,str):
        if os.path.isfile(img):
            f = fabio.open(img)
            img = f.data
            del f
        else:
            raise FileNotFoundError("File",img,"does not exist")
    else:
        pass
    # make sure that is float (and create a local copy)
    img = img.astype(float)
    return img

def _prepare_mask(mask,img):
    """ interpret/prepare mask """
    if mask is None:
        mask = np.zeros_like(img,dtype=bool)
    else:
        if isinstance(mask,np.ndarray):
            mask = mask.astype(bool)
        else:
            mask = interpretMasks(mask,img.shape)
    return mask

def _calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def _chi2(c, x, y):
    """ calculate the algebraic distance between the data points and the mean
        circle centered at c=(xc, yc) """
    Ri = _calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    """ Utility funciton to fit a circle given x,y positions of points """
    # coordinates of the baricenter
    center_estimate = np.nanmean(x), np.nanmean(y)
    center, ier = optimize.leastsq(_chi2, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = _calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return DataStorage(center=np.asarray((xc,yc)),radius=R)


def fit_ellipse(x,y):
    """ Utility funciton to fit an ellipse given x,y positions of points """
    # from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    A = V[:,n]

    # center
    b,c,d,f,g,a = A[1]/2, A[2], A[3]/2, A[4]/2, A[5], A[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    center = np.array([x0,y0])


    # angle of rotation
    b,c,d,f,g,a = A[1]/2, A[2], A[3]/2, A[4]/2, A[5], A[0]
    angle = np.rad2deg(0.5*np.arctan(2*b/(a-c)))


    # axis
    b,c,d,f,g,a = A[1]/2, A[2], A[3]/2, A[4]/2, A[5], A[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    axis = np.array([res1, res2])
    return DataStorage(center=center,axis=axis,angle=angle,radius=np.mean(axis))

def find_center_liquid_peak(img, X=None, Y=None, mask=None,
        percentile=(90,99), plot=False):
    """ Find beam center position fitting a ring (usually liquid peak)

        Parameters
        ==========
        img: array or string
          image to use, if string, reads it with fabio

        X,Y: None or arrays
          position of center of pixels, if given they will have to have
          same shape as img, if None, they will be created with meshgrid

        mask: boolean mask or something trx.mask.interpretMasks can understand
          True are pixels to mask out

        percentile: tuple
          range of intensity to use (in percentile values)
    """

    # interpret inputs
    img = _prepare_img(img)
    mask = _prepare_mask(mask,img)

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
    fit = leastsq_circle(xfit,yfit)
    xc = fit.center[0]
    yc = fit.center[1]
    R = fit.radius
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

    # interpret inputs
    img = _prepare_img(img)

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




def find_center_using_rings(img, sigma=3, high_threshold=20, low_threshold=10,
        mask=None, center=None, nrings=10, min_dist=100, max_peak_width=60,
        use_ellipse=False, plot=True, clim = "auto", verbose=False,
        reprocess=False):
    """ Find beam center position finding powder rings and fitting them

        This functions tries to automatically find the power peaks.
        It uses several steps:
        1. finds pixels belonging to a sharp peak by using skimage.canny
        2. given an initial guess of the center, it build a distance
           histogram
        3. finds peaks in histogram that should represent the edges of
           of a powder ring
        4. fit pixels belowning to each peak (i.e. pixels within a ring)
           to a circle/ellipse
        5. does some sanity check before each fit (minimum number of
           pixels, width of the peak, etc) and after (center cannot move
           too much
        6. uses previously find centers (median value) to build distance
           histogram of pixels found by the canny filter

        Parameters
        ==========

        img: array or string
          image to use, if string, reads it with fabio

        sigma: float
          used by canny filter, see skimage.feature.canny doc

        {low|high}_threshold: float
          used by canny filter, see skimage.feature.canny. In general
          low=10, high=20 seems to work very well for wear and strong intensity
          images

        mask: boolean mask or something trx.mask.interpretMasks can understand
          True are pixels to mask out

        center: None or tuple
          if tuple, use it as first guess of center (has to be good to few
          tens of pixels)
          if None, it proposes a "click" method

        nrings: int
          number of rings to look for, can be high, if less peaks are found it
          will not bug out

        min_dist: float
          mimum distance to look for peaks, to avoid possible high intensity
          close to beam or beamstop

        max_peak_width: float
          do not even try to fit peaks that are broader than max_peak_width

        use_ellipse: bool
          if True fits with ellipse, else uses circle

        plot: bool
          if True plots rings, histograms and 2d integration (quite useful)

        clim: "auto" or tuple
          color scale to use for plots, if auto uses 20%,85% percentile

        verbose: bool
          increases verbosity (possibly too much !)

        reprocess: bool
          if True, at the end of the fits of all rings, it reruns with current
          best estimate (median of centers) to find other peaks that could not
          be intentified with initial guess
    """

    # interpret inputs
    img = _prepare_img(img)
    mask = _prepare_mask(mask,img)

    if isinstance(clim,str) and clim =="auto":
        if mask is None:
            clim = np.percentile(img,(10,95))
        else:
            clim = np.percentile(img[~mask],(10,95))

    if center is None:
        plt.figure("Pick center")
        plt.imshow(img,clim=clim)
        print("Click approximate center")
        center = plt.ginput(1)[0]

    # use skimage canny filter
    # note: mask is "inverted" to be consistent with trx convention: True
    # are masked out

    edges = feature.canny(img, sigma,low_threshold=low_threshold,
            high_threshold=high_threshold,mask=~mask)
    points = np.array(np.nonzero(edges)).T

    if points.shape[0] == 0:
        raise ValueError("Could not find any points, try changing the threshold or\
                the initial center")
    else:
        print("Found %d points"%points.shape[0])

    # swap x,y
    points = points[:,::-1]

    image = np.zeros_like(img)
    if plot:
        plt.figure("fit ring")
        plt.imshow(img,clim=clim,cmap=plt.cm.gray_r)
    colors = plt.rcParams['axes.prop_cycle']
    storage_fit = []
    last_n_peaks = 0
    for i,color in zip(range(nrings),colors):

        ## find points in a given circle based on histogam of distances ...

        # dist is calculate here because we can use previous cycle to
        # have improve peaks/beackground separation
        dist = np.linalg.norm(points-center,axis=1).ravel()

        if plot:
            plt.figure("hist")
            plt.hist(dist,1000,histtype='step',**color,label="ring %d"%(i+1))


        ## next is how to find the regions of the historam that should
        # represent peaks ...

        # we can start by some smoothing

        dist_hist,bins = np.histogram(dist,bins=np.arange(min_dist,dist.max()))
        bins_center = (bins[1:] + bins[:-1])/2
        N=sigma*2
        # use triangular kernel
        kernel = np.concatenate( (np.arange(int(N/2)), N/2-np.arange(int(N/2)+1)) )
        # normalize it
        kernel = kernel/(N**2)/4
        dist_hist_smooth = np.convolve( dist_hist, kernel,mode='same' )


        if plot:
            temp = dist_hist_smooth/dist_hist_smooth.max()*dist_hist.max()
            plt.plot(bins_center,temp,'--',**color)

        peaks_ranges = find_hist_ranges(dist_hist_smooth,x=bins_center,max_frac=0.1)
        n_peaks = peaks_ranges.shape[0]

        if verbose:
            peaks_ranges_str = map(str,peaks_ranges)
            peaks_ranges_str = ",".join(peaks_ranges_str)
            print("Iteration %d, found %d peaks, ranges %s"%
                    (i,n_peaks,peaks_ranges_str))

        if i >= n_peaks:
            print("asked for peaks than found, stopping")
            break

        idx = (dist>peaks_ranges[i,0]) & (dist<peaks_ranges[i,1])
#        log.debug("dist_range",dist_range,idx.sum(),idx.shape)


        # sanity check

        if points[idx].shape[0] == 0:
            print("No point for circle",i)
            continue

        if points[idx].shape[0] < 20:
            print("Too few points to try fit, skipping to next circle")
            continue

        peak_width = peaks_ranges[i][1]-peaks_ranges[i][0]
        if peak_width > max_peak_width:
            print("Peak %d seems too large (%.0f pixels), skipping"%
                    (i,peak_width) )
            continue
        else:
            if verbose:
                print("Peak %d width %.0f pixels"%
                    (i,peak_width) )



        ## Do fit
        try:
            if use_ellipse:
                fit = fit_ellipse(points[idx,0],points[idx,1])
            else:
                fit = leastsq_circle(points[idx,0],points[idx,1])
        except (TypeError,np.linalg.LinAlgError):
            print("Fit failed for peak",i)
            continue


        # prevent outlayers to messup next circle
        is_ok = (n_peaks >= last_n_peaks-2) & \
                (np.linalg.norm(fit.center-center) < 50)

        if not is_ok:
            continue

        center = fit.center #model_robust.params[0],model_robust.params[1]
        last_n_peaks = n_peaks
        storage_fit.append(fit)


        ## prepare out
        if use_ellipse:
            out_string = "ring %s"%(i+1)
            out_string += " center: %.3f %.3f"%tuple(fit.center)
            out_string += " axis : %.3f %.3f"%tuple(fit.axis)
            out_string += " angle : %+.1f"%fit.angle
        else:
            out_string = "ring %s"%(i+1)
            out_string += " center: %.3f %.3f"%tuple(fit.center)
            out_string += " radius : %.3f" % fit.radius
        print(out_string)

        if plot:
            plt.figure("fit ring")
            #plt.imshow(image)
            plt.plot(points[idx, 0], points[idx, 1], 'b.', markersize=1,**color)
            plt.plot(center[0],center[1],".",markersize=20,**color)
            circle = plt.Circle(fit.center,radius=fit.radius,**color,
                    fill=False)
            ax = plt.gca()
            ax.add_patch(circle)
            plt.pause(0.01)

    # package output
    out = DataStorage()
    for key in storage_fit[0].keys():
        out[key] = np.asarray( [f[key] for f in storage_fit] )
        out["%s_median"%key] = np.median(out[key],axis=0)
        out["%s_rms"%key] = np.std(out[key],axis=0)

    if plot:
        ai = azav.getAI(xcen=out["center_median"][0],
                ycen=out["center_median"][1],pixel=1e-3,
                distance=0.2)
        plt.figure("2D integration")
        x,y,i2d=azav.do2d(ai,img,mask=mask, unit="r_mm")
        vmin,vmax = np.percentile(i2d,(20,90))
        plt.pcolormesh(x,y,i2d,vmin=vmin,vmax=vmax)



    if reprocess:
        plt.close("all")
        return find_center_using_rings(img, sigma=sigma,
                high_threshold=high_threshold, low_threshold=low_threshold,
                mask=mask, plot=plot, center=out["center_median"],
                nrings=nrings, clim=clim, min_dist=min_dist,
                use_ellipse=use_ellipse, verbose=verbose, reprocess=False)


    return out


def test_find_center_using_rings(verbose=False):
    """ Run with some images the find_center_using_rings function"""
    folder =  os.path.join(os.path.dirname(__file__) ,
            "data/calibrant_examples")

    if not os.path.isdir(folder):
        raise FileNotFoundError("Folder %s does not exists"%folder)

    def dofile(fname,**kw):
        fname = os.path.join(folder,fname)
        ret=find_center_using_rings(fname, plot=True, verbose=verbose,
                reprocess=True, nrings=20, use_ellipse=True,**kw)
        input("ok to continue to next file, (will close all plots)")
        plt.close("all")
        return ret


    r1=dofile("al2o3_capillary_run03_0002.edf", center=(950,950), sigma=2)
    r2=dofile("calibrants_al2o3_capillary_run03.edf", center=(960,970))
    r3=dofile("lab6_run03_0017.edf", center=(960,970))
    r4=dofile("rbmnfe05_run20_0098.edf", center=(960,1400))
    r5=dofile("run02_0002_rayonix.edf", center=(960,1170))
    return (r1,r2,r3,r4,r5)
