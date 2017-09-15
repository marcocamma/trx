# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import numpy as np
np.seterr(all='ignore')
from .string import timeToStr

try:
  import matplotlib.pyplot as plt
except ImportError:
  log.warn("Can't import matplotlib !")

def plotdata(data,x=None,plotAverage=True,showTrend=True,title=None,\
      clim='auto',fig=None):
    """ Plot 2D array to look for trends/stability or outlier

        It usually works best with normalized data

        Parameters
        ----------
        data: dict-like object or 2D array.
          if dict-like it has to have keys
          'q' used for one axis
          'data' for 2D array[numimg,nQ]
        x: array or None
          if given it has to have a length of numimg
          if None will be 0,1,2,3,...numimg and interpreted
          as image number
        plotAverage: bool
          if True, plot average
        showTrend: bool
          if True, show 2D plot (most common use)
        clim: 'auto' or tuple
          color scale to use, is 'auto' 1.5%,98.5% percentiles are used
        fig:
          matplotlib fig instance to use, if None creates a new one
        title: str
          title to use for plot (if None and data['folder'] exists
          the value is used as title
          
    """
    if isinstance(data,np.ndarray):
      q = np.arange( data.shape[-1] )
    else:
      if title is None and 'folder' in data: title = data["folder"]
      q = data["q"]
      data = data["data"]
    if not (plotAverage or showTrend): return

    if x is None: x = np.arange(data.shape[0])
    if clim == 'auto': clim = np.nanpercentile(data,(1.5,98.5))
    one_plot = showTrend or plotAverage
    two_plot = showTrend and plotAverage

    if one_plot and not two_plot:
      if fig is None:
        fig,ax = plt.subplots(1,1)
      else:
        fig.clear()
        ax = fig.axes
    if two_plot:
      if fig is None:
        fig,ax = plt.subplots(2,1,sharex=True)
      else:
        ax = fig.axes
 
    ax = np.atleast_1d(ax)
    if showTrend:
      plt.sca(ax[1])
      plt.pcolormesh(q,x,data)
      plt.ylabel("image number, 0 being older")
      plt.xlabel(r"q ($\AA^{-1}$)")
      plt.clim( *clim )
    if plotAverage:
        ax[0].plot(q,np.nanmean(data,axis=0))
    if (plotAverage or showTrend) and title is not None:
      plt.title(title)

def plotdiffs(data,select=None,err=None,absSignal=None,absSignalScale=10,
           showErr=False,cmap=plt.cm.jet,fig=None,title=None,plotDiffRef=False):
    """ Plot difference data

        Parameters
        ----------
        data: dict-like object or (q,diffs,scan)tuple
          if dict-like it has to have keys
          'q' used for one axis
          'diffs' for 2D array[numscan,nQ]
          'scan'  scan points
          if present 'err' is used for errorbars
          if present 'diffs_plus_ref' is used
        select: slice
          to select time delays (for example select=slice(None,None,2))
        absSignal: array[nQ]
          if present, ass this data to the plot (could be the average)
        absSignalScale: float
          the factor to use to divide the absSignal for, it is kept as
          different parameters because the scaling it is added to the legend
        showErr: bool
          if True, plot errorbars as well
        cmap: matplotlib color map
          color map to use
        fig:
          matplotlib fig instance to use, if None creates a new one
        title: str
          title to use for plot (if None, no title is added)
        plotDiffRef:
          plot not only the diff but also the diffs_plus_ref (if present in data)
    """
    if isinstance(data,(list,tuple)):
        q,diffs,scan = args
        diffs_abs = None
    else:
        q = data["q"]
        diffs = data["diffs"]
        scan  = data["scan"]
        err   = data.get("err",None)
        diffs_abs = data.get("diffs_plus_ref",None)

    # this selection trick done in this way allows to keep the same colors when 
    # subselecting (because I do not change the size of diffs)
    if select is not None:
        indices = range(*select.indices(scan.shape[0]))
    else:
        indices = range(len(scan))
 
    if fig is None: fig = plt.figure()
 
    lines_diff = []
    lines_abs = []
    if absSignal is not None:
        line = plt.plot(q,absSignal/absSignalScale,lw=3,
                      color='k',label="absSignal/%s"%str(absSignalScale))[0]
        lines_abs.append(line)
    for linenum,idiff in enumerate(indices):
        color = cmap(idiff/(len(diffs)-1))
        label = timeToStr(scan[idiff])
        kw = dict( color = color, label = label )
        if err is not None and showErr:
            line = plt.errorbar(q,diffs[idiff],err[idiff],**kw)[0]
            lines_diff.append(line)
        else:
            line = plt.plot(q,diffs[idiff],**kw)[0]
            lines_diff.append(line)
            if diffs_abs is not None and plotDiffRef:
                line = plt.plot(q,diffs_abs[idiff],color=color)[0]
                lines_abs.append(line)
    if title is not None: fig.axes[0].set_title(title)
    legend = plt.legend(loc=4)
    plt.grid()
    plt.xlabel(r"q ($\AA^{-1}$)")
    # we will set up a dict mapping legend line to orig line, and enable
    # picking on the legend line
    lined = dict()
    for legline, origline in zip(legend.get_lines(), lines_diff):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline
  
    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig    = plt.gcf() 
        fig.canvas.draw()
 
    fig.canvas.mpl_connect('pick_event', onpick)
    return lines_diff,lines_abs

def updateLines(lines,data):
    for l,d in zip(lines,data):
        l.set_ydata(d)
