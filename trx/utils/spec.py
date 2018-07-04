""" 
This script is useful for reading the motor positions in a given specscan
    It is even more useful to know what on hell has chenged between two scans
    just type specscan filename scan1 scan2 to have a nice printout that can 
    help you find the difference
"""

from __future__ import division,print_function
try:
    import PyMca5
except ImportError:
    print("Cannot import PyMca5, utils.spec will not work")
    pass

import numpy as np
import matplotlib.pylab as plt
import matplotlib
import sys
import datastorage



class SpecFile(object):
    def __init__(self,fname):
        """ Esrf Spec file object """
    
        self.fname = fname
        self.f = PyMca5.specfile.Specfile(fname)

    def showMotors(self,scanno=1):
        """
        function that can display in a kind-of-nice way the motor positions during a scan
        it highlights the differences
        """
        if isinstance(scanno,(str,int)): scanno = (scanno,)
        scans = [self.f.select(str(s)) for s in scanno]
        motors = scans[0].allmotors()
        pos    = [scan.allmotorpos() for scan in scans]
        idx = np.argsort(motors)
        print("Specfile %s"%self.fname)
        print("%13s"%"ScanNum",end="")
        [print(" %13s"%scan,end="") for scan in scanno]
        print("")
        print("%13s"%"Motor",end="")
        [print(" %13s"%"Position",end="") for scan in scanno]
        print("")
        for i in idx:
            print("%13s"%motors[i],end="")
            for p in pos:
                print(" %13s"%p[i],end="")
                if p[i] != pos[0][i]: print(" <-----",end="")
          # highlight if they are not equal
#         if not np.allclose(np.asarray([p[i] in pos]) == pos[0][i],1e-4): print(" <-----",end="")
            print("")

    def getScan(self,scanno=1,ycol='all',normalize="max"):
        scan = self.f.select(str(scanno)) 
        data = scan.data()
        lbl = scan.alllabels()
        lbl = [label.replace("/","_") for label in lbl]
        if ycol == "all":
            ret = datastorage.DataStorage()
            for i,name in enumerate(lbl):
                ret[name] = data[i]
            return ret
        else:
            if normalize is not None:
              if isinstance(normalize,(int,slice)):
                  y /= np.mean(y[normalize])
              if normalize == "max":
                  y /= y.max()
            return x,y
 
    
    def showScan(self, scanno=1):
        if isinstance(scanno,(str,int)): scanno = (scanno,)
        scans = [self.f.select(str(s)) for s in scanno]
        data = scans[0].data()
        lbl = scans[0].alllabels()
        if matplotlib.is_interactive():
            plt.ioff()
        plt.plot(data[0], data[3]/np.mean(data[3][:5]))
        plt.xlabel("%s" % lbl[0]);    plt.ylabel("Trasmission (%s)" % lbl[3])
        plt.title("Capillary profile scan %s " % scanno)
        plt.show()    

