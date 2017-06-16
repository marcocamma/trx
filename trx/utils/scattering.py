# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import numpy as np
np.seterr(all='ignore')

def volumeFraction(concentration=1,molWeight=17,density=1.347):
  """ molWeight is kDa 
      concentration in mM
      density g/ml
  """
  concentration_mg_ml = concentration*molWeight
  volume_fraction    = concentration_mg_ml/density/1e3
  return volume_fraction

def molecularMass_from_I0(I0,c,density=1.347):
    """ calculate the molecular mass in Da

        based upon:
        D Orthaber, A Bergmann and O Glatter
        J. Appl. Cryst. (2000). 33, 218-225


        Paramters
        ---------
        I0 : float
            extrapolated value (in cm-1)
        c : float
            concentration (in g/ml)
        density : float
            protein density (in g/ml)
    """
    delta_rho = 2.67e10; # [cm-2]
    delta_rho_M = delta_rho/density
    Navo = 6e23
    M = I0*Navo / (c*delta_rho_M**2)
    return M


def radToQ(theta,**kw):
  """ theta is the scattering angle (theta_out - theta_in);
      kw should be have either E or wavelength
      it returns the scattering vector in the units of wavelength """
  # Energy or wavelength should be in kw
  assert "E" in kw or "wavelength" in kw,\
    "need wavelength or E to convert rad to Q"
  # but not both
  assert not ("E" in kw and "wavelength" in kw),\
    "conflicting arguments (E and wavelength)"
  if "E" in kw: kw["wavelength"] = 12.398/kw["E"]
  return 4*np.pi/kw["wavelength"]*np.sin(theta/2)

def degToQ(theta,**kw):
  theta = theta/180.*np.pi
  return radToQ(theta,**kw)
degToQ.__doc__ = radToQ.__doc__

def qToTwoTheta(q,asDeg=False,**kw):
  """ Return scattering angle from q (given E or wavelength) """
  # Energy or wavelength should be in kw
  assert "E" in kw or "wavelength" in kw,\
    "need wavelength or E to convert rad to Q"
  # but not both
  assert not ("E" in kw and "wavelength" in kw),\
    "conflicting arguments (E and wavelength)"
  if "E" in kw: kw["wavelength"] = 12.398/kw["E"]
  theta = 2*np.arcsin(q*kw["wavelength"]/4/np.pi)
  if asDeg: theta = np.rad2deg(theta)
  return theta
