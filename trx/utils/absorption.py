# -*- coding: utf-8 -*-
from __future__ import print_function,division,absolute_import

import logging
log = logging.getLogger(__name__)  # __name__ is "foo.bar" here

import numpy as np
np.seterr(all='ignore')

def attenuation_length(compound=None, density=None, natural_density=None,energy=None, wavelength=None):
  """ extend periodictable.xsf capabilities """
  import periodictable.xsf
  if energy is not None: wavelength = periodictable.xsf.xray_wavelength(energy)
  assert wavelength is not None, "scattering calculation needs energy or wavelength"
  if (np.isscalar(wavelength)): wavelength=np.array( [wavelength] )
  n = periodictable.xsf.index_of_refraction(compound=compound,
                           density=density, natural_density=natural_density,
                           wavelength=wavelength)
  attenuation_length = (wavelength*1e-10)/ (4*np.pi*np.imag(n))
  return np.abs(attenuation_length)


def transmission(compound='Si',thickness=100e-6, att_len=None,density=None, 
    natural_density=None,energy=None, wavelength=None,angle=None):
    """ calculate transmission of a given thickness of compound.
        if att_len (in m-1) is not given, it is calculated based on compound 
        information and energy or wavelength)
        if angle is given, the thickness is divided by cos(angle)
        angle has to be in deg. angle=0 means perpendicular
    """
    if att_len is None:
        att_len = attenuation_length(compound=compound,density=density,
                  natural_density=natural_density,energy=energy,
                  wavelength=wavelength)
    # takes care if both are arrays ...
    if angle is not None:
      thickness /= np.cos(np.deg2rad(angle))
    argument = np.squeeze(np.outer(thickness,1/att_len))
    return np.exp(-argument)

def A(compound='Si',thickness=100e-6, att_len=None,density=None, 
    natural_density=None,energy=None, wavelength=None,angle=None):
    """ calculate absorption of a given thickness of compound.
        if att_len (in m-1) is not given, it is calculated based on compound 
        information and energy or wavelength)
        if angle is given, the thickness is divided by cos(angle)
        angle has to be in deg. angle=0 means perpendicular
    """
    T = transmission(compound=compound,density=density,att_len=att_len,
                  natural_density=natural_density,energy=energy,
                  thickness=thickness, wavelength=wavelength,angle=angle)
    return 1-T


def _phosphorAbsorption(twotheta,mu=17700,thickness=40e-6,energy=None):
  """ return 1-np.exp(-mu*thick/np.cos(np.deg2rad(theta)))
      - mu is the neperian absorption linear coefficient (m-1)
      - twotheta is the scattering angle (in degrees)
  """
  if mu =='auto':
    assert energy is not None, "phosphorAbsorption with automatic mu requires energy"
    att_len = attenuation_length("Ce",energy=energy,density=4.75)
  else:
    att_len = 1/mu
  return A(att_len=att_len,thickness=thickness,angle=twotheta)

def phosphorCorrection(twotheta,mu=17700,thickness=40e-6,energy=None,normalizeToZeroAngle=False):
  """ helper function to correct for angle dependent absorption of the phosphor screen for an
      x-ray detector.
      return the correction factor one has to *multiply* the data for.
      - mu is the neperian absorption linear coefficient (m-1) 
          [could be 'auto' if energy is given]
      - twotheta is the scattering angle (in degrees)
  """
  corr = 1/_phosphorAbsorption(twotheta,mu=mu,thickness=thickness,energy=energy)
  if normalizeToZeroAngle:
    corr = corr*_phosphorAbsorption(0,mu=mu,thickness=thickness,energy=energy)
  return corr

def liquidSheetAbsorptionCorrection(twotheta=0,compound="H2O",thickness=300e-6,
    density=1,att_len=None,energy=None,wavelength=None,
    normalizeToZeroAngle=False):
    """ helper function for angular dependence of sample absorption

        returns the correction factor one has to *multiply* the intensity for
        the uncorrected I(theta) is divided by the T(theta)/T(0)
        where the transmittion T(theta) is:
        1/(mu*l) cos(theta)/(1-cos(theta))*(exp(-mu*l)-exp(-mu*l/cos(theta)))
        twotheta is in deg
    """
    if att_len is None : 
        att_len = attenuation_length(compound=compound,density=density,
                  energy=energy,wavelength=wavelength)
    if np.isscalar(twotheta): twotheta = np.asarray( (twotheta,) )
    twotheta = np.deg2rad(twotheta)
    idx = (twotheta < 0.001)
    twotheta[idx] = 0.001
    cos      = np.cos(twotheta)
    t1 = transmission(att_len=att_len,thickness=thickness)
    t2 = transmission(att_len=att_len,thickness=thickness/cos)
    arg = thickness/att_len
    temp = 1/arg*cos/(1-cos)*(t1-t2)
    if normalizeToZeroAngle: temp /= transmission(att_len=att_len,thickness=thickness)
    return 1/temp


def chargeToPhoton(chargeOrCurrent,compound="Si",thickness=100e-6,energy=10,e_hole_pair=3.63):
  """
    Function to convert charge (or current to number of photons (or number 
    of photons per second)
    
    Parameters
    ----------
    
    chargeOrCurrent: float or array
    compound : str
       Used to calculate 
     
  """
  # calculate absorption
  abs_fraction = A(compound=compound,energy=energy,thickness=thickness)
  chargeOrCurrent = chargeOrCurrent/abs_fraction

  e_hole_pair_energy_keV = e_hole_pair*1e-3
  n_charge_per_photon = energy/e_hole_pair_energy_keV
  # convert to Q
  charge_per_photon = n_charge_per_photon*1.60217662e-19
  nphoton = chargeOrCurrent/charge_per_photon
 
  if len(nphoton) == 1: nphoton = float(nphoton)
  return nphoton
