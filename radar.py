import numpy as np
from scipy.constants import c
    
_x_table = np.array([
    0.10997,  0.11685,  0.12444,  0.13302,  0.14256,  0.15256,  0.16220,
    0.17138,  0.18039,  0.18982,  0.20014,  0.21237,  0.22902,  0.25574,
    0.30537,  0.42028,  0.56287,  0.71108,  0.86714,  1.0529,   1.2790,
    1.5661,   1.8975])

_z_table = np.array([
    0.001220, 0.001735, 0.002468, 0.003511, 0.004993, 0.007102, 0.01010,
    0.01437,  0.02044,  0.02907,  0.04135,  0.05881,  0.08365,  0.1190,
    0.1692,   0.2407,   0.3424,   0.4870,   0.6927,   0.9852,   1.401,
    1.993,    2.835])
    
#-------------------------------------------------------------------
# Define the x and z value limit check conditions     
# invert the z value limit checks to get the equivalent x value limits

_x_optical_limit = np.sqrt(4.*5/np.pi)
_x_rayleigh_limit = (4.*0.03/(9. * np.pi**5))**(1./6)
_z_optical_limit = 5
_z_rayleigh_limit = 0.03

def diameter_to_rcs(diameter, frequency):

 
    wavelength = c/frequency
    x = diameter/wavelength
    
    # Vectorize all the things
    optical_cond = x > _x_optical_limit
    rayleigh_cond = x < _x_rayleigh_limit
    mie_cond = np.logical_not(optical_cond, rayleigh_cond)

    # allocate some space for the results
    z = np.empty_like(diameter)

    # are we in the optical regime?
    z[optical_cond] = np.pi * x[optical_cond]**2 / np.pi

    # are we in the Rayleigh regime?
    z[rayleigh_cond] = 9. * x[rayleigh_cond]**6 * np.pi**5 / 4
    
    # we are in the Mie resonance regime    
    z[mie_cond] = np.interp(x[mie_cond], _x_table, _z_table)

    return z * wavelength**2 


def rcs_to_diameter(rcs, frequency):

    wavelength = c/frequency
    z = rcs/(wavelength**2)

   # Vectorize all the things
    optical_cond = z > _z_optical_limit
    rayleigh_cond = z < _z_rayleigh_limit
    mie_cond = np.logical_not(optical_cond, rayleigh_cond)

    # allocate some space for the results
    x = np.empty_like(rcs)

   # are we in the optical regime?
    x[optical_cond] = np.sqrt(4.*z[optical_cond]/np.pi)

    # are we in the Rayleigh regime?
    x[rayleigh_cond] = (4.*z[rayleigh_cond]/(9. * np.pi**5))**(1./6.)

    # we are in the Mie resonance regime
    x[mie_cond] = np.interp(z[mie_cond], _z_table, _x_table)

    return x*wavelength


if __name__ == '__main__':
    
    frequency = 400e6

    diameter = np.linspace(0.001, 10)

    rcs = diameter_to_rcs(diameter, frequency)

    diameter_check = rcs_to_diameter(rcs, frequency)

    for d, c in zip(diameter, diameter_check):
        print(f'{d:25.3f}  {c:25.3f}')