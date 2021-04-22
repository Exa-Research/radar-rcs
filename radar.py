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
    
# invert the z value limit checks to get the equivalent x value limits
_x_optical_limit = np.sqrt(4.*5/np.pi)
_x_rayleigh_limit = (4.*0.03/(9. * np.pi**5))**(1./6)


def diameter_to_rcs(diameter, frequency):

    wavelength = c/frequency
    x = diameter/wavelength
    
    # are we in the optical regime?
    if x > _x_optical_limit:
        z = np.pi * x**2 / np.pi

    # are we in the Rayleigh regime?
    elif x < _x_rayleigh_limit:
        z = 9. * x**6 * np.pi**5 / 4.
    
    # we are in the Mie resonance regime    
    else:
        z = np.interp(x, _x_table, _z_table)

    return z * wavelength**2 


def rcs_to_diameter(rcs, frequency):

    wavelength = c/frequency
    z = rcs/(wavelength**2)

    # are we in the optical regime?
    if z > 5:
        x = np.sqrt(4.*z/np.pi)

    # are we in the Rayleigh regime?
    elif z < 0.03:
        x = (4.*z/(9. * np.pi**5))**(1./6.)

    # we are in the Mie resonance regime
    else:
        x = np.interp(z, _z_table, _x_table)

    return x*wavelength


if __name__ == '__main__':
    
    frequency = 400e6

    diameter = np.logspace(0.01, 10)

    for d in diameter:

        rcs = diameter_to_rcs(d, frequency)

        diameter_check = rcs_to_diameter(rcs, frequency)

        print(f'{d:25.3f}  {diameter_check:25.3f}')