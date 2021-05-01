import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

#-------------------------------------------------------------------
# Look up tables for the piecewise approximation function g(z). Note
# that the table is incomplete because the z values do not extend all
# the way to z=5 (the threshold for transitioning to the optical region).
# This causes inaccurate interpolation when operating in this region.
# So we'll add one additional point at z=5 to make it complete.
#-------------------------------------------------------------------

_x_table = np.array([
    0.10997,  0.11685,  0.12444,  0.13302,  0.14256,  0.15256,  0.16220,
    0.17138,  0.18039,  0.18982,  0.20014,  0.21237,  0.22902,  0.25574,
    0.30537,  0.42028,  0.56287,  0.71108,  0.86714,  1.0529,   1.2790,
    1.5661,   1.8975,   np.sqrt(4*5/np.pi)])

_z_table = np.array([
    0.001220, 0.001735, 0.002468, 0.003511, 0.004993, 0.007102, 0.01010,
    0.01437,  0.02044,  0.02907,  0.04135,  0.05881,  0.08365,  0.1190,
    0.1692,   0.2407,   0.3424,   0.4870,   0.6927,   0.9852,   1.401,
    1.993,    2.835,    5.0])


def plot_g_z_table():
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.scatter(_x_table, _z_table)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title("Interpolating g(z) Table")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

#-------------------------------------------------------------------
# Define the x and z value limit check conditions     
#-------------------------------------------------------------------

_z_optical_limit = 5
_z_rayleigh_limit = 0.03
_x_optical_limit = np.sqrt(4. * _z_optical_limit/np.pi)
_x_rayleigh_limit = (4. * _z_rayleigh_limit/(9. * np.pi**5))**(1./6)

def norm_diameter_to_norm_rcs(x):
    
    # Vectorize all the things and make sure they work with scalar inputs
    optical_cond  = np.asarray(x > _x_optical_limit)
    rayleigh_cond = np.asarray(x < _x_rayleigh_limit)
    mie_cond = np.logical_not(np.logical_or(optical_cond, rayleigh_cond))

    # allocate some space for the results
    z = np.empty_like(x)

    # are we in the optical regime?
    z[optical_cond] = np.pi * x[optical_cond]**2 / 4

    # are we in the Rayleigh regime?
    z[rayleigh_cond] = 9. * x[rayleigh_cond]**6 * np.pi**5 / 4
    
    # we are in the Mie resonance regime    
    z[mie_cond] = np.interp(x[mie_cond], _x_table, _z_table)

    return z


def norm_rcs_to_norm_diameter(z):

    # Vectorize all the things and make sure they work with scalar inputs
    optical_cond  = np.asarray(z > _z_optical_limit)
    rayleigh_cond = np.asarray(z < _z_rayleigh_limit)
    mie_cond = np.logical_not(np.logical_or(optical_cond, rayleigh_cond))

    # allocate some space for the results
    x = np.empty_like(z)

   # are we in the optical regime?
    x[optical_cond] = np.sqrt(4.*z[optical_cond]/np.pi)

    # are we in the Rayleigh regime?
    x[rayleigh_cond] = (4.*z[rayleigh_cond]/(9. * np.pi**5))**(1./6)

    # we are in the Mie resonance regime
    x[mie_cond] = np.interp(z[mie_cond], _z_table, _x_table)

    return x


if __name__ == '__main__':
    
    frequency = 400e6

    diameter = np.linspace(0.001, 10)

    # calculate the RCS from the diameter
    rcs = diameter_to_rcs(diameter, frequency)

    # now try inverting the equation to get the diameter back
    diameter_check = rcs_to_diameter(rcs, frequency)

    # do we match?
    print(np.allclose(diameter, diameter_check, rtol=0.001))

    # where did we fail?
    for d, c in zip(diameter, diameter_check):
        print(f'{d:25.3f}  {c:25.3f}   {d==c:10b}  {100*(c-d)/d:12.1f}% error')

    test_rcs = diameter_to_rcs(diameter[9], frequency)
    rcs_to_diameter(test_rcs, frequency)