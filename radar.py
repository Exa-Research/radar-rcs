# SPDX-FileCopyrightText: © 2021 Exa Research, LLC <tj@exaresearch.com>

#-------------------------------------------------------------------
# This file implements various routines used to calculate the radar cross
# section of a sphere given it's diameter and radar frequency. It uses the
# NASA Size Estimation Model (SEM) as documented in "Haystack and HAX Radar
# Measurements of the Orbital Debris Environment; 2003", Section 4.0.
# https://www.orbitaldebris.jsc.nasa.gov/library/haystack_hax_radar2003.pdf
#-------------------------------------------------------------------

import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

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

def norm_diameter_to_norm_rcs(x: float) -> float:

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


def norm_rcs_to_norm_diameter(z: float) -> float:

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


def diameter_to_rcs(frequency: float, diameter: float) -> float:
    """Calculates the radar cross section (RCS) of a sphere at the specified
    diameter based on the radar frequency.

    Args:
        frequency (float): Frequency of the radar in Hz.
        diameter (float or array of floats): Diameter of the sphere in m.

    Returns:
        float or array of floats: Radar cross section in m^2.
    """

    wavelength = scipy.constants.c / frequency
    norm_diameter = np.asarray(diameter)/wavelength
    norm_rcs = norm_diameter_to_norm_rcs(norm_diameter)
    rcs = norm_rcs * wavelength**2
    return rcs


def rcs_to_diameter(frequency: float, rcs: float) -> float:
    """Calculates the diameter of a sphere given its RCS based on the
    radar frequency.

    Args:
        frequency (float): Frequency of the radar in Hz.
        rcs (float): Radar cross section in m^2.

    Returns:
        float: Diameter in m.
    """

    wavelength = scipy.constants.c / frequency
    norm_rcs = np.asarray(rcs)/(wavelength**2)
    norm_diameter = norm_rcs_to_norm_diameter(norm_rcs)
    diameter = norm_diameter * wavelength
    return diameter


def plot_rcs(frequency, title=None, diameter=None, ref_diameter=None, use_db_scale=None, figsize=None):
    """Plots the radar cross section (RCS) as a function of diameter for a radar operating
    at the specified frequency.

    Args:
        frequency (float): Frequency of the radar in Hz.
        title ([str], optional): A text string to display as the title of the figure. Defaults to None.
        diameter ([float], optional): An array or list of diameters [m] used to generate the continous line. If None then
        we'll use values in the range from 0.01 to 10 meters. Defaults to None.
        ref_diameter ([float], optional): An array or list of diameters [m] used to mark particular reference points
        on the graph. If none then we'll use values of 0.02, 0.1, 1, 5, and 10 meters. Defaults to None.
        use_db_scale ([bool], optional): Flag to indicate whether the Y-axis should be expressed in decibels dB m^2
        or linear units. The default value is True.

    Returns:
        fig, ax: The matplotlib figure and axes objects.
    """

    # Default diameters range is 0.01 to 10 meters. We'll use these to
    # plot the background smooth curve.

    if diameter is None:
        diameter = np.logspace(-2, 1)
    rcs = diameter_to_rcs(frequency, diameter)

    # Default reference diameters are 2 cm, 5 cm, 10 cm, 50 cm, 1 m, 2 m, 5 m, and 10 m. We'll use
    # these to annotate specific points on the curve

    if ref_diameter is None:
        ref_diameter = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
    ref_rcs = diameter_to_rcs(frequency, ref_diameter)

    if title is None:
        title = f'RCS at Frequency {frequency/1e6} MHz'

    if use_db_scale is None:
        use_db_scale = True

    if figsize is None:
        figsize = (10, 7)

    # Plot the results
    locator = LogLocator(base=10, subs=(0.2, 0.5, 1))
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

    fig, ax = plt.subplots(figsize=figsize)

    if use_db_scale:
        ax.set_ylabel('RCS [dB sm]')
        ax.set_ylim(ymin=-100, ymax=20)
        rcs = 10*np.log10(rcs)
        ref_rcs = 10*np.log10(ref_rcs)
    else:
        ax.set_yscale('log')
        ax.set_ylim(ymin=1e-10, ymax=100)
        ax.set_ylabel('RCS [$m^2$]')

    ax.set_xscale('log')
    ax.set_xlim(xmin=0.01)
    ax.set_xlabel('Sphere Diameter [m]')

    ax.plot(diameter, rcs, color='black')
    ax.scatter(ref_diameter, ref_rcs, color='black')
    ax.annotate(f'Reference spheres of diameter {ref_diameter} m', xy=(0.5, 0.1), xycoords='axes fraction',
                ha='center')
    # label the reference points
    for i, txt in enumerate(ref_rcs):
        ax.annotate(f'{txt:.3g}' , xy=(ref_diameter[i], ref_rcs[i]), textcoords='offset points',
                    xytext=(-20, 10))

    ax.set_title(title)
    ax.xaxis.set_major_formatter(formatter);
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    return fig, ax


def max_antenna_gain(model, size, frequency, efficiency):
    
    wavelength = scipy.constants.c / frequency
    
    if model == 'Parabolic':
        max_gain = efficiency * (np.pi * size / wavelength)**2
        
    else:
        
        raise ValueError('Unknown antenna model')
        
    # return value in dB
    return 10*np.log10(max_gain)


if __name__ == '__main__':

    frequency = 400e6

    diameter = np.linspace(0.001, 10)

    # calculate the RCS from the diameter
    rcs = diameter_to_rcs(frequency, diameter)

    # now try inverting the equation to get the diameter back
    diameter_check = rcs_to_diameter(frequency, rcs)

    # do we match?
    rtol = 0.0001
    print(f"Do the values match at rtol={rtol}? ", np.allclose(diameter, diameter_check, rtol=rtol))

    # How well did they match?
    for d, c in zip(diameter, diameter_check):
        print(f'{d:25.3f}  {c:25.3f}   {d==c:10b}  {100*(c-d)/d:12.2f}% error')

    # test that things work with individual numbers, not just arrays.
    test_rcs = diameter_to_rcs(frequency, diameter[9])
    test_diameter = rcs_to_diameter(frequency, test_rcs)
