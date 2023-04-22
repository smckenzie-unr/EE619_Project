#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:31:42 2023

@author: smckenzie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.constants import pi, c
from time import sleep

#%% Utility functions
def find_nearest(array, value):
    """
    Returns the index of an array that is closest to the value
    
    Parameters
    -------------
    array: input array to be searched for closest k value 
    value: k value to be used to find closest value 
    """ 
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def pulse(t, Fc):
    """
    Returns a complex pulse signal with carrier frequency 
    
    Parameters
    -------------
    t: time vector 
    Fc: carrier frequency 
    """ 
    
    return np.exp(1j * 2 * pi * t * Fc)

def doppler_freq(Fc, velocity):
    """
    Returns the doppler frequency due to radial velocity 
    
    Parameters
    -------------
    Fc: carrier frequency 
    velocity: radial veclocity of target 
    """ 
    
    lambda_val = c / Fc
    return 2 * velocity / lambda_val

def rect(t):
    """
    Returns the output of a rect(t) function  
    
    Parameters
    -------------
    t: time vector 
    """ 
    return np.where(abs(t) <= 0.5, 1, 0)

def lfm_pulses(t, tau, PRI, beta, L, M):
    """
    Returns a complex linear frequency modulated pulse signal with carrier frequency 
    
    Parameters
    -------------
    t: time vector 
    tau: pulse width 
    PRI: Pulse repetition interval 
    beta: frequency modulation bandwidth 
    L: Samples per PRI 
    M: Samples per CPI 
    """ 
    
    tmod = t % PRI
    rect_p = rect((t[0:L] - (tau / 2)) / tau)
    rect_train = np.tile(rect_p, M)
    return np.exp(1j * pi * (beta / tau) * (tmod **2)) * rect_train    

def fft_convolve(x1, x2):
    """
    Returns the convolutio of two signals 
    
    Parameters
    -------------
    x1: first signal to be convolved 
    x2: second signal to be convolved 
    """ 
    
    N = max(len(x1), len(x2))
    X1 = fft(x1, n = N)
    X2 = fft(x2, n = N)
    Y = X1 * X2
    return ifft(Y)

def white_noise(level_dB, n, mu = 0.0):
    """
    Returns white noise signal with spectral noise density 
    
    Parameters
    -------------
    level_dB: spectral noise density unit/SQRT(Hz) 
    n: number of points 
    mu: mean value 
    """
    
    sigma = 10 ** (level_dB / 20)
    noise = np.random.normal(mu, sigma, n) + 1j * np.random.normal(mu, sigma, n)
    return noise

def interpolation(f_k, alpha, y_kp1, y_kn1, y_k):
    """
    Returns interpolated estimated index
    
    Parameters
    -------------
    f_k: Max frequency/range index 
    alpha: scale factor 
    y_kp1: abolute signal value at max index + 1 
    y_kn1: abolute signal value at max index - 1 
    y_k1: abolute signal value at max index 
    """
    
    num = y_kp1 - y_kn1
    return f_k + alpha * (num / y_k)

def time_delay_vector(target_range):
    """
    Returns a vector of time delays based on range
    
    Parameters
    -------------
    target_range: target position vector 
    """
    
    time_delay = np.zeros(len(target_range))
    for t in range(len(target_range)):
        time_delay[t] = target_range[t] * 2 / c
    return time_delay

def find(element, matrix):
    """
    Returns row and column of element in matrix
    
    Parameters
    -------------
    element: value to be searched for in matrix 
    matrix: matrix to be searched 
    """
    
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == element:
                return (i, j)









#%% TESTBENCH
if(__name__ == "__main__"):
    tau = 1e-6
    beta = 50e6
    t = np.linspace(0, tau * 4, 1000)
    chirp = np.zeros(len(t), dtype = complex)
    chirp[0:250] = lfm_pulse(t[0:250], beta, tau)
    mf = np.flip(np.conjugate(chirp[0:250]))
    td = np.linspace(tau, -tau, 1000)
    remember = chirp

    plt.figure()
    chirp[0:250] = lfm_pulse(t[0:250] - td[0], beta, tau) * pulse(t[0:250], 500e3)
    plt.plot(np.real(chirp))
    s_o = fft_convolve(chirp, mf)
    t_vals = np.linspace(0, tau * 4, len(s_o))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_vals / tau, np.abs(remember), linewidth = 4)
    line1, = ax.plot(t_vals / tau, np.abs(s_o) / np.max(np.abs(s_o)), linewidth = 4)
    plt.grid(True)
    for i in range(1, len(td), 1):  
        chirp[0:250] = lfm_pulse(t[0:250] - td[i], beta, tau) * pulse(t[0:250], 500e3) + white_noise(2, 250)
        s_o = fft_convolve(chirp, mf)
        line1.set_ydata(np.abs(s_o) / np.max(np.abs(s_o)))
        fig.canvas.draw()
        fig.canvas.flush_events()
        ax.set_xlim(0, t_vals[len(t_vals) - 1] / tau)
        sleep(0.005)
        
