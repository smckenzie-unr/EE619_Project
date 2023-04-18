#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:36:08 2023

@author: smckenzie
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import doppler_freq, fft_convolve
from scipy.constants import pi, c
from scipy.fft import fft
from antenna import Antenna
from matplotlib import animation
from utilities import lfm_pulses, pulse, white_noise

PLOT_SNR = True
PLOT_DOPPLER = True
PLAY_RUN = True  

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 1      

if(__name__ == "__main__"):
    plt.close("all")
    print("Scott L. McKenzie EE619 Project.")
    print("LFM Radar simulation.\nVersion {0:d}.{1:d}.{2:d}".format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH))
    
    Fc              = 10e9                                                      #Hz
    TX_pwr          = 2e3                                                       #watts
    sys_loss        = 8                                                         #dB
    reciever_noise  = 5                                                         #dB
    radar_height    = 10                                                        #m
    rcs             = -10                                                       #dBsm
    target_height   = 200                                                       #m
    init_range      = 4.5e3                                                     #m
    target_veloctiy = 100                                                       #m/s
    Fs              = 40e6                                                      #Hz
    M               = 128                                                       #n PRI's
    tau             = 2e-6                                                      #s
    beta            = 20e6                                                      #Hz
    PRF             = 20e3                                                      #Hz
    nDwells         = 16                                                        #n
    thermal_noise   = -174                                                      #dBm / Hz
    noise_density   = 2                                                         #dB / Hz
    
    ant             = Antenna(2 * c / Fc, 2 * c / Fc, Fc, radar_height)
    
    dwell_time      = nDwells * 1 / PRF * M
    process_bw      = PRF / M
    duty_factor     = tau * PRF
    
    time_vect       = np.arange(0.0, init_range / target_veloctiy, dwell_time)
    target_range    = init_range - (target_veloctiy * time_vect)
    radial_range    = np.sqrt(((target_height - radar_height) ** 2) + (target_range ** 2))
    
    ant.get_pattern(image = True)
    dir_gain        = ant.directional_gain(radial_range, 0, target_height, plot = True)
    signal_dB       = 10.0 * np.log10(duty_factor * TX_pwr * 1e3) + 2 * dir_gain + 20.0 * np.log10(c / Fc) + rcs - (30 * np.log10(4 * pi) + 40 * np.log10(radial_range) + sys_loss)
    noise_dB        = thermal_noise + 10 * np.log10(process_bw) + reciever_noise
    SNR             = signal_dB - noise_dB
    
    if(PLOT_SNR):
        plt.figure("SNR vs Range")
        plt.plot(radial_range, SNR)
        plt.xlim(min(radial_range), max(radial_range))
        plt.ylim(-10, 30)
        plt.xlabel("Range [m]")
        plt.ylabel("SNR [dB]")
        plt.title("SNR vs Range")
        plt.grid(True)
        # plt.show(block = False)
        plt.draw()
        plt.pause(0.01)  
    
    time_delay      = np.linspace(np.max(radial_range) * 2 / c, np.min(radial_range) * 2 / c, len(time_vect)) - tau
    angle_target    = np.arcsin((target_height - radar_height) / radial_range)
    radial_vel      = target_veloctiy - target_veloctiy * np.sin(angle_target)
    dopp_freqs      = doppler_freq(Fc, radial_vel)
    
    if(PLOT_DOPPLER):
        plt.figure("Doppler vs Range")
        plt.plot(radial_range, dopp_freqs)
        plt.xlim(min(radial_range), max(radial_range))
        plt.ylim(min(dopp_freqs), max(dopp_freqs))
        plt.xlabel("Range [m]")
        plt.ylabel("Doppler frequency [Hz]")
        plt.title("Doppler frequency vs Range")
        plt.grid(True)
        # plt.show(block = False)
        plt.draw()
        plt.pause(0.01)  
    
    L               = round(Fs / PRF)
    PRI             = 1 / PRF
    dt              = 1 / Fs
    N               = int(tau * Fs)
    
    t_scan          = np.linspace(0, M / PRF, int(M / PRF * Fs))
    t_pri           = np.linspace(0, 1 / PRF, int(1 / PRF * Fs))
    time_pulse      = np.linspace(0, M * PRI - dt, int((M * PRI) / dt))
    dopp_imag       = np.zeros(shape = (M, len(t_pri)), dtype = complex)
    matched_filter  = np.flip(np.conjugate(lfm_pulses(time_pulse[0:N + 1], tau, PRI, beta, L, 1)))
    weights         = np.hamming(M)
    
    RTI             = np.zeros(shape = (len(time_vect), M, len(t_pri)), dtype = complex)
    RDI             = np.zeros(shape = (len(time_vect), M, len(t_pri)), dtype = complex)
    
    for td_idx in range(len(time_vect)):
        scaler_lvl          = 10 ** ((SNR[td_idx] + noise_density) / 20)
        rx_signal           = scaler_lvl * lfm_pulses(time_pulse - time_delay[td_idx], tau, PRI, beta, L, M) * pulse(time_pulse, dopp_freqs[td_idx]) + white_noise(10 ** (noise_density / 20), len(time_pulse))
        mf_output           = fft_convolve(rx_signal, matched_filter)
        RTI[td_idx, :, :]   = mf_output.reshape((M, len(t_pri)))
        fd                  = np.fft.fftshift(fft(RTI[td_idx, :, :].T * weights), axes=(1,))
        RDI[td_idx, :, :]   = fd.T
        print("\rProcess progress: {0:.2f}%.".format((td_idx + 1) / len(time_vect) * 100), end = '')
    
    def update_image(td_idx):
        rti.set_data(10 * np.log10(abs(RTI[td_idx, :, :])))
        rdi.set_data(10 * np.log10(abs(RDI[td_idx, :, :])))
    
    if(PLAY_RUN):
        RTE_E               = [0, (L / Fs) * c / 2, 0, M]
        RDI_E               = [0, (L / Fs) * c / 2, -PRF / 2 * 1e-3, PRF / 2 * 1e-3]
        
        fig                 = plt.figure("Fly in")
        ax                  = fig.add_subplot(211)
        rti                 = ax.imshow(10 * np.log10(abs(RTI[0, :, :])), vmin = 12.5, vmax = 30.0, origin = "lower", interpolation = "none", cmap = "jet", extent = RTE_E, aspect='auto', animated = True)
        ax.title.set_text("Range-Time Image")
        plt.xlabel("Range [m]")
        plt.ylabel("PRI [n]")
        
        ax                  = fig.add_subplot(212)
        rdi                 = ax.imshow(10 * np.log10(abs(RDI[0, :, :])), vmin = 20.0, vmax = 50.0, origin = "lower", interpolation = "none", cmap = "jet", extent = RDI_E, aspect='auto', animated = True)
        ax.title.set_text("Range-Doppler Image")
        plt.xlabel("Range [m]")
        plt.ylabel("Doppler Frequency [kHz]")
        
        ani = animation.FuncAnimation(fig, update_image, interval=0.01, frames = len(time_vect), repeat = False)
        plt.subplots_adjust(hspace = 0.6)
        plt.show()
 