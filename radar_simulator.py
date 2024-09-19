#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:36:08 2023

@author: smckenzie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c
from scipy.fft import fft
from antenna import Antenna
from matplotlib import animation
from utilities import lfm_pulses, pulse, white_noise
from utilities import doppler_freq, fft_convolve, interpolation, time_delay_vector, find

PLOT_SNR = True
PLOT_DOPPLER = True
PLAY_RUN = True  

ISOMETRIC_PATTERN = True
DIRECTION_GAIN_PLOT = True

PLOT_RANGE_ERROR = True
DOPPLER_RANGE_ERROR = True

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 3     

if(__name__ == "__main__"):
    plt.close("all")
    
    #%% Header
    print("Scott L. McKenzie EE619 Project.")
    print("LFM Radar simulation.\nVersion {0:d}.{1:d}.{2:d}\n".format(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH))
    
    #%% Radar Parameters
    Fc = 10e9                                                                   #Hz
    TX_pwr = 2e3                                                                #watts
    sys_loss = 8                                                                #dB
    reciever_noise = 5                                                          #dB
    radar_height = 10                                                           #m
    rcs = -10                                                                   #dBsm
    target_height = 200                                                         #m
    init_range = 4.5e3                                                          #m
    target_veloctiy = 100                                                       #m/s
    Fs = 40e6                                                                   #Hz
    M = 128                                                                     #n PRI's
    tau = 2e-6                                                                  #s
    beta = 20e6                                                                 #Hz
    PRF = 20e3                                                                  #Hz
    nDwells = 16                                                                #n
    thermal_noise = -174                                                        #dBm / Hz
    alpha = 0.5961                                                              #ratio
    sigma_n = 2.5                                                               #Scaler
    
    #%% Calculations and Vector setup
    ant = Antenna(2 * c / Fc, 2 * c / Fc, Fc, radar_height)
    dwell_time = nDwells * 1 / PRF * M
    process_bw = PRF / M
    duty_factor = tau * PRF
    
    time_vect = np.arange(0.0, init_range / target_veloctiy, dwell_time)
    target_range = init_range - (target_veloctiy * time_vect)
    radial_range = np.sqrt(((target_height - radar_height) ** 2) + (target_range ** 2))
    
    dir_gain = ant.directional_gain(radial_range, 0, target_height)
    signal_dB = 10.0 * np.log10(duty_factor * TX_pwr * 1e3) + 2 * dir_gain + 20.0 * np.log10(c / Fc) + rcs - (30 * np.log10(4 * pi) + 40 * np.log10(radial_range) + sys_loss)
    noise_dB = thermal_noise + 10 * np.log10(process_bw) + reciever_noise
    SNR = signal_dB - noise_dB
    
    time_delay = time_delay_vector(radial_range) - tau
    angle_target = np.arcsin((target_height - radar_height) / radial_range)
    radial_vel = target_veloctiy * np.cos(angle_target)
    dopp_freqs = doppler_freq(Fc, radial_vel)
    
    L = round(Fs / PRF)
    PRI = 1 / PRF
    dt = 1 / Fs
    N = int(tau * Fs)
    unamb_rng = PRI * c / 2
    
    time_pulse = np.linspace(0, M * PRI - dt, int((M * PRI) / dt))
    dopp_imag = np.zeros(shape = (M, L), dtype = complex)
    matched_filter = np.flip(np.conjugate(lfm_pulses(time_pulse[0:N + 1], tau, PRI, beta, L, 1)))
    weights = np.hamming(M)
    
    RTI = np.zeros(shape = (len(time_vect), M, L), dtype = complex)
    RDI = np.zeros(shape = (len(time_vect), M, L), dtype = complex)
    
    range_meas = []
    range_actual = []
    range_error = []
    
    dopp_meas = []
    dopp_actual = []
    dopp_error = []
    
    print("Max target radial range:{1:>10}{0:.2f} Meters".format(max(radial_range), ' '))
    print("Max Doppler frequency:{1:>12}{0:.2f} Hz".format(max(dopp_freqs), ' '))
    print("Minimum target elevation angle:{1:>3}{0:.1f} degrees".format(min(angle_target * 180 / pi), ' '))
    print("Max SNR:{1:>26}{0:.2f} dB\n".format(max(SNR), ' '))
    
    #%% Simulation Process
    for td_idx in range(len(time_vect)):
        scaler_lvl = 10 ** ((SNR[td_idx] + noise_dB) / 20)
        rx_signal  = scaler_lvl * lfm_pulses(time_pulse - time_delay[td_idx], tau, PRI, beta, L, M) * pulse(time_pulse, dopp_freqs[td_idx]) + sigma_n * white_noise(noise_dB, len(time_pulse))
        mf_output = fft_convolve(rx_signal, matched_filter)
        mf_reshape = mf_output.reshape((M, L))
        RTI[td_idx, :, :] = mf_reshape
        fd = np.fft.fftshift(fft(RTI[td_idx, :, :].T * weights), axes = (1, ))
        RDI[td_idx, :, :] = fd.T
        
        range_idx = []
        for i in range(1, M - 1, 1):
            ranger = np.abs(mf_reshape[i, :])
            bound_idx = [round(L / unamb_rng * radial_range[td_idx] * 0.1), round(L / unamb_rng * radial_range[td_idx] * 1.9)]
            idx = max(min(ranger.argmax(), bound_idx[1]), bound_idx[0])
            if((idx != L - 1) and (idx > 0)):
                range_idx.append(interpolation(idx, alpha, ranger[idx + 1], ranger[idx - 1], ranger[idx]))
        rng_meas = np.mean(range_idx) * unamb_rng / L
        range_error.append(rng_meas - radial_range[td_idx])
        range_meas.append(rng_meas)
        range_actual.append(radial_range[td_idx])
        

        rdi_search = np.abs(np.fft.fftshift(RDI[td_idx, :, :].T).T)
        max_val = max(map(max, rdi_search))
        row, col = find(max_val, rdi_search)
        bound_idx = [round(M / PRF * dopp_freqs[td_idx] * 0.8), round(M / PRF * dopp_freqs[td_idx] * 1.2)]
        row = max(min(row % M, bound_idx[1]), bound_idx[0])
        col = col % L
        curr_dopp = interpolation(row, alpha, rdi_search[row + 1, col], rdi_search[row - 1, col], rdi_search[row, col]) * PRF / M 
        dopp_meas.append(curr_dopp)
        dopp_actual.append(dopp_freqs[td_idx])
        dopp_error.append(dopp_freqs[td_idx] - curr_dopp)
        print("\rSimulation Processing:{1:>12}{0:.2f}% Complete".format((td_idx + 1) / len(time_vect) * 100, ' '), end = '')
        
    print("\n")
    #%% Plotting
    ant.get_pattern(image = ISOMETRIC_PATTERN)
    ant.directional_gain(radial_range, 0, target_height, plot = DIRECTION_GAIN_PLOT)
    
    if(PLOT_SNR):
        plt.figure("SNR vs Range")
        plt.plot(radial_range, SNR)
        plt.xlim(min(radial_range), max(radial_range))
        plt.ylim(-10, 30)
        plt.xlabel("Range [m]")
        plt.ylabel("SNR [dB]")
        plt.title("SNR vs Range")
        plt.grid(True)
        
    if(PLOT_DOPPLER):
        plt.figure("Doppler vs Range")
        plt.plot(radial_range, dopp_freqs)
        plt.xlim(0, max(radial_range))
        plt.ylim(min(dopp_freqs), max(dopp_freqs) * 1.20)
        plt.xlabel("Range [m]")
        plt.ylabel("Doppler frequency [Hz]")
        plt.title("Doppler frequency vs Range")
        plt.grid(True)
        
    if(PLOT_RANGE_ERROR):
        fig = plt.figure("Range Measurment")
        ax = fig.add_subplot(211)
        ax.plot(time_vect, range_actual, linewidth = 4, color = "black", label = "Range Actual")
        ax.plot(time_vect, range_meas, 'o', color = "orange", label = "Range Measured")
        ax.set_xlim(min(time_vect), max(time_vect))
        ax.set_title("Range vs time")
        ax.set_ylabel("Range [m]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc = "upper right")
        ax.grid(True)
        
        ax = fig.add_subplot(212)
        ax.plot(time_vect, range_error, linewidth = 2, label = "Range Error")
        ax.set_xlim(min(time_vect), max(time_vect))
        ax.set_ylim(-400, 400)
        ax.set_title("Range Error")
        ax.set_ylabel("Range Error [m]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc = "upper right")
        ax.grid(True)
        
    if(DOPPLER_RANGE_ERROR):
        fig = plt.figure("Doppler Measurment")
        ax = fig.add_subplot(211)
        ax.plot(time_vect, dopp_actual, linewidth = 4, color = "black", label = "Doppler Actual")
        ax.plot(time_vect, dopp_meas, 'o', color = "orange", label = "Doppler Measured")
        ax.set_xlim(min(time_vect), max(time_vect))
        ax.set_title("Doppler vs time")
        ax.set_ylabel("Doppler [Hz]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc = "upper right")
        ax.grid(True)
        
        ax = fig.add_subplot(212)
        ax.plot(time_vect, dopp_error, linewidth = 2, label = "Doppler Error")
        ax.set_xlim(min(time_vect), max(time_vect))
        ax.set_ylim(-100, 100)
        ax.set_title("Doppler Error")
        ax.set_ylabel("Doppler Error [m]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc = "upper right")
        ax.grid(True)
    
    if(PLAY_RUN):
        def update_image(td_idx):
            rti.set_data(10 * np.log10(abs(RTI[td_idx, :, :])))
            rdi.set_data(10 * np.log10(abs(RDI[td_idx, :, :])))
        
        RTE_E = [0, unamb_rng, 0, M]
        RDI_E = [0, unamb_rng, -PRF / 2 * 1e-3, PRF / 2 * 1e-3]
        
        fig = plt.figure("Fly in")
        ax = fig.add_subplot(211)
        rti = ax.imshow(10 * np.log10(abs(RTI[0, :, :])), vmin = -63, vmax = -30.0, origin = "lower", interpolation = "nearest", cmap = "jet", extent = RTE_E, aspect='auto', animated = True)
        ax.set_title("Range-Time Image")
        ax.set_xlabel("Range [m]")
        ax.set_ylabel("PRI [n]")
        
        ax = fig.add_subplot(212)
        rdi = ax.imshow(10 * np.log10(abs(RDI[0, :, :])), vmin = -53.0, vmax = -30.0, origin = "lower", interpolation = "nearest", cmap = "jet", extent = RDI_E, aspect='auto', animated = True)
        ax.set_title("Range-Doppler Image")
        ax.set_xlabel("Range [m]")
        ax.set_ylabel("Doppler Frequency [kHz]")
        
        ani = animation.FuncAnimation(fig, update_image, interval=0.01, frames = len(time_vect), repeat = True)
        plt.subplots_adjust(hspace = 0.6)
        plt.show()
 