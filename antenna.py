#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:20:09 2023

@author: smckenzie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from utilities import find_nearest

#%% Constants
N = 1000

#%% Default values
DEFAULT_LX = 1.0
DEFAULT_LY = 1.0
DEFAULT_CARRIER = 10e9
DEFAULT_HEIGHT = 10.0

#%% Antenna class object
class Antenna(object):
    """
    Object that holds the parameters of a radar antenna and produces 
    antenna pattern, peak antenna gain, and directional gain. 

    
    Attributes
    -------------
    Lx: Antenna dimension in the x direction 
    Ly: Antenna dimension in the y direction 
    Fc: Carrier Frequency 
    height: Antenna height from the ground 
    Lambda: Antenna wave length 
    """

    def __init__(self, Lx = DEFAULT_LX, Ly = DEFAULT_LY, Fc = DEFAULT_CARRIER, height = DEFAULT_HEIGHT):
        """
        Initialized the parameters of antenna to defualt parameters 
        
        Parameters
        -------------
        Lx: Antenna dimension in the x direction 
        Ly: Antenna dimension in the y direction 
        Fc: Carrier Frequency 
        height: Antenna height from the ground 
        """ 
        
        self.Lx = Lx
        self.Ly = Ly
        self.carr_freq = Fc
        self.Lambda = c / self.carr_freq
        self.height_antenna = height
    
    def set_aperture(self, Lx, Ly):
        """
        Sets the antenna dimension in the x and y directions 
        
        Parameters
        -------------
        Lx: Antenna dimension in the x direction 
        Ly: Antenna dimension in the y direction 
        """ 
        
        self.Lx = Lx
        self.Ly = Ly
        
    def set_carrier(self, Fc):
        """
        Sets the antenna carrier frequency 
        
        Parameters
        -------------
        Fc: Carrier Frequency 
        """ 
        
        self.carr_freq = Fc
        self.Lambda = c / self.carr_freq
        
    def set_antenna_height(self, height):
        """
        Sets the antenna height from the ground 
        
        Parameters
        -------------
        height: Antenna height from the ground 
        """ 
        
        self.height_antenna = height  
    
    def peak_gain(self, dB = True):
        """
        Returns the peak gain of the antenna 
        
        Parameters
        -------------
        dB: set true if the return gain is to be in dB 
        """ 

        if(dB):
            peak_gain = 10.0 * np.log10(4 * pi * self.Lx * self.Ly / (self.Lambda ** 2))
        else:
            peak_gain = 4 * pi * self.Lx * self.Ly / (self.Lambda ** 2)
        return peak_gain
    
    def get_pattern(self, dB = True, image = False):
        """
        Returns the antenna pattern 
        
        Parameters
        -------------
        dB: set true if the return pattern is to be in dB 
        image: set true to view the isometric antenna pattern 
        """ 
        
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        mesh_x, mesh_y = np.meshgrid(x, y)
        if(dB):
            LVL = [-50.0, 1.0]
            pattern = 20.0 * np.log10(np.abs(np.sinc(1 / pi * 2 * pi / self.Lambda * self.Lx / 2 * mesh_x) * np.sinc(1 / pi * 2 * pi / self.Lambda * self.Ly / 2 * mesh_y)))
        else:
            LVL = [0.05, 1.0]
            pattern = np.sinc(np.abs(1 / pi * 2 * pi / self.Lambda * self.Lx / 2 * mesh_x) * np.sinc(1 / pi * 2 * pi / self.Lambda * self.Ly / 2 * mesh_y))
        if(image):
            E = [np.arcsin(-1) * 180 / pi, np.arcsin(1) * 180 / pi, np.arcsin(-1) * 180 / pi, np.arcsin(1) * 180 / pi]
            plt.figure("Isometric Pattern")
            plt.imshow(pattern, vmin = LVL[0], vmax = LVL[1], interpolation = "spline36", cmap = "jet", extent = E)
            plt.xlabel(r"Theta $\theta$ [deg $\degree$]")
            plt.ylabel(r"Phi $\phi$ [deg $\degree$]")
            plt.grid(True)
            cbar = plt.colorbar(shrink = 1.0)
            cbar.ax.set_ylabel("[dBi]")
            plt.draw()
            plt.pause(0.01)  
        return pattern
    
    
    def directional_gain(self, target_range, target_x, target_y, dB = True, plot = False):
        """
        Returns the directional gain of the antenna 
        
        Parameters
        -------------
        target_range: The radial range of the target fromt the radar 
        target_x: The distance of the target from the antenna on the x dimension 
        target_y: The distance of the target from the antenna on the y dimension 
        dB: Set true if the return gain is to be in dB 
        """ 
        
        if((type(target_range) == float) or (type(target_range) == int)):
            nAngles = 1
        else:
            nAngles = len(target_range)
        phi = np.arcsin(np.array((target_y - self.height_antenna) / target_range)) * 180 / pi
        theta = np.arcsin(np.array(target_x / target_range)) * 180 / pi
        pattern = self.get_pattern(dB)
        peak_gain = self.peak_gain(dB)
        angle = np.arcsin(np.linspace(-1, 1, N)) * 180 / pi
        if(nAngles == 1):
            phi_idx = find_nearest(angle, phi)
            theta_idx = find_nearest(angle, theta)
        else:
            phi_idx = []
            for p in phi:
                phi_idx.append(find_nearest(angle, p))
            theta_idx = []
            for t in theta:
                theta_idx.append(find_nearest(angle, t))            
        if(dB):
            gain = peak_gain + pattern[phi_idx, theta_idx]
        else:
            gain = peak_gain * pattern[phi_idx, theta_idx]
        if(plot and (nAngles > 1)):
            plt.figure("Gain vs Range")
            plt.plot(target_range, gain)
            plt.xlim(target_range[nAngles - 1], target_range[0])
            plt.ylim(-30, 30)
            plt.xlabel("Range [m]")
            plt.ylabel("Gain [dB]")
            plt.grid(True)
            # plt.show(block = False)
            plt.draw()
            plt.pause(0.01)  
        return gain
