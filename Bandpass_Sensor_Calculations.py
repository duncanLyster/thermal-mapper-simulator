# -*- coding: utf-8 -*-
""" This module is a refactored version of the LTM SNR calculator for the TIRI instrument, originally written by Neil Bowles and adapted by Duncan Lyster in November 2023. The module has been restructured to function as a callable module, allowing it to be integrated with the Thermal Instrument Model code. The module provides functions to calculate various parameters related to signal-to-noise ratio (SNR) and detector performance for microbolometer arrays. The original code has been significantly modified to improve readability and modularity, while maintaining the core functionality.

Original: Created on Thu Aug 28 22:02:53 2014 Tweaked for LTM/TIRI/Icy Moons 3/5/22 @author: bowles """

import numpy as np

from numpy import exp, sqrt, loadtxt, tan
from scipy import integrate, constants
from scipy.constants import h, c, k
from tqdm import tqdm

def calculate_spectral_radiance(temp, wavelength):
    """
    Calculate and return the spectral radiance for a given wavelength and temperature.

    Parameters:
    temp (float): The temperature in Kelvin.
    wavelength (float): The wavelength in meters.

    Returns:
    float: The spectral radiance in spectral units of m.
    """
    # Radiation constants
    C1 = 2 * h * (c ** 2)
    C2 = (h * c) / k

    # Calculate radiance using Planck's law
    exponent = C2 / (wavelength * temp)
    radiance = C1 / ((wavelength ** 5) * (np.exp(exponent) - 1))

    return radiance

def calculate_spectral_radiance_derivative(temp, wavelength):
    """
    Returns a dB/dT value for a particular temperature.
    Note that the wavelength is in meters!
    'exitance' i.e. irradiance
    """

    # Radiation constants
    C3 = constants.pi * constants.h * (constants.c ** 2)
    C4 = (constants.h * constants.c) / constants.k

    radiance1 = C3 * C4 * exp(C4 / (temp * wavelength))
    radiance2 = ((temp ** 2) * (wavelength ** 6)) * ((exp(C4 / (temp * wavelength)) - 1) ** 2)

    radiance_deriv = radiance1 / radiance2

    return radiance_deriv
    
def calculate_detector_parameters(det_fnumber, detector_area, telescope_diameter, detector_absorption):
    half_angle = np.arctan(1.0 / (2 * det_fnumber))
    det_solid_angle = 2.0 * constants.pi * (1 - np.cos(half_angle))
    telescope_area = constants.pi * (telescope_diameter / 2.0) ** 2
    AOmega = detector_area * det_solid_angle
    scene_solid_angle = AOmega / telescope_area
    half_angle = np.arccos(1 - (scene_solid_angle / (2 * constants.pi)))
    total_throughput = 0.8*0.98*0.98*0.98*0.98*0.98*0.98*detector_absorption #estimate of window./filter plus 6 off 98% mirrors. DOES ETM HAVE A WINDOW??
    return half_angle, det_solid_angle, AOmega, total_throughput

def deltaf_calc (orbital_altitude, target_mass, target_radius, Half_angle):
    Orbital_velocity = sqrt ((constants.G*target_mass)/(target_radius+orbital_altitude))
    Ground_IFOV = 2*orbital_altitude*tan(Half_angle)
    deltaf = Orbital_velocity/Ground_IFOV #Readout frequency in Hz
    return deltaf

class SNR_data_per_band:
    
    wavenumber_array = []
    scene_temperatures_array = []
    bandpass_array = []
    temperature_snr = 0.0
    emissivity_snr = 0.0
    nedt = 0.0
    ner = 0.0
    dBdT = 0.0
    power_detector = 0.0
    
def calculate_snr(wavenumber_min, wavenumber_max, detector_area, telescope_diameter, detector_absorption, det_fnumber, orbital_altitude, target_mass, target_radius, temperature_array, Dstar, TDI_pixels):
    """
    This function calculates the SNR for a given set of parameters. BUG: Cannot currently deal with correct wavenumber input values for ETM. Numbers in main function are chosen to ensure this runs but they are not right. 
    """

    #Convert the wavenumber range in cm^-1 to a wavelength range in m
    lambda1 = 1.0E-2 / wavenumber_max
    lambda2 = 1.0E-2 / wavenumber_min
    
    wavelength_array = np.arange(lambda1, lambda2, 1.0E-9) #Set the wavelength array with 1nm steps.
    
    #DERIVED DETECTOR PARAMETERS
    Half_angle, det_solid_angle, AOmega, total_throughput = calculate_detector_parameters(det_fnumber, 
    detector_area, telescope_diameter, detector_absorption)
    
    #DERIVED ORBITAL PARAMETERS
    deltaf = deltaf_calc (orbital_altitude, target_mass, target_radius, Half_angle)

    #Setup the arrays to hold the results
    snr = 0.0
    snr_delta_power = 0.0
    integrated_deriv_radiance = 0.0
    integrated_power_at_detector = 0.0
   
    #Load the bandpass file
    bandpass_file = "nightingale_bandpass.txt"              #This is the bandpass file for the Nightingale mission
    raw_band_in=loadtxt(bandpass_file,delimiter=',')        #load the bandpass file
    number_of_bands = len(raw_band_in)                      #determine the number of bands in the file
    
    #Calculate the NEP for the detector
    noise_eq_power = (sqrt(detector_area)*100.0)*sqrt(deltaf)/Dstar
    noise_eq_power = noise_eq_power/sqrt(TDI_pixels) #This is the NEP per pixel, so we need to scale it for the number of pixels in the TDI.

    bandpass_results_array = [] #create an array to hold the results for each bandpass
    total_throughput_array = np.full(wavelength_array.size, total_throughput) #create an array of the total throughput for each wavelength
    
    #Loop over the bandpasses and calculate the SNR for each bandpass
    for band_number in tqdm(range(number_of_bands), desc="Calculating SNR for each bandpass"):
        bandpass_array = raw_band_in[band_number]
        
        lower_index=np.where(wavelength_array>bandpass_array[0]*1E-6)[0][0] #convert the bandpass wavelengths from microns to m and find the index of the first wavelength in the bandpass
        upper_index=np.where(wavelength_array>bandpass_array[1]*1E-6)[0][0] #convert the bandpass wavelengths from microns to m and find the index of the last wavelength in the bandpass
            
        filter_wavelength_array = wavelength_array[lower_index:upper_index]         #create an array of the wavelengths in the bandpass
        filter_throughput_array = total_throughput_array[lower_index:upper_index]   #create an array of the total throughput for each wavelength in the bandpass
        
        #Setup the arrays to hold the results
        bandpass_results = SNR_data_per_band () #create a new results object for each bandpass
        radiance_array = np.zeros(filter_wavelength_array.size)
        deriv_radiance_array = np.zeros(filter_wavelength_array.size)
        bandpass_results.temperature_snr = 0.0
        bandpass_results.emissivity_snr = 0.0
        bandpass_results.nedt = 0.0
        bandpass_results.ner = 0.0
        bandpass_results.dBdT = 0.0
        bandpass_results.power_detector = 0.0
        
        integrated_NEP = (sqrt(detector_area)*100)*sqrt(deltaf)/(Dstar*np.mean(filter_throughput_array))
        integrated_NEP = integrated_NEP / sqrt(TDI_pixels)
        integrated_NER = integrated_NEP / AOmega 

        # Initialize accumulators
        total_power_at_detector = 0.0
        total_deriv_radiance = 0.0
        
        # Loop over the temperatures and calculate the SNR for each temperature in the temperature array
        for temp_index in range(temperature_array.size):
            # Calculate radiance and its derivative for each wavelength in the bandpass for the current temperature
            for wavelength_index in range(filter_wavelength_array.size):
                temperature = temperature_array[temp_index]
                wavelength = filter_wavelength_array[wavelength_index]
                radiance_array[wavelength_index] = calculate_spectral_radiance(temperature, wavelength)
                deriv_radiance_array[wavelength_index] = calculate_spectral_radiance_derivative(temperature, wavelength)

            # Calculate the flux and power at the detector
            flux_at_detector = radiance_array * filter_throughput_array * det_solid_angle
            power_at_detector = flux_at_detector * detector_area

            # Integrate the power and the derivative of the radiance over the bandpass scaling by number of temperatures in fov
            integrated_power_at_detector = integrate.simps(power_at_detector, filter_wavelength_array) / temperature_array.size
            integrated_deriv_radiance = integrate.simps(deriv_radiance_array, filter_wavelength_array) / temperature_array.size

            total_power_at_detector += integrated_power_at_detector
            total_deriv_radiance += integrated_deriv_radiance

        # Calculate the SNR for the detector
        snr = total_power_at_detector / noise_eq_power
        snr_delta_power = (total_power_at_detector - (0.99 * total_power_at_detector)) / noise_eq_power #This is the SNR for a 1% change in emissivity

        # Calculate the NEDT for the detector
        noise_eq_delta_T = integrated_NER / total_deriv_radiance

        # Store results in bandpass_results
        bandpass_results.emissivity_snr = snr_delta_power
        bandpass_results.temperature_snr = snr
        bandpass_results.nedt = noise_eq_delta_T
        bandpass_results.ner = integrated_NER
        bandpass_results.dBdT = total_deriv_radiance
        bandpass_results.power_detector = total_power_at_detector
    
        #Store the results for each bandpass in an array
        bandpass_results.scene_temperatures_array = temperature_array 
        bandpass_results.wavenumber_array=filter_wavelength_array
        bandpass_results.bandpass_array = bandpass_array
        bandpass_results_array.append(bandpass_results)

    return bandpass_results_array

'''
if __name__ == "__main__":

    #WAVENUMBER RANGE chosen to give lambda range of 1-300 microns
    wavenumber_min = 3.33E1
    wavenumber_max = 1.0E4

    #DETECTOR PARAMETERS
    Dstar = 1E9                     #This is the broadband D*, assumed constant for the detector. In m Hz^1/2 W^-1
    TDI_pixels = 16                 #Assume 16 pixel TDI average, actual number depends on number of filters.
    detector_absorption = 0.95      #Sets absorption value for the detector material. This reduces the effective signal level seen by the detector.
    det_fnumber = 1.4               #f number of the detector
    detector_area = (35.0E-6)**2    #area of detector element in m (Detector element size, INO = 35x35 µm )
    telescope_diameter = 0.05       #LTM 5cm telescope  Diviner =4 cm

    #ORBITAL PARAMETERS
    orbital_altitude = 100.0E3      #Distance to the target body
    target_mass = 7.34767309E22     #Mass of the target body in kg
    target_radius = 1737E3          #radius of target body in m

    #Load the temperature array from the test tile
    temperatures = np.loadtxt("test_tile.csv", delimiter=',', encoding='utf-8-sig')

    # Flatten the 2D temperatures array into a 1D array
    temperature_array = temperatures.flatten()

    bandpass_results_array = calculate_snr(wavenumber_min, wavenumber_max, detector_area, telescope_diameter, detector_absorption, det_fnumber, orbital_altitude, target_mass, target_radius, temperature_array, Dstar, TDI_pixels)
    
    #Print the results
    for bandpass_results in bandpass_results_array:
        print("Bandpass: ", bandpass_results.bandpass_array)
        print("Temperature SNR: ", bandpass_results.temperature_snr)
        print("Emissivity SNR: ", bandpass_results.emissivity_snr)
        print("NEDT: ", bandpass_results.nedt)
        print("NER: ", bandpass_results.ner)
        print("dB/dT: ", bandpass_results.dBdT)
        print("Power at detector: ", bandpass_results.power_detector)

    print("Done!")
'''