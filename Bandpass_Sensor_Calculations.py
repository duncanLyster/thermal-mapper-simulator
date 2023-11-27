# -*- coding: utf-8 -*-
""" This module is a refactored version of the LTM SNR calculator for the TIRI instrument, originally written by Neil Bowles and adapted by Duncan Lyster in November 2023. The module has been restructured to function as a callable module, allowing it to be integrated with the Thermal Instrument Model code. The module provides functions to calculate various parameters related to signal-to-noise ratio (SNR) and detector performance for microbolometer arrays. The original code has been significantly modified to improve readability and modularity, while maintaining the core functionality.

Original: Created on Thu Aug 28 22:02:53 2014 Tweaked for LTM/TIRI/Icy Moons 3/5/22 @author: bowles """

import numpy as np
import json

from numpy import exp, sqrt, loadtxt, tan
from scipy import integrate, constants
from scipy.constants import h, c, k
from tqdm import tqdm

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
    
class MissionConfig:
    def __init__(self, config):
        self.instrument_name = config["instrument_name"]

        self.wavenumber_min = config["detector_properties"]["wavenumber_min"]
        self.wavenumber_max = config["detector_properties"]["wavenumber_max"]

        self.spectral_resolution = config["detector_properties"]["spectral_resolution"]
        self.Dstar = config["detector_properties"]["Dstar"]
        self.detector_absorption = config["detector_properties"]["detector_absorption"]
        self.detector_fnumber = config["detector_properties"]["detector_fnumber"]
        self.detector_side_length = config["detector_properties"]["detector_side_length"]
        self.telescope_diameter = config["detector_properties"]["telescope_diameter"]
        self.fov_horizontal_deg = config["detector_properties"]["fov_horizontal_deg"]
        self.fov_vertical_deg = config["detector_properties"]["fov_vertical_deg"]

        self.latitude_deg = config["orbital_properties"]["latitude_deg"]
        self.longitude_deg = config["orbital_properties"]["longitude_deg"]
        self.altitude_m = config["orbital_properties"]["altitude_m"]
        self.pointing_azimuth_deg = config["orbital_properties"]["pointing_azimuth_deg"]
        self.pointing_elevation_deg = config["orbital_properties"]["pointing_elevation_deg"]
        self.roll_deg = config["orbital_properties"]["roll_deg"]
        self.target_mass_kg = config["orbital_properties"]["target_mass_kg"]
        self.target_radius_m = config["orbital_properties"]["target_radius_m"]

        self.bandpasses = config['bandpasses']

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
    
def calculate_detector_parameters(det_fnumber, detector_side_length, telescope_diameter, detector_absorption):
    half_angle = np.arctan(1.0 / (2 * det_fnumber))
    det_solid_angle = 2.0 * constants.pi * (1 - np.cos(half_angle))
    telescope_area = constants.pi * (telescope_diameter / 2.0) ** 2
    detector_area = detector_side_length * detector_side_length
    AOmega = detector_area * det_solid_angle
    scene_solid_angle = AOmega / telescope_area
    half_angle = np.arccos(1 - (scene_solid_angle / (2 * constants.pi)))
    total_throughput = 0.8*0.98*0.98*0.98*0.98*0.98*0.98*detector_absorption #estimate of window./filter plus 6 off 98% mirrors. DOES ETM HAVE A WINDOW??
    return half_angle, det_solid_angle, AOmega, total_throughput, detector_area

def deltaf_calc (orbital_altitude, target_mass, target_radius, Half_angle):
    Orbital_velocity = sqrt ((constants.G*target_mass)/(target_radius+orbital_altitude))
    Ground_IFOV = 2*orbital_altitude*tan(Half_angle)
    deltaf = Orbital_velocity/Ground_IFOV #Readout frequency in Hz
    return deltaf

def calculate_snr(mission_config, temperature_array):
    """
    This function calculates the SNR for a given set of parameters. BUG: Cannot currently deal with correct wavenumber input values for ETM. Numbers in main function are chosen to ensure this runs but they are not right. 
    """

    #TDI pixels - this needs to be changed so it takes this input for each bandpass
    TDI_pixels = 16

    #Convert the wavenumber range in cm^-1 to a wavelength range in m
    wavelength_min = 1.0E-2 / mission_config.wavenumber_max
    wavelength_max = 1.0E-2 / mission_config.wavenumber_min
    
    wavelength_array = np.arange(wavelength_min, wavelength_max, 1.0E-9) #Set the wavelength array with 1nm steps.
    
    #DERIVED DETECTOR PARAMETERS
    Half_angle, detector_solid_angle, AOmega, total_throughput, detector_area = calculate_detector_parameters(mission_config.detector_fnumber, 
    mission_config.detector_side_length, mission_config.telescope_diameter, mission_config.detector_absorption)
    
    #DERIVED ORBITAL PARAMETERS
    deltaf = deltaf_calc (mission_config.altitude_m, mission_config.target_mass_kg, mission_config.target_radius_m, Half_angle)

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
    noise_eq_power = (sqrt(detector_area)*100.0)*sqrt(deltaf)/mission_config.Dstar
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
        
        integrated_NEP = (sqrt(detector_area)*100)*sqrt(deltaf)/(mission_config.Dstar*np.mean(filter_throughput_array))
        integrated_NEP = integrated_NEP / sqrt(TDI_pixels)
        integrated_NER = integrated_NEP / AOmega # This is how it was calculated in Neil's code, but his slides say it should be radiance divided by SNR instead.

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
            flux_at_detector = radiance_array * filter_throughput_array * detector_solid_angle
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

if __name__ == "__main__":

    # Load spacecraft configuration from JSON
    with open("Nightingale_Configuration.json", "r") as f:
        config_file = json.load(f)

    # Extract parameters from configuration TO DO: Creat an object to store these parameters and make this a function
    mission_config = MissionConfig(config_file)

    #Load the temperature array from the test tile
    temperatures = np.loadtxt("test_tile.csv", delimiter=',', encoding='utf-8-sig')

    # Flatten the 2D temperatures array into a 1D array
    temperature_array = temperatures.flatten()

    bandpass_results_array = calculate_snr(mission_config, temperature_array)
    
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