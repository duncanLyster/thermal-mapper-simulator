# -*- coding: utf-8 -*-
"""
This is an adapted version of the LTM SNR calculator for the TIRI instrument Written by Neil Bowles. 
It was adapted by Duncan Lyster in November 2023 to improve readbility in order to integrate with Thermal Instrument Digital Twin code. 

Original:
Created on Thu Aug 28 22:02:53 2014
Tweaked for LTM/TIRI/Icy Moons 3/5/22
@author: bowles

D* from NETD calculator for microbolometer arrays
Most are quoted at T=300K, f/1 and some detector size
"""

from numpy import exp, sqrt, loadtxt, tan
from matplotlib import pyplot as plt
from scipy import integrate, constants
from scipy.constants import h, c, Boltzmann
from tqdm import tqdm
import numpy as np

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
    C2 = (h * c) / Boltzmann

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
    C4 = (constants.h * constants.c) / constants.Boltzmann

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
    return half_angle, det_solid_angle, telescope_area, AOmega, scene_solid_angle, total_throughput

def orbit_calc (altitude):
    Orbital_velocity = sqrt ((constants.G*target_mass)/(target_radius+altitude))
    Orbital_period = (2*constants.pi*(target_radius+altitude))/Orbital_velocity
    Ground_IFOV = 2*orbital_altitude*tan(Half_angle)
    deltaf = Orbital_velocity/Ground_IFOV #Readout frequency in Hz
    return Orbital_velocity, Orbital_period, Ground_IFOV, deltaf

def plot_snr_vs_temperature(bandpass_results_array, raw_band_in, deltaf, Ground_IFOV):
    """
    This function plots the Signal to Noise ratio against the scene temperature for different bandpasses.

    Parameters:
    bandpass_results_array (list): A list of bandpass results.
    raw_band_in (list): A list of raw band inputs.
    deltaf (int): The nominal integration time in Hz.
    Ground_IFOV (int): The ground instantaneous field of view in meters.
    """

    # Set up the figure
    plt.figure(1)
    plt.rcParams.update({'font.size': 16})
    plt.xlabel('Scene Temperature (K)')
    plt.ylabel('Signal to Noise ratio')
    plt.title("SNR performance for multiple bandpasses at "+ str(int(deltaf)) + " Hz nominal integration time, IFOV=" + str(int(Ground_IFOV)) + " m")
    plt.grid(which='both')
    plt.yscale('log')

    # Plot the SNR for each bandpass
    for i in range(len(bandpass_results_array)):
        label_string = str(raw_band_in[i][0]) + " - " + str(raw_band_in[i][1])
        #print(label_string)
        plt.plot(
            bandpass_results_array[i].scene_temperature,
            bandpass_results_array[i].temperature_snr_array,
            label=label_string
        )

    plt.legend(loc='lower right')
    plt.show()

def plot_snr_for_emissivity_change(number_of_bands, raw_band_in, bandpass_results_array):
    """
    This function plots the Signal to Noise ratio against the scene temperature for different bandpasses,
    considering a 1% change in emissivity.

    Parameters:
    number_of_bands (int): The number of bands.
    raw_band_in (list): A list of raw band inputs.
    bandpass_results_array (list): A list of bandpass results.
    """

    # Set up the figure
    plt.figure(2)
    plt.grid(which='both')
    plt.ylim(bottom=0.0)
    plt.xlabel('Scene Temperature (K)')
    plt.ylabel('Signal to Noise ratio')
    plt.title("SNR for 1% change in emissivity for multiple bands")

    # Plot the SNR for each bandpass
    for i in range(number_of_bands):
        label_string = str(raw_band_in[i][0]) + " - " + str(raw_band_in[i][1])
        #print(label_string)
        plt.plot(
            bandpass_results_array[i].scene_temperature,
            bandpass_results_array[i].emissivity_snr_array,
            label=label_string
        )

    plt.legend(loc='lower right')
    plt.show()

def plot_detector_response(wavelength_array, absorption_array):
    """
    This function plots the interpolated response against the wavelength.

    Parameters:
    wavelength_array (list): A list of wavelengths.
    absorption_array (list): A list of absorption values.
    """

    # Set up the figure
    plt.figure(3)
    plt.xlabel('Wavelength um')
    plt.ylabel('Response')
    plt.title('Estimate of detector response from Oxford data')
    plt.grid(which='both')

    # Plot the interpolated response
    plt.plot(wavelength_array*1.0E6, absorption_array, label="Interpolated response")

    plt.legend(loc='upper right')
    plt.show()

def plot_nedt_vs_temperature(number_of_bands, raw_band_in, bandpass_results_array, deltaf, TDI_pixels, Ground_IFOV):
    """
    This function plots the Noise Equivalent Differential Temperature (NEDT) against the scene temperature for different bandpasses.

    Parameters:
    number_of_bands (int): The number of bands.
    raw_band_in (list): A list of raw band inputs.
    bandpass_results_array (list): A list of bandpass results.
    deltaf (float): The frequency difference.
    TDI_pixels (int): The number of TDI pixels.
    Ground_IFOV (float): The ground IFOV.
    """

    # Set up the figure
    plt.figure(4)
    plt.rcParams.update({'font.size': 12})
    plt.grid(which='both')
    plt.xlabel('Scene Temperature (K)')
    plt.ylabel('NEDT (K)')
    plt.yscale('log')
    plt.ylim(0.01,30.0)
    plt.title("NEDT performance for multiple bandpasses at "+ str (int(deltaf))+ " Hz nominal integration time TDI:"+str(int(TDI_pixels)) +" pixels, IFOV="+str(int(Ground_IFOV))+" m")

    # Plot the NEDT for each bandpass
    for i in range(number_of_bands):
        label_string = str(raw_band_in[i][0]) + " - " + str(raw_band_in[i][1])
        #print(label_string)
        plt.plot(
            bandpass_results_array[i].scene_temperature,
            bandpass_results_array[i].nedt_array,
            label=label_string
        )

    plt.legend(loc='lower left')
    plt.show()

def plot_nedt_vs_temperature_v2(number_of_bands, raw_band_in, bandpass_results_array, deltaf, Ground_IFOV):
    """
    This function plots the Noise Equivalent Differential Temperature (NEDT) against the scene temperature for different bandpasses.

    Parameters:
    number_of_bands (int): The number of bands.
    raw_band_in (list): A list of raw band inputs.
    bandpass_results_array (list): A list of bandpass results.
    deltaf (float): The frequency difference.
    Ground_IFOV (float): The ground IFOV.
    """

    # Set up the figure
    plt.figure(5)
    plt.grid(which='both')
    plt.xlabel('Scene Temperature (K)')
    plt.ylabel('NEDT (K)')
    plt.yscale('log')
    plt.title("NEDT performance for multiple bandpasses at "+ str (int(deltaf))+ " Hz nominal integration time, IFOV="+str(int(Ground_IFOV))+" m")

    # Plot the NEDT for each bandpass
    for i in range(number_of_bands):
        label_string = str(raw_band_in[i][0]) + " - " + str(raw_band_in[i][1])
        #print(label_string)
        plt.plot(
            bandpass_results_array[i].scene_temperature,
            bandpass_results_array[i].nedt_array,
            label=label_string
        )

    plt.legend(loc='lower right')
    plt.show()

def save_results_to_csv(detector_area, telescope_area, det_solid_angle, AOmega, Scene_solid_angle, Half_angle, Ground_IFOV, Orbital_velocity, deltaf, number_of_bands, bandpass_results_array, target_radius, orbital_altitude):
    """
    This function saves the results of calculations to CSV files.

    Parameters:
    detector_area (float): The detector area.
    telescope_area (float): The telescope area.
    det_solid_angle (float): The detector solid angle.
    AOmega (float): The AOmega.
    Scene_solid_angle (float): The scene solid angle.
    Half_angle (float): The half angle.
    Ground_IFOV (float): The ground track IFOV.
    Orbital_velocity (float): The orbital velocity.
    deltaf (float): The readout frequency.
    number_of_bands (int): The number of bands.
    bandpass_results_array (list): A list of bandpass results.
    target_radius (float): The target radius.
    orbital_altitude (float): The orbital altitude.
    """

    # Section to write out calculations to csv file.
    header_string = f"""Detector Area={detector_area}m^2
    Telescope area = {telescope_area}m^2
    Detector Solid Angle = {det_solid_angle}Sr
    AOmega = {AOmega}
    Scene Solid Angle = {Scene_solid_angle}Sr
    IFOV = {2*Half_angle*1E3} mrad
    Ground track IFOV = {Ground_IFOV} m
    Orbital Velocity = {Orbital_velocity} m/s
    Integration time for IFOV = {Ground_IFOV/Orbital_velocity} s
    Readout Frequency = {deltaf} Hz
    """

    # Setup the header info 
    #print ("Header:")
    #print (header_string)

    # Setup the output file names
    snr_file = "LTM_SNR_Calcs_TDI_microbolometer.csv"
    nedt_file = "LTM_NEDT_Calcs_TDI_microbolometer.csv"
    ner_file = "LTM_NER_calcs_TDI_microbolometer.csv"
    dbdt_file = "LTM_dbdt_calcs_TDI_microbolometer.csv"
    power_at_det_file = "LTM_Power_calcs_TDI_microbolometer.csv"

    snr_out =[]
    nedt_out=[]
    ner_out=[]
    dbdt_out= []
    filter_lower =[]
    filter_upper=[]
    power_array_out = []
    temperature_out=bandpass_results_array[0].scene_temperature

    # Loop over the bandpasses and write out the results to a csv file.
    for i in range (number_of_bands):
       snr_out1=bandpass_results_array[i].temperature_snr_array
       snr_out.append(snr_out1)

       nedt_out1 = bandpass_results_array[i].nedt_array
       nedt_out.append(nedt_out1)
       filter_lower1= bandpass_results_array[i].bandpass[0]
       filter_upper1 = bandpass_results_array[i].bandpass[1]

       filter_lower.append(filter_lower1)
       filter_upper.append(filter_upper1)

       ner_out1 = bandpass_results_array[i].ner_array
       ner_out.append(ner_out1)

       dbdt_out1 = bandpass_results_array[i].dBdT_array
       dbdt_out.append(dbdt_out1)

       power_array1 = bandpass_results_array[i].power_detector_array
       power_array_out.append(power_array1)

    snr_final = np.row_stack((temperature_out,snr_out))
    nedt_final = np.row_stack((temperature_out,nedt_out))
    NER_final = np.row_stack((temperature_out,ner_out))
    dbdt_final = np.row_stack((temperature_out,dbdt_out))
    power_final = np.row_stack((temperature_out,power_array_out))

    # Write out the results to csv files
    np.savetxt(snr_file,snr_final,delimiter =",",header=header_string)
    np.savetxt(nedt_file,nedt_final,delimiter =",",header=header_string)
    np.savetxt(ner_file,NER_final,delimiter=",",header=header_string)  
    np.savetxt(dbdt_file,dbdt_final, delimiter=",", header=header_string)
    np.savetxt(power_at_det_file,power_final,delimiter=",",header=header_string)  

    # Print out the orbital parameters
    #print ("Orbital Period =", ((2*constants.pi*(target_radius+orbital_altitude))/Orbital_velocity)/60.0, " mins")

class SNR_data_per_band:
    
    wavenumber_array =[]
    temperature_snr_array =[]
    emissivity_snr_array =[]
    scene_temperature = []
    bandpass = []
    nedt_array =[]
    ner_array=[]
    dBdT_array = [] #This is the derivative of the radiance with respect to temperature
    power_detector_array=[]
    
if __name__ == '__main__':

    #print ("Signal to Noise estimates for a specific D* and filter set for various input temperatures.")

    #WAVELENGTH RANGE
    lambda1 = 1.0E-6                #lower wavelength in m
    lambda2= 300.0E-6               #upper wavelength in m
    
    wavelength_array = np.arange(lambda1, lambda2, 1.0E-9) #Set the wavelength array with 1nm steps.
    
    absorption_array = np.full(wavelength_array.size,0.95) #Set the absorption to 0.95 for the moment.
    
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

    #DERIVED DETECTOR PARAMETERS
    Half_angle, det_solid_angle, telescope_area, AOmega, Scene_solid_angle, total_throughput = calculate_detector_parameters(det_fnumber, detector_area, telescope_diameter, detector_absorption)
    
    #DERIVED ORBITAL PARAMETERS
    Orbital_velocity, Orbital_period, Ground_IFOV, deltaf = orbit_calc (orbital_altitude)

    #Setup the header string for the output files.
    header_string = f"""Detector Area={detector_area}m^2
    Telescope area = {telescope_area}m^2
    Detector Solid Angle = {det_solid_angle}Sr
    AOmega = {AOmega}
    Scene Solid Angle = {Scene_solid_angle}Sr
    IFOV = {2*Half_angle*1E3} mrad
    Ground track IFOV = {Ground_IFOV} m
    Orbital Velocity = {Orbital_velocity} m/s
    Integration time for IFOV = {Ground_IFOV/Orbital_velocity} s
    Readout Frequency = {deltaf} Hz
    """

    #print(header_string)

    #Icy moon Temperatures
    temperature_array = np.arange(45.0,180.0,10.0)
    
    #Setup the arrays to hold the results
    snr_array = np.zeros(temperature_array.size)
    snr_delta_power = np.zeros(temperature_array.size)
    NER_array = np.zeros(temperature_array.size)
    NEP_array = np.zeros(temperature_array.size)
    NEDT_array=np.zeros(temperature_array.size)
    integrated_deriv_radiance_array = np.zeros(temperature_array.size)
    power_at_detector_array = np.zeros(temperature_array.size)
   
    #Load the bandpass file
    bandpass_file = "nightingale_bandpass.txt"              #This is the bandpass file for the Nightingale mission
    raw_band_in=loadtxt(bandpass_file,delimiter=',')        #load the bandpass file
    number_of_bands = len(raw_band_in)                      #determine the number of bands in the file
    
    #Calculate the NEP for the detector
    NEP = (sqrt(detector_area)*100.0)*sqrt(deltaf)/Dstar
    NEP = NEP/sqrt(TDI_pixels) #This is the NEP per pixel, so we need to scale it for the number of pixels in the TDI.
   
    #print ("Note that this code assumes an integration time matched to IFOV")
    #print ("NEP calculated to be ",NEP," W Hz^-1/2")
    
    bandpass_results_array = [] #create an array to hold the results for each bandpass
    total_throughput_array = np.full(wavelength_array.size, total_throughput) #create an array of the total throughput for each wavelength
    
    #Loop over the bandpasses and calculate the SNR for each bandpass
    for band_number in tqdm(range(number_of_bands), desc="Calculating SNR for each bandpass"):
        bandpass_array = raw_band_in[band_number]
        #print ("Band pass =",bandpass_array[0],bandpass_array[1])
        
        lower_index=np.where(wavelength_array>bandpass_array[0]*1E-6)[0][0]
        upper_index=np.where(wavelength_array>bandpass_array[1]*1E-6)[0][0]
            
        filter_wavelength_array = wavelength_array[lower_index:upper_index]
        filter_Tput_array = total_throughput_array[lower_index:upper_index]
        
        #print ("Lower Filter wavelength=",filter_wavelength_array[0])
        #print ("Upper Filter wavelength=", filter_wavelength_array[filter_wavelength_array.size-1])
        
        #Setup the arrays to hold the results
        bandpass_results = SNR_data_per_band () #create a new results object for each bandpass
        radiance_array = np.zeros(filter_wavelength_array.size)
        deriv_radiance_array = np.zeros(filter_wavelength_array.size)
        bandpass_results.temperature_snr_array=np.zeros(temperature_array.size)
        bandpass_results.emissivity_snr_array=np.zeros(temperature_array.size)
        bandpass_results.nedt_array=np.zeros(temperature_array.size)
        bandpass_results.ner_array=np.zeros(temperature_array.size)
        bandpass_results.dBdT_array=np.zeros(temperature_array.size)
        bandpass_results.power_detector_array=np.zeros(temperature_array.size)
        
        integrated_NEP =(sqrt(detector_area)*100)*sqrt(deltaf)/(Dstar*np.mean(filter_Tput_array))
        integrated_NEP = integrated_NEP/sqrt(TDI_pixels)
        
        # Loop over the temperatures and calculate the SNR for each temperature
        for temp_index in range(temperature_array.size):
            # Calculate radiance and its derivative for each wavelength in the bandpass
            for wavelength_index in range(filter_wavelength_array.size):
                temperature = temperature_array[temp_index]
                wavelength = filter_wavelength_array[wavelength_index]
                radiance_array[wavelength_index] = calculate_spectral_radiance(temperature, wavelength)
                deriv_radiance_array[wavelength_index] = calculate_spectral_radiance_derivative(temperature, wavelength)

            # Calculate the flux and power at the detector
            flux_at_detector = radiance_array * filter_Tput_array * det_solid_angle
            power_at_detector = flux_at_detector * detector_area

            # Integrate the power and the derivative of the radiance over the bandpass
            integrated_power_at_detector = integrate.simps(power_at_detector, filter_wavelength_array)
            integrated_deriv_radiance = integrate.simps(deriv_radiance_array, filter_wavelength_array)

            # Store the derivative of the radiance and the power at the detector for each temperature
            integrated_deriv_radiance_array[temp_index] = integrated_deriv_radiance
            power_at_detector_array[temp_index] = integrated_power_at_detector

            #print(f"Temperature= {temperature_array[temp_index]}, Power at the detector = {integrated_power_at_detector}")

            # Calculate the SNR for the detector
            snr_array[temp_index] = integrated_power_at_detector / NEP
            snr_delta_power[temp_index] = (integrated_power_at_detector - (0.99 * integrated_power_at_detector)) / NEP #This is the SNR for a 1% change in emissivity

            # Calculate the NEDT for the detector
            integrated_NER = integrated_NEP / AOmega 
            NEDT_array[temp_index] = integrated_NER / integrated_deriv_radiance
            NER_array[temp_index] = integrated_NER

            #print(f"Integrated NEP={integrated_NEP}, Integrated NER={integrated_NER}")
            #print(f"Temperature = {temperature_array[temp_index]}, NEDT={NEDT_array[temp_index]}")

            # Store results in bandpass_results
            bandpass_results.emissivity_snr_array[temp_index] = snr_delta_power[temp_index]
            bandpass_results.temperature_snr_array[temp_index] = snr_array[temp_index]
            bandpass_results.nedt_array[temp_index] = NEDT_array[temp_index]
            bandpass_results.ner_array[temp_index] = NER_array[temp_index]
            bandpass_results.dBdT_array[temp_index] = integrated_deriv_radiance_array[temp_index]
            bandpass_results.power_detector_array[temp_index] = power_at_detector_array[temp_index]
        
        #Store the results for each bandpass in an array
        bandpass_results.scene_temperature = temperature_array 
        bandpass_results.wavenumber_array=filter_wavelength_array
        bandpass_results.bandpass = bandpass_array
        bandpass_results_array.append(bandpass_results)

    Desired_groundtrack= 100.0 #this is the estimate for averaging pixels
        
    number_of_pixels = int (Desired_groundtrack/Ground_IFOV)
        
    #print ("Number of pixels for grond track of ",Desired_groundtrack," m = ",number_of_pixels)

    #Plot the results
    #FIGURE 1 - SNR vs temperature
    #plot_snr_vs_temperature(bandpass_results_array, raw_band_in, deltaf, Ground_IFOV)
    
    #FIGURE 2 - SNR vs emissivity
    #plot_snr_for_emissivity_change(number_of_bands, raw_band_in, bandpass_results_array)
    
    #FIGURE 3 - Plot the detector response vs wavelength
    #plot_detector_response(wavelength_array, absorption_array)
    
    #FIGURE 4 - NEDT vs temperature for multiple bandpasses at different pixel sizes
    #plot_nedt_vs_temperature(number_of_bands, raw_band_in, bandpass_results_array, deltaf, TDI_pixels, Ground_IFOV)
   
    #FIGURE 5 - NEDT vs temperature
    #plot_nedt_vs_temperature_v2(number_of_bands, raw_band_in, bandpass_results_array, deltaf, Ground_IFOV)
    
    #Save the results to a CSV file
    #save_results_to_csv(detector_area, telescope_area, det_solid_angle, AOmega, Scene_solid_angle, Half_angle, Ground_IFOV, Orbital_velocity, deltaf, number_of_bands, bandpass_results_array, target_radius, orbital_altitude)
    
    print ("Completed calculations")

print ("Finished")