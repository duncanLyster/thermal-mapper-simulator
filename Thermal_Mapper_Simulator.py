# IMPORTANT NOTE: THIS CODE HAS BEEN SUPERSEDED BY A JUPYTER NOTEBOOK 'THERMAL_INSTRUMENT_DIGITAL_TWIN'. 
# THIS VERSION HAS NOT BEEN UPDATED SINCE 30 OCTOBER 2023. 

# This code simulates infrared spectral observations for space missions, specifically designed for the Nightingale proposal
# for NASA's New Frontiers (NF5) opportunity. It reads spacecraft configuration from a JSON file and a thermal map in HEALPix format.

import math
import json
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.constants import h, c, k

# Identify HEALPix pixels within the spacecraft's field of view (FoV).
# The function accounts for spacecraft pointing and roll. It assumes a square FoV 
# It is only written for a square FoV and uses the approximation that FoV is small and far from s/c.
def pixels_in_fov(lat, lon, fov_deg, roll_deg, nside):
    # Convert latitude, longitude, FoV, and roll to radians
    theta, phi, half_fov_rad, roll_rad = np.radians([90 - lat, lon, fov_deg / 2.0, roll_deg])
    
    # Initialize rotation object for roll angle
    r = R.from_euler('z', roll_rad, degrees=False)
    
    # Initialize list to store pixels within FoV
    fov_pixels = []
    
    # Query pixels within a circular region as candidates for square FoV
    circle_radius = np.sqrt(2) * half_fov_rad
    candidate_pixels = hp.query_disc(nside, hp.ang2vec(theta, phi), circle_radius)
    
    for pixel in candidate_pixels:
        # Get angular coordinates of the candidate pixel
        theta_pixel, phi_pixel = hp.pix2ang(nside, pixel)
        
        # Calculate angular distance from spacecraft to pixel
        d_theta, d_phi = theta - theta_pixel, phi - phi_pixel
        
        # Apply roll rotation to the angular distances
        rotated_point = r.apply([d_theta, d_phi, 0])[:2]
        
        # Check if the rotated point falls within the square FoV
        if np.abs(rotated_point[0]) <= half_fov_rad and np.abs(rotated_point[1]) <= half_fov_rad:
            fov_pixels.append(pixel)

    fov_pixels = np.array(fov_pixels)

    # Use line to see actual pixel indices for debugging
    # print(f"Pixel indices within the rotated square FoV: {fov_pixels}")
    
    return fov_pixels

# Visualize pixels within the FoV on a new HEALPix map.
def show_fov_pixels(pixels, nside):
    # Initialize a new map with zeros
    fov_map = np.zeros(hp.nside2npix(nside))
    
    # Set the pixels within the FoV to 1
    fov_map[pixels] = 1.0
    
    # Display the map with overlaid latitude and longitude lines
    hp.mollview(fov_map, title="Field of View", unit="FoV", hold=True)
    hp.graticule()
    plt.show()

# Extract temperatures from the thermal map based on pixel indices within the FoV
def extract_temperatures(pixels, thermal_map):
    # Initialize an empty list to store temperatures
    temperatures = []
    
    # Loop through each pixel index to get its temperature from the thermal map
    for pixel in pixels:
        temperature = thermal_map[pixel]
        temperatures.append(temperature)
        
    return np.array(temperatures)

# Calculate blackbody radiance using Planck's Law
def planck(wavenumber, temperature):
    # Convert wavenumber to frequency
    wavenumber_si = wavenumber * 100
    frequency = wavenumber_si * c
    
    # Calculate the spectral radiance
    exponent = h * frequency / (k * temperature)
    radiance = (2 * h * frequency**3 / c**2) / (np.exp(exponent) - 1)

    # Convert radiance to units of nW cm^-2 str^-1 cm
    radiance *= 10**13
    
    return radiance

# Calculate composite blackbody curve - not producing good values currently 
def composite_blackbody_curve(temperatures, wavenumber_min, wavenumber_max, spectral_resolution):
    # Initialize wavenumber array based on instrument specs
    wavenumbers = np.arange(wavenumber_min, wavenumber_max + spectral_resolution, spectral_resolution)
    
    # Initialize array to store composite radiance values
    composite_radiance = np.zeros_like(wavenumbers)
    
    # Loop through each temperature to calculate its blackbody curve and add it to the composite curve
    for temperature in temperatures:
        radiance = planck(wavenumbers, temperature)
        composite_radiance += radiance / len(temperatures)
    
    return wavenumbers, composite_radiance

# FUNCTION
# Applies instrument sensitivity function to the blackbody curve.

# FUNCTION
# Applies an instrument noise function to the curve 

# Load spacecraft configuration from JSON
with open("Nightingale_Configuration.json", "r") as f:
    config = json.load(f)

# Read thermal map from the FITS file
thermal_map = hp.read_map("random_map.fits")

# Extract parameters from configuration
fov_deg = config["field_of_view"]["horizontal"]
lat = config["spacecraft_state"]["position"]["latitude"]
lon = config["spacecraft_state"]["position"]["longitude"]
roll_deg = config["spacecraft_state"]["pointing"]["roll"]
wavenumber_min = config["wavenumber_range"]["min"]
wavenumber_max = config["wavenumber_range"]["max"]
spectral_resolution = config["spectral_resolution"]
nside = 2048

# Identify pixels within FoV
pixels = pixels_in_fov(lat, lon, fov_deg, roll_deg, nside)

# Show the pixels within the FoV on a new map
show_fov_pixels(pixels, nside)

# Extract temperatures for the identified pixels
temperatures = extract_temperatures(pixels, thermal_map)

# Option to print out temperatures within FoV
# print(f"Temperatures within the FoV: {temperatures}")

# Calculate composite blackbody curve
wavenumbers, composite_radiance = composite_blackbody_curve(temperatures, wavenumber_min, wavenumber_max, spectral_resolution)

# Optionally, you can plot or further process the composite_radiance array
# For example, to plot:
plt.plot(wavenumbers, composite_radiance)
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Radiance (nW cm^-2 str^-1 cm)')
plt.show()