# This code generates a HEALPix thermal map of Enceladus based on geometrically accurate location information
# for the 'tiger stripe' fractures at the South Pole. 

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sklearn.neighbors import KDTree
import csv 

# Calculate pixel size
def pixel_size(nside, planetary_radius):
    pixel_area_sr = hp.nside2pixarea(nside)
    pixel_area_m2 = pixel_area_sr * (planetary_radius ** 2)
    pixel_width_m = math.sqrt(pixel_area_m2)

    return pixel_width_m

# Populate the map with the background temperature 
def background_temp(nside, T_background):
    temperatures = np.full(hp.nside2npix(nside),T_background)

    return temperatures

# Read in coordinates and return list of lists of pixel indices they correspond to. 
def read_and_convert_coordinates(filename, nside):
    pixel_indices = [[] for _ in range(5)]
    
    with open(filename, "r") as f:
        for line in f:
            coords = line.strip().split(",")
            coords = [float(x) for x in coords]
            
            for i in range(5):
                lon = coords[i * 2]
                lon = 360 - lon
                lat = coords[i * 2 + 1]
                
                if lat == 0 and lon == 360:
                    continue
                
                theta = np.radians(90 - lat)
                phi = np.radians(lon)
                
                try:
                    pixel_index = hp.ang2pix(nside, theta, phi)
                except ValueError as e:
                    print(f"Skipping due to error: {e}")
                    continue
                
                pixel_indices[i].append(pixel_index)
    
    return pixel_indices

# Re-define pixel values based on a function of distance from the nearest fracture pixel
def thermal_spread(temperatures, fracture_pixel_indices, planetary_radius):
    nside = hp.npix2nside(len(temperatures))
    new_temperatures = np.copy(temperatures)
    
    # Convert pixel indices to angular coordinates and then to Cartesian coordinates
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    x, y, z = hp.ang2vec(theta, phi).T
    
    # Do the same for fracture pixels
    theta_f, phi_f = hp.pix2ang(nside, fracture_pixel_indices)
    x_f, y_f, z_f = hp.ang2vec(theta_f, phi_f).T
    
    # Create a KD-Tree from fracture pixels
    tree = KDTree(np.column_stack([x_f, y_f, z_f]))
    
    # Query this tree to find distances to the nearest fracture pixel for each pixel
    distances, _ = tree.query(np.column_stack([x, y, z]), k=1)
    
    # Convert angular distances to physical distances in meters
    physical_distances = 2 * planetary_radius * np.sin(distances / 2)

    # Apply the 1/x temperature scaling with tqdm for progress tracking
    for i in tqdm(range(len(new_temperatures)), desc="Processing pixels"):
        distance = physical_distances[i][0]  # Extract the scalar value
        if distance == 0:
            new_temperatures[i] = np.random.normal(270, 100)  # Mean 220K, Std Dev 70K
        elif distance < 1000:  # Less than 1 km
            # new_temperatures[i] = 20.15913 + ((176.9095 - 20.15913) / (1 + (distance / 97.6375) ** 0.3843642)) + np.random.normal(0, 60 * distance / 1000) # TO DO - have this scale depending on source temp/use physical conduction equation 
            new_temperatures[i] = 20.15913 + ((176.9095 - 20.15913) / (1 + (distance / 97.6375) ** 0.3843642)) + np.random.normal(0, 60 * distance / 1000) # TO DO - have this scale depending on source temp/use physical conduction equation 
        else:
            new_temperatures[i] = np.random.normal(65, 5)  # Mean 65K, Std Dev 10K

        if new_temperatures[i] < 0:
            new_temperatures[i] = 0

    return new_temperatures

# Apply Gaussian blur to map to get more realistic temperature zones 
def map_blur(map, sigma_degrees):

    sigma_radians = np.radians(sigma_degrees)
    blurred_map = hp.smoothing(map, sigma=sigma_radians)
    return blurred_map

nside = 2048
npix = hp.nside2npix(nside)
T_background = 65 # K
T_fracture = 180 # K 
planetary_radius = 25200 # in m 
temperatures = background_temp(nside, T_background)

# Read in tigerstripe data and convert to pixel indices 
filename = "tigerstripe_length_east.txt"
fracture_pixel_indices = read_and_convert_coordinates(filename, nside) # Now pixel_indices contains pixels corresponding to coordinates

# Re-calculate pixel values for entire map based on function of proximity to the nearest pixel in pixel_indices
fracture_pixel_indices = [index for sublist in fracture_pixel_indices for index in sublist]
temperatures = thermal_spread(temperatures, fracture_pixel_indices, planetary_radius)

# Create a noise map with the same shape as temperatures
noise_mean = 0
noise_std_dev = 25  # You can adjust this value
noise_map = np.random.normal(noise_mean, noise_std_dev, temperatures.shape)

# Apply Gaussian blur to map to get more realistic temperature zones 
noise_map = map_blur(noise_map, sigma_degrees=2)

# Cast temperatures to float64
temperatures = temperatures.astype(np.float64)

# Add noise to fracture temps map 
temperatures += noise_map

# Apply smoothing to the output
temperatures = map_blur(temperatures, sigma_degrees=0.1)

# Save to the .fits file 
hp.write_map("fracture_map.fits", temperatures, coord='G', column_names=['TEMPERATURE'], column_units='K', overwrite=True)

# Visualise the map 
# hp.mollview(temperatures, title="Fracture Based Temperatures Map", unit="K", norm="hist", cmap="jet", rot=(0, -90, 0), xsize=1600)
# hp.gnomview(temperatures, title="Fracture Based Temperatures Map", unit="K", rot=(0, -90), reso=10, xsize=800, ysize=800, norm="hist", cmap="jet")
hp.orthview(temperatures, title="Fracture Based Temperatures Map", unit="K", rot=(0, -90, 0), half_sky=True, norm="linear", cmap="jet")
hp.graticule(dpar=10, dmer=30, verbose=False)

pix_width = pixel_size(nside, planetary_radius)

plt.figtext(0.15, 0.16, f"Approx Pixel Width: {pix_width:.2f} m", ha='left', va='bottom', backgroundcolor='white')

# Show the map 
plt.show()