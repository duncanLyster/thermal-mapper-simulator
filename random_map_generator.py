import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import math

nside = 2048
npix = hp.nside2npix(nside)
T_max = 180.0 # K
T_min = 75.0 # K 

planetary_radius = 25200 # in m 

# Generate random temperatures to populate the map 
temperatures = np.random.uniform(T_min, T_max, npix)

# Save to the .fits file 
hp.write_map("random_map.fits", temperatures, coord='G', column_names=['TEMPERATURE'], column_units='K', overwrite=True)

# Calculate pixel size
pixel_area_sr = hp.nside2pixarea(nside)
pixel_area_m2 = pixel_area_sr * (planetary_radius ** 2)
pixel_width_m = math.sqrt(pixel_area_m2)

# Visualise the map 
hp.mollview(temperatures, title="Random Temperatures Map", unit="K", norm="hist", cmap="jet")
hp.graticule()

plt.figtext(0.15, 0.16, f"Approx Pixel Width: {pixel_width_m:.2f} m", ha='left', va='bottom', backgroundcolor='white')

plt.show()