import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
import scipy.signal as signal
import time

TARGET_TIC = "TIC 261136679"
print(f"--> [ARSIV MADENCILIGI]: {TARGET_TIC} TESS CVZ Hedefi Taraniyor...")

search = lk.search_targetpixelfile(TARGET_TIC, mission="TESS")
tpf = search[0].download(quality_bitmask="default")

time_arr = tpf.time.value
flux_cube = tpf.flux.value
valid = ~np.isnan(flux_cube).any(axis=(1, 2)) & ~np.isnan(time_arr)
time_arr, flux_cube = time_arr[valid], flux_cube[valid]
T_len, H, W = flux_cube.shape

ra, dec = tpf.ra, tpf.dec
query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.03))
ORDER BY phot_g_mean_mag ASC
"""
gaia_stars = Gaia.launch_job(query).get_results()
sky_coords = SkyCoord(gaia_stars["ra"], gaia_stars["dec"], unit="deg")
pix_x, pix_y = tpf.wcs.world_to_pixel(sky_coords)

in_bounds = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H)
pix_x, pix_y = pix_x[in_bounds], pix_y[in_bounds]
K_stars = len(pix_x)
print(f"--> Gorus Alani Icindeki Ayristirilacak Gaia Yildizi: {K_stars}")

sigma_psf = 0.85
grid_y, grid_x = np.mgrid[0:H, 0:W]
A = np.zeros((H * W, K_stars + 1))
for k in range(K_stars):
    psf_k = np.exp(-((grid_x - pix_x[k])**2 + (grid_y - pix_y[k])**2) / (2 * sigma_psf**2))
    psf_k /= np.sum(psf_k)
    A[:, k] = psf_k.flatten()
A[:, -1] = 1.0

flat_I = flux_cube.reshape(T_len, H * W).T
clean_target_flux = np.dot(np.linalg.pinv(A), flat_I)[0, :]
clean_norm = clean_target_flux / np.median(clean_target_flux)

trend = signal.medfilt(clean_norm, kernel_size=101)
detrended_flux = clean_norm / trend

bls = BoxLeastSquares(time_arr, detrended_flux)
periods = np.linspace(0.5, 15.0, 8000)
durations = np.linspace(0.04, 0.18, 12)
periodogram = bls.power(periods, durations)

best_idx = np.argmax(periodogram.power)
best_period = periodogram.period[best_idx]
best_depth = periodogram.depth[best_idx]
snr = periodogram.power[best_idx] / np.std(periodogram.power)

print("\n=========================================================================")
print("                   TRANSIT KESIF VE SINYAL RAPORU                        ")
print("=========================================================================")
print(f"--> Tespit Edilen En Guclu Periyot : {best_period:.4f} Gun")
print(f"--> Transit Derinligi              : {best_depth * 1e6:.1f} ppm (%{(best_depth)*100:.4f})")
print(f"--> Sinyal-Gurultu Orani (SNR)     : {snr:.1f} sigma")
print("=========================================================================")
