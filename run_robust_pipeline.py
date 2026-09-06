import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
import time

print("--> Donanim: Python / SciPy Analitik Matris Motoru (Monster Abra A5)")
TARGET_TIC = "TIC 261136679"

print(f"\n1. TESS CVZ Verisi Yukleniyor: {TARGET_TIC}...")
search = lk.search_targetpixelfile(TARGET_TIC, mission="TESS")
tpf = search[0].download(quality_bitmask="default")

time_arr = tpf.time.value
flux_cube = tpf.flux.value # (T, Y, X)

# Temiz maske
valid_mask = ~np.isnan(flux_cube).any(axis=(1, 2)) & ~np.isnan(time_arr)
time_arr = time_arr[valid_mask]
flux_cube = flux_cube[valid_mask]
T_len, H, W = flux_cube.shape

# Ham SAP Akisi
raw_sap = np.sum(flux_cube, axis=(1, 2))
raw_norm = raw_sap / np.median(raw_sap)

# 2. Gaia Yildizlarinin TESS Piksel Matrisine Izdusumu
ra, dec = tpf.ra, tpf.dec
query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag 
FROM gaiadr3.gaia_source 
WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.03))
ORDER BY phot_g_mean_mag ASC
"""
job = Gaia.launch_job(query)
gaia_stars = job.get_results()

sky_coords = SkyCoord(gaia_stars["ra"], gaia_stars["dec"], unit="deg")
pix_x, pix_y = tpf.wcs.world_to_pixel(sky_coords)

# Sadece goruntu alani icine dusen Gaia yildizlarini al
in_bounds = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H)
pix_x, pix_y = pix_x[in_bounds], pix_y[in_bounds]
g_mags = gaia_stars["phot_g_mean_mag"][in_bounds]
K_stars = len(pix_x)

print(f"--> Piksel Alani Icindeki Kesin Gaia Kaynak Sayisi: {K_stars}")

# 3. Analitik 2D-Gaussian PSF Tasarim Matrisi Olustur
sigma_psf = 0.85
grid_y, grid_x = np.mgrid[0:H, 0:W]

# Design Matrix A: (Num_pixels x (K_stars + 1_background))
A = np.zeros((H * W, K_stars + 1))
for k in range(K_stars):
    psf_k = np.exp(-((grid_x - pix_x[k])**2 + (grid_y - pix_y[k])**2) / (2 * sigma_psf**2))
    psf_k /= np.sum(psf_k)
    A[:, k] = psf_k.flatten()
A[:, -1] = 1.0 # Duzgun arkaplan sutunu

print("\n2. Analitik PSF Dekonvolusyonu Calistiriliyor...")
t0 = time.time()

# Her zaman adiminda dogrusal sistem cozumu: I_t = A * F_t
flat_I = flux_cube.reshape(T_len, H * W).T # (H*W, T)
ATA_inv_AT = np.linalg.pinv(A)
F_solutions = np.dot(ATA_inv_AT, flat_I) # (K_stars + 1, T)

# 0. indeks: TIC 261136679'un saf akisi
target_clean_flux = F_solutions[0, :]
clean_norm = target_clean_flux / np.median(target_clean_flux)

print(f"--> Analitik ayrastirma {time.time() - t0:.2f} saniyede tamamlandi.")

# 4. Gurultu (CDPP) Olcumu
def calc_scatter(flux_series):
    diff = np.diff(flux_series)
    return (np.std(diff) / np.sqrt(2)) * 1e6

raw_ppm = calc_scatter(raw_norm)
clean_ppm = calc_scatter(clean_norm)

print(f"\n=========================================================================")
print(f"         GERCEK FOTOMETRIK GURULTU RAPORU (ANALITIK PSF)                 ")
print(f"=========================================================================")
print(f"Ham TESS Basit Toplam Gurultusu (SAP)   : {raw_ppm:.1f} ppm")
print(f"Gaia PSF ile Ayrastirilmis Hedef Gurultu : {clean_ppm:.1f} ppm")
print(f"=========================================================================")

# 5. Transit Arama Motoru (Box Least Squares)
print("\n3. Isik Egrisinde Gezegen Sinyali Araniyor (BLS Motoru)...")
bls = BoxLeastSquares(time_arr, clean_norm)
periods = np.linspace(0.5, 15.0, 5000)
durations = np.linspace(0.05, 0.2, 10)
periodogram = bls.power(periods, durations)

best_p = periodogram.period[np.argmax(periodogram.power)]
best_depth = periodogram.depth[np.argmax(periodogram.power)]

print(f"\n[TRANSIT ANALIZI TAMAMLANDI]:")
print(f"--> En Olasi Yorunge Periyodu : {best_p:.4f} Gun")
print(f"--> Olculen Transit Derinligi : {best_depth * 1e6:.1f} ppm ({(best_depth)*100:.3f}%)")

# 6. CSV'ye Kaydet
output_csv = "tic261136679_calibrated_lightcurve.csv"
df = pd.DataFrame({
    "time_bjd": time_arr,
    "raw_flux": raw_norm,
    "clean_flux": clean_norm
})
df.to_csv(output_csv, index=False)
print(f"\n[ISLEM BASARILI]: Isik egrisi '{output_csv}' dosyasina yazildi!")
