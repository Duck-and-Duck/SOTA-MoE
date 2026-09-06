import warnings
import logging
import time
import torch
import numpy as np

# Tum uyarilari kesin olarak sustur
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from astropy.timeseries import BoxLeastSquares
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import lightkurve as lk

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print("      OSTE-MoE: OTEGEZEGEN KESIF VE ATMOSFERIK TESHIS MOTORU             ")
print(f"--> DONANIM HIZLANDIRICI: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

# Cepteki 300-Kanal Atmosfer Modelini Yukle
try:
    atm_posterior = torch.load("posterior_300ch.pt", map_location=device, weights_only=False)
    print("--> Katman 4 [Cepteki Atmosfer Uzmani]: 'posterior_300ch.pt' Aktif.")
except Exception as e:
    print(f"[BILGI]: Atmosfer modeli bulunamadi ({e}). Katman 4 pasif modda.")
    atm_posterior = None

# =========================================================================
# KATMAN 1: UZAMSAL AYRISTIRMA (GAIA PSF) VE ZAMANSAL FILTRELEME
# =========================================================================
class SpatialTemporalExpert:
    def __init__(self, sigma_psf=0.85):
        self.sigma_psf = sigma_psf

    def process_tpf(self, tpf, gaia_stars):
        # Sadece algoritmik hesaplama suresini olc (Ag beklemesi haric)
        t0 = time.perf_counter()
        
        time_arr = tpf.time.value
        flux_cube = tpf.flux.value # (T, Y, X)

        valid = ~np.isnan(flux_cube).any(axis=(1, 2)) & ~np.isnan(time_arr)
        time_arr, flux_cube = time_arr[valid], flux_cube[valid]
        T_len, H, W = flux_cube.shape

        sky_coords = SkyCoord(gaia_stars["ra"], gaia_stars["dec"], unit="deg")
        pix_x, pix_y = tpf.wcs.world_to_pixel(sky_coords)

        in_bounds = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H)
        pix_x, pix_y = pix_x[in_bounds], pix_y[in_bounds]
        K_stars = len(pix_x)

        if K_stars == 0:
            # Fallback: Klasik SAP akisi
            raw_sap = np.sum(flux_cube, axis=(1, 2))
            clean_flux = raw_sap / np.median(raw_sap)
            return time_arr, clean_flux, flux_cube, (W/2, H/2), (time.perf_counter() - t0)*1000

        # 2D-Gaussian PSF Tasarim Matrisi (A)
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        A = np.zeros((H * W, K_stars + 1))
        for k in range(K_stars):
            psf_k = np.exp(-((grid_x - pix_x[k])**2 + (grid_y - pix_y[k])**2) / (2 * self.sigma_psf**2))
            psf_k /= (np.sum(psf_k) + 1e-8)
            A[:, k] = psf_k.flatten()
        A[:, -1] = 1.0 # Arka plan

        # Matris Tersi: F = pinv(A) * I
        flat_I = flux_cube.reshape(T_len, H * W).T
        pinv_A = np.linalg.pinv(A)
        F_solutions = np.dot(pinv_A, flat_I)
        target_flux = F_solutions[0, :]
        norm_flux = target_flux / (np.median(target_flux) + 1e-8)

        # Causal Detrending: Kayan medyan filtresi (Pencere: 45 adim)
        pad = 22
        padded = np.pad(norm_flux, pad, mode='edge')
        trend = np.array([np.median(padded[i:i+45]) for i in range(len(norm_flux))])
        clean_flux = norm_flux / (trend + 1e-8)

        compute_latency_ms = (time.perf_counter() - t0) * 1000
        return time_arr, clean_flux, flux_cube, (pix_x[0], pix_y[0]), compute_latency_ms

# =========================================================================
# KATMAN 2: ZAMANSAL GECIS TESPIT MOTORU (TRANSIT DETECTION)
# =========================================================================
class TransitDetectionEngine:
    def __init__(self):
        pass

    def detect_candidate(self, time_arr, flux_arr):
        t0 = time.perf_counter()
        
        bls = BoxLeastSquares(time_arr, flux_arr)
        periods = np.linspace(0.8, 12.0, 3000)
        durations = np.linspace(0.04, 0.16, 8)
        periodogram = bls.power(periods, durations)

        best_idx = np.argmax(periodogram.power)
        period = float(periodogram.period[best_idx])
        t0_epoch = float(periodogram.transit_time[best_idx])
        duration = float(periodogram.duration[best_idx])
        depth = float(periodogram.depth[best_idx])
        
        std_p = np.std(periodogram.power)
        snr = float(periodogram.power[best_idx] / (std_p if std_p > 0 else 1e-7))

        latency_ms = (time.perf_counter() - t0) * 1000
        tce = {
            "period": period,
            "t0": t0_epoch,
            "duration": duration,
            "depth": max(depth, 1e-6),
            "snr": snr
        }
        return tce, latency_ms

# =========================================================================
# KATMAN 3: ROBO-VETTER EXPERT (IKILI YILDIZ VE ANOMALI KATILI)
# =========================================================================
class RoboVetterExpert:
    def __init__(self):
        pass

    def vet(self, time_arr, flux_arr, flux_cube, target_pos, tce):
        t0 = time.perf_counter()
        p = tce["period"]
        epoch = tce["t0"]
        dur = tce["duration"]

        phase = ((time_arr - epoch + 0.5 * p) % p) - 0.5 * p
        in_transit = np.abs(phase) < (dur / 2.0)

        # 1. TEST: Tek / Cift (Odd / Even) Derinlik Testi
        transit_nums = np.round((time_arr - epoch) / p)
        odd_mask = in_transit & (transit_nums % 2 != 0)
        even_mask = in_transit & (transit_nums % 2 == 0)

        depth_odd = float(1.0 - np.median(flux_arr[odd_mask])) if np.sum(odd_mask) > 0 else tce["depth"]
        depth_even = float(1.0 - np.median(flux_arr[even_mask])) if np.sum(even_mask) > 0 else tce["depth"]

        denom = (np.std(flux_arr) / np.sqrt(max(np.sum(in_transit), 1))) + 1e-7
        z_odd_even = abs(depth_odd - depth_even) / denom
        z_odd_even = float(np.nan_to_num(z_odd_even, nan=0.0))
        pass_odd_even = z_odd_even < 2.5

        # 2. TEST: In-Transit Centroid Kaymasi
        T, H, W = flux_cube.shape
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        
        in_flux = flux_cube[in_transit]
        out_flux = flux_cube[~in_transit]

        sum_in = np.sum(in_flux)
        sum_out = np.sum(out_flux)

        if sum_in > 0 and sum_out > 0:
            cen_x_in = np.sum(in_flux * grid_x) / sum_in
            cen_y_in = np.sum(in_flux * grid_y) / sum_in
            cen_x_out = np.sum(out_flux * grid_x) / sum_out
            cen_y_out = np.sum(out_flux * grid_y) / sum_out
            centroid_shift = float(np.sqrt((cen_x_in - cen_x_out)**2 + (cen_y_in - cen_y_out)**2))
        else:
            centroid_shift = 0.0

        centroid_shift = float(np.nan_to_num(centroid_shift, nan=0.0))
        pass_centroid = centroid_shift < 0.25

        # 3. TEST: Sinyal-Gürültü Güvenilirliği (SNR)
        pass_snr = tce["snr"] >= 7.1

        # Skorlama
        score = 0.0
        if pass_odd_even: score += 0.35
        if pass_centroid: score += 0.40
        if pass_snr: score += 0.25

        decision = "ONAYLANMIS OTEGEZEGEN ADAYI" if score >= 0.85 else "SAHTE POZITIF / ELEME"
        latency_ms = (time.perf_counter() - t0) * 1000

        report = {
            "score": score,
            "decision": decision,
            "pass_odd_even": pass_odd_even,
            "z_odd_even": z_odd_even,
            "centroid_shift_pix": centroid_shift,
            "pass_centroid": pass_centroid,
            "pass_snr": pass_snr,
            "latency_ms": latency_ms
        }
        return report, latency_ms

# =========================================================================
# KATMAN 4: CEPTEKI UZMAN - HIZLI ATMOSFERIK TESHIS (SBI / MAF)
# =========================================================================
class PocketedAtmosphericExpert:
    def __init__(self, model):
        self.model = model

    def infer(self, depth, gravity=5.0):
        if self.model is None:
            return None, 0.0
        t0 = time.perf_counter()
        
        # 305 Boyutlu Uyumlu Girdi Vektoru
        norm_detrended = torch.zeros(300, device=device)
        robust_amp = torch.tensor([depth * 1000.0], device=device)
        h2o_c = torch.tensor([depth * 0.4 * 1000.0], device=device)
        co2_c = torch.tensor([depth * 0.6 * 1000.0], device=device)
        curv = torch.tensor([0.0], device=device)
        norm_g = torch.tensor([(gravity - 6.25) / 2.25], device=device)

        x_in = torch.cat([norm_detrended, robust_amp, h2o_c, co2_c, curv, norm_g]).unsqueeze(0)
        samples = self.model.sample((1500,), x=x_in, show_progress_bars=False)

        h2o = float(samples[:, 0].mean().item())
        co2 = float(samples[:, 1].mean().item())
        temp = float(samples[:, 5].mean().item())
        latency_ms = (time.perf_counter() - t0) * 1000

        diag = {"h2o": h2o, "co2": co2, "temp": temp}
        return diag, latency_ms

# =========================================================================
# UÇTAN UCA ÇALIŞTIRMA KONTROLÜ
# =========================================================================
print("\n>>> 1. Arşiv Verisi Çekiliyor (Ağ I/O Süresi Hesaplanıyor)...")
t_io = time.perf_counter()
search = lk.search_targetpixelfile("TIC 261136679", mission="TESS")
tpf = search[0].download(quality_bitmask="default")

ra, dec = tpf.ra, tpf.dec
query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.03))
ORDER BY phot_g_mean_mag ASC
"""
gaia_stars = Gaia.launch_job(query).get_results()
io_latency = (time.perf_counter() - t_io) * 1000
print(f"--> [Ag & Disk I/O]: Veriler {io_latency:.1f} ms icinde hafizaya alindi.")

expert1 = SpatialTemporalExpert()
expert2 = TransitDetectionEngine()
expert3 = RoboVetterExpert()
expert4 = PocketedAtmosphericExpert(atm_posterior)

print("\n>>> 2. OSTE-MoE Boru Hattı Çalıştırılıyor (Saf Hesaplama)...")

# Katman 1
time_arr, clean_flux, flux_cube, target_pos, lat1 = expert1.process_tpf(tpf, gaia_stars)
print(f"  [KATMAN 1]: Gaia PSF Dekonvolüsyonu & Filtre : {lat1:6.2f} ms")

# Katman 2
tce, lat2 = expert2.detect_candidate(time_arr, clean_flux)
print(f"  [KATMAN 2]: Evrişimsel Geçiş Arama (TCE Motoru) : {lat2:6.2f} ms")
print(f"              -> Aday Periyot: {tce['period']:.4f} Gün | Derinlik: {tce['depth']*1e6:.1f} ppm | SNR: {tce['snr']:.1f}")

# Katman 3
vetting, lat3 = expert3.vet(time_arr, clean_flux, flux_cube, target_pos, tce)
print(f"  [KATMAN 3]: Robo-Vetter (İkili Yıldız Eleme)  : {lat3:6.2f} ms")
print(f"              -> Vetting Skoru: {vetting['score']:.2f} / 1.00 | Karar: {vetting['decision']}")
print(f"              -> Odd/Even Z: {vetting['z_odd_even']:.2f} sigma | Centroid Kayması: {vetting['centroid_shift_pix']:.3f} px")

# Katman 4
lat4 = 0.0
if vetting["score"] >= 0.85 and atm_posterior is not None:
    atm_diag, lat4 = expert4.infer(tce["depth"])
    print(f"  [KATMAN 4]: Cepteki Atmosfer Motoru (SBI/MAF)   : {lat4:6.2f} ms")
    print(f"              -> log(H2O): {atm_diag['h2o']:.2f} | log(CO2): {atm_diag['co2']:.2f} | T_dengeli: {atm_diag['temp']:.1f} K")
else:
    print(f"  [KATMAN 4]: Aday Vetting esigini (0.85) asamadigi icin (SNR={tce['snr']:.1f}) atmosfer cikarimi pas gecildi.")

pure_compute_time = lat1 + lat2 + lat3 + lat4
print("\n=========================================================================")
print(f"                   MoE TELEMETRİ VE PERFORMANS RAPORU                    ")
print("=========================================================================")
print(f"--> KATMAN 1 HESAPLAMA (Gaia PSF)      : {lat1:6.2f} ms")
print(f"--> KATMAN 2 HESAPLAMA (TCE Tespiti)   : {lat2:6.2f} ms")
print(f"--> KATMAN 3 HESAPLAMA (Robo-Vetter)   : {lat3:6.2f} ms")
print(f"--> KATMAN 4 HESAPLAMA (Atmosfer/SBI)  : {lat4:6.2f} ms")
print("-------------------------------------------------------------------------")
print(f"--> TOPLAM SAF HESAPLAMA SÜRESİ        : {pure_compute_time:6.2f} ms")
print(f"--> HESAPLAMA HIZI (İŞLEM KAPASİTESİ)   : ~{1000.0 / pure_compute_time:.1f} Işık Eğrisi / Saniye")
print("=========================================================================")
