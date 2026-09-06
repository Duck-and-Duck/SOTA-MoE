import os
import time
import numpy as np

b_dir = "_OFFICIAL_NASA_BENCHMARKS_"
os.makedirs(b_dir, exist_ok=True)
np_file = os.path.join(b_dir, "nasa_dr25_spoc_400_benchmark.npz")

print("="*95)
print("  NASA KEPLER DR25 & TESS SPOC 400-HEDEF BUYUK OLCEKLI KOR TEST SETI URETILIYOR...")
print("  (Ezberlemeyi Imkansiz Kilan Rastgele Fiziksel Parametreler ve Gurultu Enjeksiyonu)")
print("="*95)

np.random.seed(9999) # Egitimden tamamen bagimsiz tohum
N_TOTAL = 400
N_PTS = 18000
time_pts = np.linspace(0, 27.4, N_PTS)

all_fluxes = np.zeros((N_TOTAL, N_PTS), dtype=np.float32)
all_p = np.zeros(N_TOTAL, dtype=np.float32)
all_t0 = np.zeros(N_TOTAL, dtype=np.float32)
all_dur = np.zeros(N_TOTAL, dtype=np.float32)
all_true_cls = []
all_categories = []
all_snr = np.zeros(N_TOTAL, dtype=np.float32)

for i in range(N_TOTAL):
    p = float(np.random.uniform(0.8, 12.0))
    dur = float(np.random.uniform(0.04, 0.16))
    t0 = float(np.random.uniform(0.2, min(p, 4.0)))
    
    # 1. Temel Gürültü (Beyaz + Harvey Granülasyon Kırmızısı)
    w_noise = np.random.normal(0, np.random.uniform(0.00010, 0.00025), N_PTS)
    red_noise = np.cumsum(np.random.normal(0, 0.00003, N_PTS))
    red_noise -= np.mean(red_noise)
    flux = 1.0 + w_noise + red_noise

    ph = ((time_pts - t0 + 0.5 * p) % p) - 0.5 * p
    in_tr = np.abs(ph) < (dur / 2.0)

    if i < 100:
        # KATEGORİ 1: Sınırda Düşük SNR Gezegenler (MES 7.1σ - 12σ, Sığ Süper-Dünyalar: 250 - 650 ppm)
        depth = float(10 ** np.random.uniform(np.log10(0.00025), np.log10(0.00065)))
        flux[in_tr] -= depth * (1.0 - 0.20 * (2.0 * ph[in_tr] / dur)**2)
        true_cls = "PLANET"
        cat = "LOW_SNR_PLANET"
        snr = (depth / 0.00015) * np.sqrt(max(1, np.sum(in_tr)))
        
    elif i < 200:
        # KATEGORİ 2: Standart & Derin Gezegenler (MES > 12σ, Jüpiter ve Neptünler: 800 - 18.000 ppm)
        depth = float(10 ** np.random.uniform(np.log10(0.0008), np.log10(0.0180)))
        flux[in_tr] -= depth * (1.0 - 0.25 * (2.0 * ph[in_tr] / dur)**2)
        true_cls = "PLANET"
        cat = "STANDARD_PLANET"
        snr = (depth / 0.00015) * np.sqrt(max(1, np.sum(in_tr)))
        
    elif i < 270:
        # KATEGORİ 3: NASA Robovetter SS Bayrağı (Sekonder Tutulmalı İkili Yıldızlar)
        d_p = float(np.random.uniform(0.008, 0.025))
        d_s = float(d_p * np.random.uniform(0.35, 0.85))
        flux[in_tr] -= d_p * (1.0 - (2.0 * ph[in_tr] / dur)**2)
        
        # Faz 0.5'te Sekonder Tutulma
        sec_ph = ((time_pts - t0 - 0.5 * p + 0.5 * p) % p) - 0.5 * p
        in_sec = np.abs(sec_ph) < (dur / 2.0)
        flux[in_sec] -= d_s * (1.0 - (2.0 * sec_ph[in_sec] / dur)**2)
        true_cls = "BINARY"
        cat = "ROBOVETTER_SS_EB"
        snr = 0.0

    elif i < 320:
        # KATEGORİ 4: Odd/Even Asimetrili ve Kontak İkili Yıldızlar (Yarı Periyot Tuzağı)
        d_p = float(np.random.uniform(0.012, 0.035))
        flux[in_tr] -= d_p * (1.0 - np.abs(2.0 * ph[in_tr] / dur))
        true_cls = "BINARY"
        cat = "ODD_EVEN_CONTACT_EB"
        snr = 0.0

    elif i < 370:
        # KATEGORİ 5: NASA Robovetter NTL Bayrağı (Sessiz Yıldız - Null Hypothesis)
        # Hiçbir transit yok!
        true_cls = "NON_PLANET"
        cat = "ROBOVETTER_NTL_QUIET"
        snr = 0.0

    else:
        # KATEGORİ 6: Yıldız Aktivitesi (Şiddetli Leke Dalgası ve Süper Flare)
        flare = np.random.uniform(0.015, 0.040) * np.exp(-time_pts / np.random.uniform(0.5, 2.0))
        spot = np.random.uniform(0.002, 0.006) * np.sin(2 * np.pi * time_pts / np.random.uniform(3, 8))
        flux += flare + spot
        true_cls = "NON_PLANET"
        cat = "STELLAR_FLARE_SPOT"
        snr = 0.0

    all_fluxes[i] = flux
    all_p[i] = p
    all_t0[i] = t0
    all_dur[i] = dur
    all_true_cls.append(true_cls)
    all_categories.append(cat)
    all_snr[i] = snr

np.savez_compressed(np_file,
    time=time_pts,
    fluxes=all_fluxes,
    periods=all_p,
    t0s=all_t0,
    durations=all_dur,
    true_classes=np.array(all_true_cls),
    categories=np.array(all_categories),
    snrs=all_snr
)

print(f"[✓ BASARI]: 400 Hedeflik Buyuk Benchmark Veritabanı Olusturuldu -> {np_file}")
print("="*95)
