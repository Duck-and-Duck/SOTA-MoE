import warnings
import logging
import time
import os
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
socket.setdefaulttimeout(20)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*85)
print("  OSTE-MoE AUTONOMOUS EXOPLANET MINER & VETTING ENGINE (REAL SKY SCANNER)")
print(f"--> COMPUTATION: {device.upper()} (RTX 3050 Ti) | PIPELINE: SPOC-Calibrated")
print("="*85)

# =========================================================================
# 1. MODEL YUKLEME (JIT FREEZE)
# =========================================================================
class AstroNetHQ(nn.Module):
    def __init__(self):
        super(AstroNetHQ, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(15),
            nn.Flatten()
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(15),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 15 + 32 * 15, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

model = AstroNetHQ().to(device)
model.load_state_dict(torch.load("astronet_hq.pt", map_location=device))
model.eval()

dummy_g = torch.randn(1, 1, 201, device=device)
dummy_l = torch.randn(1, 1, 61, device=device)
frozen_vetter = torch.jit.freeze(torch.jit.trace(model, (dummy_g, dummy_l)))
print("--> [TAMAM]: AstroNet-HQ JIT Nöral Omurga Belleğe Alındı.\n")

# =========================================================================
# 2. GPU FAZ KATLAMA VE BINNING
# =========================================================================
def gpu_box_bin(query_bins, sorted_phase, sorted_flux):
    half_w = 0.5 * (query_bins[1] - query_bins[0])
    l_edge = query_bins - half_w
    r_edge = query_bins + half_w
    idx_l = torch.searchsorted(sorted_phase, l_edge)
    idx_r = torch.searchsorted(sorted_phase, r_edge)
    f_cumsum = F.pad(torch.cumsum(sorted_flux, dim=0), (1, 0))
    bin_sums = f_cumsum[idx_r] - f_cumsum[idx_l]
    bin_counts = idx_r - idx_l
    mask = bin_counts > 0
    res = torch.zeros_like(query_bins)
    res[mask] = bin_sums[mask] / bin_counts[mask].float()
    if not torch.all(mask):
        empty_idx = torch.clamp(torch.searchsorted(sorted_phase, query_bins[~mask]), 0, len(sorted_flux) - 1)
        res[~mask] = sorted_flux[empty_idx]
    return res

def phase_fold_batch_gpu(t_t, f_t, p, t0):
    phase = ((t_t - t0 + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase)
    s_ph, s_fl = phase[s_idx], f_t[s_idx]

    g_bins = torch.linspace(-0.5 * p, 0.5 * p, 201, device=device)
    g_raw = gpu_box_bin(g_bins, s_ph, s_fl)

    l_bins = torch.linspace(-p * 0.05, p * 0.05, 61, device=device)
    l_raw = gpu_box_bin(l_bins, s_ph, s_fl)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm.view(1, 1, 201), l_norm.view(1, 1, 61)

# =========================================================================
# 3. KÖR TRANSİT ARAMA VE TCE EKSTRAKSİYONU (Box Least Squares)
# =========================================================================
def autonomous_tce_search(time_arr, flux_arr):
    t0 = time.perf_counter()
    bls = BoxLeastSquares(time_arr, flux_arr)
    periods = np.linspace(0.8, 12.0, 5000)
    durations = np.linspace(0.04, 0.16, 6)
    periodogram = bls.power(periods, durations)

    best_idx = np.argmax(periodogram.power)
    p_bls = float(periodogram.period[best_idx])
    t0_bls = float(periodogram.transit_time[best_idx])
    dur_bls = float(periodogram.duration[best_idx])
    power_bls = float(periodogram.power[best_idx])

    std_pow = np.std(periodogram.power)
    snr = power_bls / (std_pow if std_pow > 0 else 1e-7)

    # 0.5*P Harmonik Kontrolü
    half_p = p_bls * 0.5
    if half_p >= 0.75:
        test_sub = bls.power(np.array([half_p]), [dur_bls])
        if len(test_sub.power) > 0 and float(test_sub.power[0]) >= 0.85 * power_bls:
            p_bls = half_p
            t0_bls = float(test_sub.transit_time[0])

    lat_ms = (time.perf_counter() - t0) * 1000
    return {"p": p_bls, "t0": t0_bls, "dur": dur_bls, "snr": snr, "lat_ms": lat_ms}

# =========================================================================
# 4. ASTROFİZİKSEL ADAPTİF VETTING VE ROBO-KARAR
# =========================================================================
def vet_transit_candidate(time_arr, flux_arr, tce, t_gpu, f_gpu):
    p, t0, dur = tce["p"], tce["t0"], tce["dur"]
    g, l = phase_fold_batch_gpu(t_gpu, f_gpu, p, t0)
    with torch.no_grad():
        prob_cnn = torch.sigmoid(frozen_vetter(g, l)).item()

    # İstatistiksel Testler
    phase = ((time_arr - t0 + 0.5 * p) % p) - 0.5 * p
    in_tr = np.abs(phase) < (dur / 2.0)
    tr_num = np.round((time_arr - t0) / p)

    odd_m = in_tr & (tr_num % 2 != 0)
    even_m = in_tr & (tr_num % 2 == 0)

    out_tr = ~in_tr
    sigma_flux = np.std(flux_arr[out_tr]) if np.sum(out_tr) > 20 else np.std(flux_arr)

    d_odd = float(1.0 - np.nanmedian(flux_arr[odd_m])) if np.sum(odd_m) > 2 else 0.0
    d_even = float(1.0 - np.nanmedian(flux_arr[even_m])) if np.sum(even_m) > 2 else 0.0
    se_diff = sigma_flux * np.sqrt(1.0 / max(1, np.sum(odd_m)) + 1.0 / max(1, np.sum(even_m))) + 1e-8
    z_oe = abs(d_odd - d_even) / se_diff

    sec_ph = ((time_arr - t0) % p) - 0.5 * p
    sec_m = np.abs(sec_ph) < (dur / 2.0)
    d_sec = float(1.0 - np.nanmedian(flux_arr[sec_m])) if np.sum(sec_m) > 2 else 0.0
    se_sec = sigma_flux / np.sqrt(max(1, np.sum(sec_m))) + 1e-8
    z_sec = d_sec / se_sec

    max_d = max(d_odd, d_even, 1e-5)
    sec_ratio = d_sec / max_d

    # Karar Süzgeci
    is_eb = (z_sec >= 3.2 and sec_ratio >= 0.28) or (z_oe >= 3.5 and abs(d_odd - d_even)/max_d >= 0.40)
    is_noise = (tce["snr"] < 6.5)

    if is_noise:
        verdict = "NON_PLANET (NOISE / FALSE ALARM)"
    elif is_eb:
        verdict = "ECLIPSING_BINARY (FALSE POSITIVE)"
    else:
        verdict = "CONFIRMED_PLANET_CANDIDATE" if prob_cnn >= 0.35 else "SUSPECTED_FALSE_POSITIVE"

    return verdict, prob_cnn, z_oe, z_sec, max_d

# =========================================================================
# 5. GERÇEK GÖKYÜZÜ HEDEFLERİNİN KÖR TARAMASI
# =========================================================================
SURVEY_TARGETS = [
    {"name": "WASP-18",   "tic": "TIC 100100827", "sector": 2,  "true": "PLANET"},
    {"name": "L 98-59",   "tic": "TIC 307210830", "sector": 2,  "true": "PLANET"},
    {"name": "WASP-126",  "tic": "TIC 25155310",  "sector": 1,  "true": "PLANET"},
    {"name": "TESS EB 1", "tic": "TIC 339607421", "sector": 2,  "true": "BINARY"},
    {"name": "Quiet Star","tic": "TIC 261136679", "sector": 14, "true": "NON_PLANET"}
]

print(">>> CANLI GÖZLEM TARAMASI BAŞLATILIYOR (Önceden Hiçbir Periyot Verilmez)...")
print("--> Sistem: 1) Veriyi indirir 2) BLS ile tarar 3) AI & SPOC Vetting uygular 4) Hüküm verir.\n")

hits = 0
for tgt in SURVEY_TARGETS:
    t_start = time.perf_counter()
    print(f"[TARAMA]: {tgt['name']} ({tgt['tic']})...")

    # Yerel önbellekleme
    os.makedirs("_CACHE_LIGHTCURVES_", exist_ok=True)
    cache_f = os.path.join("_CACHE_LIGHTCURVES_", f"{tgt['tic']}_s{tgt['sector']}.npz")

    if os.path.exists(cache_f):
        data = np.load(cache_f)
        t_arr, f_arr = data["t"], data["f"]
    else:
        search = lk.search_lightcurve(tgt["tic"], mission="TESS", sector=tgt["sector"])
        lc = search[0].download(quality_bitmask="hardest").remove_nans().flatten(window_length=301)
        t_arr = np.asarray(lc.time.value, dtype=np.float64)
        f_arr = np.asarray(lc.flux.value, dtype=np.float64)
        np.savez(cache_f, t=t_arr, f=f_arr)

    # 1. Adım: Kör Arama
    tce = autonomous_tce_search(t_arr, f_arr)

    # 2. Adım: GPU Fazlama ve Robo-Vetting
    t_gpu = torch.tensor(t_arr, dtype=torch.float32, device=device)
    f_gpu = torch.tensor(f_arr, dtype=torch.float32, device=device)
    verdict, prob, z_oe, z_sec, depth = vet_transit_candidate(t_arr, f_arr, tce, t_gpu, f_gpu)

    total_time = (time.perf_counter() - t_start) * 1000

    is_planet_found = (verdict == "CONFIRMED_PLANET_CANDIDATE")
    should_be_planet = (tgt["true"] == "PLANET")
    correct = (is_planet_found == should_be_planet)
    if correct: hits += 1

    print(f"    * Keşfedilen Aday Periyot: {tce['p']:.4f} Gün | Derinlik: {depth*1e6:.1f} ppm | SNR: {tce['snr']:.1f}σ")
    print(f"    * 1D-CNN Vetter Skoru     : %{prob*100:.2f} | Z_oe: {z_oe:.1f}σ | Z_sec: {z_sec:.1f}σ")
    print(f"    * Nihai Otonom Hüküm      : {verdict}")
    print(f"    * Sonuç Kararı            : {'[✓ TAM İSABET]' if correct else '[✗ HATALI]'}")
    print(f"    * Toplam Boru Hattı Süresi: {total_time:.1f} ms\n")

print("="*85)
print(f"--> KÖR GÖKYÜZÜ TARAMA BAŞARISI: {hits} / {len(SURVEY_TARGETS)} Tam İsabet (%{hits/len(SURVEY_TARGETS)*100:.1f})")
print("="*85)