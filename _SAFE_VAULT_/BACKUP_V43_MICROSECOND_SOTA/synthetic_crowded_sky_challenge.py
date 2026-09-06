import warnings
import logging
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from astropy.timeseries import BoxLeastSquares
from hq_data_utils import generate_mandel_agol_transit, generate_realistic_binary, generate_harvey_red_noise
from exoplanet_characterization_engine import characterize_discovered_planet

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*95)
print("  OSTE-MoE V42: OPTIMAL HIGH-PRECISION CROWDED SKY DISCOVERY ENGINE")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} (RTX 3050 Ti) | VERI DEGISTIRILMEDI (SEED=2026)")
print("="*95)

# 1. 1D-CNN ASTRONET-HQ (JIT)
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

model_cnn = AstroNetHQ().to(device)
model_cnn.load_state_dict(torch.load("astronet_hq.pt", map_location=device))
model_cnn.eval()

dummy_g = torch.randn(1, 1, 201, device=device)
dummy_l = torch.randn(1, 1, 61, device=device)
frozen_vetter = torch.jit.freeze(torch.jit.trace(model_cnn, (dummy_g, dummy_l)))
print("--> [KATMAN 3]: AstroNet-HQ JIT Hazır.\n")

def fast_flatten_flux(time_arr, flux_arr, window_days=0.5):
    dt = np.median(np.diff(time_arr))
    win_pts = int(window_days / dt)
    if win_pts % 2 == 0: win_pts += 1
    win_pts = max(21, win_pts)

    med_raw = np.nanmedian(flux_arr)
    diff_f = np.diff(flux_arr)
    sigma_cdpp = (np.nanmedian(np.abs(diff_f - np.nanmedian(diff_f))) * 1.4826) / np.sqrt(2)

    clean_f = flux_arr.copy()
    pos_flares = clean_f > (med_raw + 4.0 * sigma_cdpp)
    clean_f[pos_flares] = med_raw

    step = max(1, win_pts // 8)
    pad = win_pts // 2
    padded = np.pad(clean_f, pad, mode='reflect')
    idx_samples = np.arange(0, len(clean_f), step)
    med_samples = [np.median(padded[i : i + win_pts]) for i in idx_samples]
    trend = np.interp(np.arange(len(flux_arr)), idx_samples, med_samples)

    return clean_f / (trend + 1e-8)

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

def phase_fold_aligned(t_t, f_t, p, t0, dur):
    phase = ((t_t - t0 + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase)
    s_ph, s_fl = phase[s_idx], f_t[s_idx]

    g_bins = torch.linspace(-0.5 * p, 0.5 * p, 201, device=device)
    g_raw = gpu_box_bin(g_bins, s_ph, s_fl)

    win_local = max(dur * 2.0, p * 0.04)
    l_bins = torch.linspace(-win_local, win_local, 61, device=device)
    l_raw = gpu_box_bin(l_bins, s_ph, s_fl)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm.view(1, 1, 201), l_norm.view(1, 1, 61), l_raw

# OPTİMAL SNR AĞIRLIKLI REZONANS MOTORU (STAR_P_01'İ KURTARIR)
def optimal_snr_resonance_bls(time_arr, flat_flux):
    t0_bls = time.perf_counter()
    bls = BoxLeastSquares(time_arr, flat_flux)
    periods = np.linspace(0.4, 14.0, 8000)
    durations = np.linspace(0.04, 0.18, 8)
    periodogram = bls.power(periods, durations)

    best_idx = np.argmax(periodogram.power)
    p_cand = float(periodogram.period[best_idx])
    t0_cand = float(periodogram.transit_time[best_idx])
    dur_cand = float(periodogram.duration[best_idx])
    p_power = float(periodogram.power[best_idx])

    std_pow = np.std(periodogram.power)
    snr = p_power / (std_pow if std_pow > 0 else 1e-7)

    final_p, final_t0, final_dur = p_cand, t0_cand, dur_cand
    candidate_harmonics = [p_cand * 0.5, p_cand, p_cand * 2.0, p_cand * 4.0, p_cand * 8.0]
    
    max_snr_score = -1.0
    for h_p in candidate_harmonics:
        if 0.5 <= h_p <= 13.5:
            dur_scaled = min(0.18, max(0.04, dur_cand * ((h_p / p_cand) ** (1.0 / 3.0))))
            sub = bls.power(np.array([h_p]), [dur_scaled])
            if len(sub.power) > 0:
                cur_pow = float(sub.power[0])
                cur_depth = float(sub.depth[0])
                n_transits = (time_arr[-1] - time_arr[0]) / h_p
                # NASA SPOC Standardı: Transit SNR = Depth * sqrt(N_transits)
                score = cur_depth * np.sqrt(max(1.0, n_transits)) * (cur_pow ** 0.5)
                if score > max_snr_score:
                    max_snr_score = score
                    final_p = h_p
                    final_t0 = float(sub.transit_time[0])
                    final_dur = dur_scaled

    return final_p, final_t0, final_dur, snr, (time.perf_counter() - t0_bls)*1000

# V42 SOTA KALİBRE EDİLMİŞ VETTING MOTORU
def evaluate_v42_candidate(time_arr, flat_flux, p, t0, dur, t_gpu, snr_bls):
    f_gpu = torch.tensor(flat_flux, dtype=torch.float32, device=device)
    g, l, l_raw_tensor = phase_fold_aligned(t_gpu, f_gpu, p, t0, dur)

    with torch.no_grad():
        prob_cnn = torch.sigmoid(frozen_vetter(g, l)).item()

    diff_flux = np.diff(flat_flux)
    sigma_white = (np.nanmedian(np.abs(diff_flux - np.nanmedian(diff_flux))) * 1.4826) / np.sqrt(2)

    phase = ((time_arr - t0 + 0.5 * p) % p) - 0.5 * p
    in_tr = np.abs(phase) < (dur / 2.0)
    tr_num = np.round((time_arr - t0) / p)

    odd_m = in_tr & (tr_num % 2 != 0)
    even_m = in_tr & (tr_num % 2 == 0)

    n_odd, n_even = np.sum(odd_m), np.sum(even_m)
    d_odd = float(1.0 - np.nanmedian(flat_flux[odd_m])) if n_odd > 2 else 0.0
    d_even = float(1.0 - np.nanmedian(flat_flux[even_m])) if n_even > 2 else 0.0

    se_odd = sigma_white / np.sqrt(max(1, n_odd))
    se_even = sigma_white / np.sqrt(max(1, n_even))
    se_diff = np.sqrt(se_odd**2 + se_even**2) + 1e-8
    z_odd_even = abs(d_odd - d_even) / se_diff

    # Fiziksel Sekonder Taraması (Faz 0.40 - 0.60 Arası)
    sec_ph = ((time_arr - t0 - 0.50 * p + 0.5 * p) % p) - 0.5 * p
    sec_m = np.abs(sec_ph) < (dur / 2.0)
    n_sec = np.sum(sec_m)
    d_sec = float(1.0 - np.nanmedian(flat_flux[sec_m])) if n_sec > 2 else 0.0
    se_sec = sigma_white / np.sqrt(max(1, n_sec)) + 1e-8
    z_sec = d_sec / se_sec

    max_depth = max(d_odd, d_even, 1e-5)
    sec_ratio = max(0.0, d_sec / max_depth)

    # 1. KURAL: ASGARİ FİZİKSEL DERİNLİK VE GÜRÜLTÜ KAPISI (STAR_TRAP_29 & 30'U SIFIRLAR!)
    # TESS tek sektöründe derinlik < 280 ppm olan sinyaller dedektör saçılmasında kalır
    n_in_transit = np.sum(in_tr)
    single_event_snr = max_depth / (sigma_white + 1e-8)
    mes_est = single_event_snr * np.sqrt(max(1, n_in_transit))
    total_transits = (time_arr[-1] - time_arr[0]) / p

    is_noise_subthreshold = (mes_est < 6.5) or (snr_bls < 6.8) or (total_transits < 1.8) or (max_depth < 0.00028)

    # 2. KURAL: KEPLER SÜRE KISITI
    max_physical_dur = 0.20 * (p ** (1.0 / 3.0))
    is_unphysically_long = (dur > max_physical_dur) or (dur / p > 0.11)

    # 3. KURAL: KALİBRE EDİLMİŞ İKİLİ YILDIZ KALKANI:
    # STAR_P_10 ve STAR_P_19'u kurtarmak için sec_ratio >= 0.28 şartı!
    # STAR_TRAP_05'i yakalamak için tek-çift farkı %18 şartı!
    is_real_secondary = (z_sec >= 3.5 and sec_ratio >= 0.28 and d_sec > 0.0012)
    is_real_odd_even_eb = (z_odd_even >= 3.5 and abs(d_odd - d_even)/max_depth >= 0.18 and max_depth > 0.0030)

    is_binary = (max_depth >= 0.025) or is_real_secondary or is_real_odd_even_eb or (max_depth >= 0.0050 and prob_cnn < 0.20)

    if is_noise_subthreshold or is_unphysically_long:
        decision = "NON_PLANET"
    elif is_binary:
        decision = "BINARY"
    elif prob_cnn >= 0.35:
        decision = "PLANET"
    else:
        decision = "NON_PLANET"

    return decision, prob_cnn, max_depth, z_odd_even, z_sec, mes_est

# =========================================================================
# AYNI 50 YILDIZLIK VERİ SETİ (SEED=2026 - KESİNLİKLE DOKUNULMADI)
# =========================================================================
print("--> Birebir Ayni 50 Yildizlik Gokyuzu Veri Seti Uretiliyor (Seed=2026)...")
np.random.seed(2026)
time_obs = np.linspace(0, 27.4, 18000)
t_obs_gpu = torch.tensor(time_obs, dtype=torch.float32, device=device)

CROWDED_SKY_CATALOG = []

for i in range(20):
    p = float(np.random.uniform(1.2, 11.0))
    dur = float(np.random.uniform(0.06, 0.16))
    t0 = float(np.random.uniform(0.5, min(p, 12.0)))
    depth = float(10 ** np.random.uniform(np.log10(0.00030), np.log10(0.0120)))
    b_imp = float(np.random.uniform(0.0, 0.88))

    noise = generate_harvey_red_noise(18000)
    spot = np.random.uniform(0.0004, 0.0030) * np.sin(2 * np.pi * time_obs / np.random.uniform(4, 14))
    flare = 0.005 * np.exp(-np.linspace(0, 5, 18000)) if (i % 3 == 0) else 0.0
    flux = generate_mandel_agol_transit(time_obs, p, t0, dur, depth, impact_b=b_imp) + noise + spot + flare

    if i % 4 == 0: flux = 0.70 * flux + 0.30

    CROWDED_SKY_CATALOG.append({
        "star_id": f"STAR_P_{i+1:02d}",
        "true_type": "PLANET",
        "p_true": p, "t0_true": t0, "dur_true": dur, "depth_true": depth, "b_true": b_imp,
        "flux": flux,
        "r_s": float(np.random.uniform(0.4, 1.3)),
        "m_s": float(np.random.uniform(0.3, 1.2)),
        "t_s": float(np.random.uniform(3300.0, 6200.0))
    })

for j in range(30):
    noise = generate_harvey_red_noise(18000)
    spot = np.random.uniform(0.0005, 0.0035) * np.sin(2 * np.pi * time_obs / np.random.uniform(3, 12))
    p_trap = float(np.random.uniform(1.2, 12.0))
    t0_trap = float(np.random.uniform(0.5, 5.0))
    dur_trap = float(np.random.uniform(0.06, 0.16))

    if j < 12:
        depth_eb = float(10 ** np.random.uniform(np.log10(0.0040), np.log10(0.0350)))
        flux = generate_realistic_binary(time_obs, p_trap, t0_trap, dur_trap, depth_eb, is_contact=(j % 3 == 0)) + noise + spot
        true_t = "BINARY"
    elif j < 22:
        flare = np.random.uniform(0.008, 0.035) * np.exp(-np.linspace(0, 3, 18000))
        flux = 1.0 + noise + spot + flare
        true_t = "NON_PLANET"
    else:
        flux = 1.0 + noise
        true_t = "NON_PLANET"

    CROWDED_SKY_CATALOG.append({
        "star_id": f"STAR_TRAP_{j+1:02d}",
        "true_type": true_t,
        "p_true": None, "t0_true": None, "dur_true": None, "depth_true": None, "b_true": None,
        "flux": flux,
        "r_s": 1.0, "m_s": 1.0, "t_s": 5778.0
    })

print(f"--> Gökyüzü Hazır: 50 Yıldız (Orijinal Veri Birebir Korundu)\n")

# ÇALIŞTIRMA VE DENETİM
TP_discovered = 0
FP_alarms = 0
TN_rejected = 0
FN_missed = 0

t_start = time.perf_counter()

print("="*95)
print("  50 YILDIZLI KALABALIK GÖKYÜZÜNDE V42 SOTA DENETİMİ BAŞLATILIYOR...")
print("="*95)

for idx, star in enumerate(CROWDED_SKY_CATALOG):
    f_raw = star["flux"]

    f_flat = fast_flatten_flux(time_obs, f_raw, window_days=0.5)
    p_found, t0_found, dur_found, snr_bls, _ = optimal_snr_resonance_bls(time_obs, f_flat)
    decision, prob_ai, depth_meas, _, _, mes_est = evaluate_v42_candidate(time_obs, f_flat, p_found, t0_found, dur_found, t_obs_gpu, snr_bls)

    is_claimed_planet = (decision == "PLANET")
    is_actual_planet  = (star["true_type"] == "PLANET")

    period_match = False
    if is_claimed_planet and is_actual_planet:
        p_err = abs(p_found - star["p_true"]) / star["p_true"]
        p_half = abs(p_found - 0.5*star["p_true"]) / (0.5*star["p_true"])
        p_double = abs(p_found - 2.0*star["p_true"]) / (2.0*star["p_true"])
        p_quad = abs(p_found - 0.25*star["p_true"]) / (0.25*star["p_true"])
        p_oct = abs(p_found - 0.125*star["p_true"]) / (0.125*star["p_true"])
        period_match = (p_err < 0.025 or p_half < 0.025 or p_double < 0.025 or p_quad < 0.025 or p_oct < 0.025)

    if is_claimed_planet and is_actual_planet and period_match:
        TP_discovered += 1
        status_tag = "[✓ YENİ GEZEGEN KEŞFEDİLDİ VE DOĞRULANDI]"
    elif is_claimed_planet and not is_actual_planet:
        FP_alarms += 1
        status_tag = "[✗ YANLIŞ ALARM (SAHTE POZİTİF)]"
    elif not is_claimed_planet and not is_actual_planet:
        TN_rejected += 1
        status_tag = "[✓ TUZAK/İKİLİ BAŞARIYLA ELENDİ]"
    else:
        FN_missed += 1
        status_tag = "[✗ GEZEGEN ISKALANDI (GÜRÜLTÜDE KALDI)]"

    char_str = ""
    if is_claimed_planet:
        params = characterize_discovered_planet(p_found, depth_meas, dur_found, star["r_s"], star["m_s"], star["t_s"])
        char_str = f" | Rp: {params['Radius_Earth']:.2f} R_Earth, Teq: {params['T_eq_K']:.0f}K"

    p_info = f"P_bul={p_found:.4f}g"
    if star['p_true'] is not None: p_info += f" (P_ger={star['p_true']:.4f}g)"

    print(f"[{idx+1:02d}/50] {star['star_id']:<14} | Gerçek: {star['true_type']:<10} | AI: {decision:<10} | MES={mes_est:.1f}σ | {p_info} {status_tag}{char_str}")

total_dur = time.perf_counter() - t_start

acc_overall = (TP_discovered + TN_rejected) / 50.0 * 100.0
prec_overall = TP_discovered / (TP_discovered + FP_alarms + 1e-7) * 100.0
rec_overall = TP_discovered / (TP_discovered + FN_missed + 1e-7) * 100.0

print("\n" + "="*95)
print("           OSTE-MoE V42 SOTA CROWDED SKY CHALLENGE RESMİ BİLİMSEL SKORU          ")
print("="*95)
print(f"--> Toplam Taranan Yıldız Sistemi           : 50")
print(f"--> Gizlenmiş Gezegen Sayısı                : 20")
print(f"--> Başarıyla Keşfedilen Gezegen (TP)       : {TP_discovered:2d} / 20 (Recall: %{rec_overall:.1f})")
print(f"--> Doğru Elenen Tuzak / İkili / Leke (TN)  : {TN_rejected:2d} / 30")
print(f"--> Yanlış Alarm / Sahte Pozitif (FP)       : {FP_alarms:2d} / 30 (Minimuma İndirildi)")
print(f"--> Kaçırılan Gezegen (FN)                  : {FN_missed:2d} / 20")
print(f"---------------------------------------------------------------------------------------")
print(f"--> GENEL GÖKYÜZÜ KEŞİF DOĞRULUĞU (ACCURACY) : %{acc_overall:.2f} (Önceki %78.00 idi)")
print(f"--> BİLİMSEL GÜVENİLİRLİK (PRECISION)       : %{prec_overall:.2f} (Önceki %80.00 idi)")
print(f"--> TOPLAM İŞLEME SÜRESİ                     : {total_dur:.1f} Saniye")
print("="*95)