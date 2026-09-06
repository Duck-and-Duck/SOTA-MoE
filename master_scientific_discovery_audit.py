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
from hq_data_utils import generate_mandel_agol_transit, generate_realistic_binary, generate_harvey_red_noise

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

socket.setdefaulttimeout(15)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*85)
print("  OSTE-MoE HIGH-PRECISION RECOVERY ENGINE (RESIDUAL NOISE-NORMALIZED)")
print(f"--> COMPUTATION ENGINE: {device.upper()} (RTX 3050 Ti)")
print("="*85)

# 1. 1D-CNN ASTRONET-HQ
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
print("--> [TAMAM]: AstroNet-HQ JIT Nöral Omurga Bellekte Hazır.\n")

# 2. GPU PREFIX-SUM ILE GURULTU BASTIRICI FAZ KATLAMA
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

def phase_fold_batch_gpu(t_t, f_t, p, t0, dur):
    phase = ((t_t - t0 + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase)
    s_ph, s_fl = phase[s_idx], f_t[s_idx]

    g_bins = torch.linspace(-0.5 * p, 0.5 * p, 201, device=device)
    g_raw = gpu_box_bin(g_bins, s_ph, s_fl)

    # NASA SPOC Standardı Dinamik Lokal Pencere: Transit Süresinin 3.5 Katı
    win_local = max(dur * 1.75, p * 0.04)
    l_bins = torch.linspace(-win_local, win_local, 61, device=device)
    l_raw = gpu_box_bin(l_bins, s_ph, s_fl)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm.view(1, 1, 201), l_norm.view(1, 1, 61)

# 3. YEREL REZİDÜEL STANDART SAPMA VE ANLAMLILIK KALKANI
def evaluate_candidate_robust(time_arr, flux_arr, p, t0, dur, t_gpu, f_gpu):
    # 1. Fazlama ve CNN Skoru
    g, l = phase_fold_batch_gpu(t_gpu, f_gpu, p, t0, dur)
    with torch.no_grad():
        prob_cnn = torch.sigmoid(frozen_vetter(g, l)).item()

    # 2. Yerel Rezidüel Beyaz Gürültü Hesabı (Leke Eğimlerinden Arındırılmış)
    # Kayan fark türevi ile makro yıldız lekesi dalgalanması elimine edilir (CDPP yöntemi)
    diff_flux = np.diff(flux_arr)
    sigma_white = (np.nanmedian(np.abs(diff_flux - np.nanmedian(diff_flux))) * 1.4826) / np.sqrt(2)

    # 3. Faz Maskelemesi
    phase = ((time_arr - t0 + 0.5 * p) % p) - 0.5 * p
    in_tr = np.abs(phase) < (dur / 2.0)
    tr_num = np.round((time_arr - t0) / p)

    odd_m = in_tr & (tr_num % 2 != 0)
    even_m = in_tr & (tr_num % 2 == 0)

    n_odd = np.sum(odd_m)
    n_even = np.sum(even_m)

    d_odd = float(1.0 - np.nanmedian(flux_arr[odd_m])) if n_odd > 2 else 0.0
    d_even = float(1.0 - np.nanmedian(flux_arr[even_m])) if n_even > 2 else 0.0

    se_odd = sigma_white / np.sqrt(max(1, n_odd))
    se_even = sigma_white / np.sqrt(max(1, n_even))
    se_diff = np.sqrt(se_odd**2 + se_even**2) + 1e-8

    z_odd_even = abs(d_odd - d_even) / se_diff

    # 4. Faz 0.5 İkincil Tepe (Sekonder Tutulma)
    sec_ph = ((time_arr - t0) % p) - 0.5 * p
    sec_m = np.abs(sec_ph) < (dur / 2.0)
    n_sec = np.sum(sec_m)
    d_sec = float(1.0 - np.nanmedian(flux_arr[sec_m])) if n_sec > 2 else 0.0
    se_sec = sigma_white / np.sqrt(max(1, n_sec)) + 1e-8
    z_secondary = d_sec / se_sec

    max_depth = max(d_odd, d_even, 1e-5)
    sec_ratio = d_sec / max_depth

    # BİLİMSEL KARAR MATRİSİ:
    # İkili Yıldız Kalkanı: Yalnızca istatistiksel olarak tartışmasız ikili tutulmaları veto eder
    is_binary_vetoed = (z_secondary >= 3.5 and sec_ratio >= 0.35) or \
                       (z_odd_even >= 4.0 and abs(d_odd - d_even)/max_depth >= 0.50)

    if is_binary_vetoed:
        decision_is_planet = False
    else:
        # CNN Morfoloji Kararı (Kalibre Bayesyen Eşik: 0.38)
        decision_is_planet = (prob_cnn >= 0.38)

    return decision_is_planet, prob_cnn, z_odd_even, z_secondary

# =========================================================================
# KULVAR 1: 2000 HEDEFLİK ADVERSARIAL GAUNTLET
# =========================================================================
print("="*85)
print("  KULVAR 1: 2000 HEDEFLIK ADVERSARIAL GAUNTLET (HER CESIT ZORLUK)")
print("="*85)

np.random.seed(999)
t_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(t_base, dtype=torch.float32, device=device)

TP_syn, FN_syn, TN_syn, FP_syn = 0, 0, 0, 0
probs_syn_planets, probs_syn_negatives = [], []
t0_syn = time.perf_counter()

for i in range(1000):
    p = np.random.uniform(1.2, 12.0)
    dur = np.random.uniform(0.06, 0.17)
    depth = 10 ** np.random.uniform(np.log10(0.00030), np.log10(0.0120))
    impact = np.random.uniform(0.0, 0.82)

    harvey = generate_harvey_red_noise(3000)
    spot = np.random.uniform(0.0003, 0.0025) * np.sin(2 * np.pi * t_base / np.random.uniform(4, 14))
    flare = 0.003 * np.exp(-np.linspace(0, 4, 3000)) if (i % 5 == 0) else 0.0
    poly_trend = 0.0005 * ((t_base / 27.4) ** 2)
    noise = harvey + spot + flare + poly_trend

    t0_sim = np.random.uniform(0.1, 1.5)
    flux = generate_mandel_agol_transit(t_base, p, t0_sim, dur, depth, impact_b=impact) + noise
    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)

    is_planet, prob, _, _ = evaluate_candidate_robust(t_base, flux, p, t0_sim, dur, t_gpu, f_gpu)
    probs_syn_planets.append(prob)
    if is_planet: TP_syn += 1
    else: FN_syn += 1

for i in range(1000):
    p = np.random.uniform(1.2, 12.0)
    dur = np.random.uniform(0.06, 0.17)
    depth = 10 ** np.random.uniform(np.log10(0.0006), np.log10(0.0180))

    harvey = generate_harvey_red_noise(3000)
    spot = np.random.uniform(0.0004, 0.0030) * np.sin(2 * np.pi * t_base / np.random.uniform(4, 14))
    poly_trend = 0.0005 * ((t_base / 27.4) ** 2)
    noise = harvey + spot + poly_trend
    t0_sim = np.random.uniform(0.1, 1.5)

    sub = i % 4
    if sub in [0, 1]:
        flux = generate_realistic_binary(t_base, p, t0_sim, dur, depth, is_contact=(sub == 1)) + noise
    elif sub == 2:
        tr = generate_mandel_agol_transit(t_base, p, t0_sim, dur, depth, impact_b=0.3)
        flux = (2.0 - tr) + noise
    else:
        flare = 0.005 * np.exp(-np.linspace(0, 3, 3000))
        flux = 1.0 + noise + flare

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    is_planet, prob, _, _ = evaluate_candidate_robust(t_base, flux, p, t0_sim, dur, t_gpu, f_gpu)
    probs_syn_negatives.append(prob)
    if is_planet: FP_syn += 1
    else: TN_syn += 1

lat_syn = (time.perf_counter() - t0_syn) * 1000
acc_syn = (TP_syn + TN_syn) / 2000.0 * 100.0
prec_syn = TP_syn / (TP_syn + FP_syn + 1e-7) * 100.0
rec_syn = TP_syn / (TP_syn + FN_syn + 1e-7) * 100.0

print(f"--> [2000 HEDEF TAMAMLANDI - {lat_syn/1000:.2f} sn]:")
print(f"    * Doğru Pozitif (TP): {TP_syn:4d} / 1000  |  Yanlış Negatif (FN): {FN_syn:4d} / 1000")
print(f"    * Doğru Negatif (TN): {TN_syn:4d} / 1000  |  Yanlış Pozitif (FP): {FP_syn:4d} / 1000")
print(f"    * SENTETİK GAUNTLET DOĞRULUĞU : %{acc_syn:.2f}")
print(f"    * KESİNLİK (PRECISION)        : %{prec_syn:.2f}")
print(f"    * YAKALAMA ORANI (RECALL)     : %{rec_syn:.2f}")

# =========================================================================
# KULVAR 2: 1000 ADET GERÇEK TELESKOP GÜRÜLTÜSÜ İLE ENJEKSİYON TESTİ
# =========================================================================
print("\n" + "="*85)
print("  KULVAR 2: 1000 ADET GERCEK TELESKOP GURULTUSU ILE ENJEKSIYON & RECOVERY")
print("="*85)

cache_dir = "_CACHE_LIGHTCURVES_"
os.makedirs(cache_dir, exist_ok=True)

def get_lightcurve_cached(tic, sector):
    cache_file = os.path.join(cache_dir, f"{tic}_s{sector}.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["t"], data["f"]
    try:
        search = lk.search_lightcurve(tic, mission="TESS", sector=sector)
        if len(search) == 0:
            search = lk.search_lightcurve(tic, mission="TESS")
        lc = search[0].download(quality_bitmask="hardest").remove_nans().flatten(window_length=301)
        t = np.asarray(lc.time.value, dtype=np.float64)
        f = np.asarray(lc.flux.value, dtype=np.float64)
        np.savez(cache_file, t=t, f=f)
        return t, f
    except Exception as e:
        print(f"    [UYARI]: {e}. Sentetik fallback devrede.")
        t = np.linspace(0, 27.4, 3000)
        f = 1.0 + generate_harvey_red_noise(3000)
        return t, f

real_t_pool, real_f_pool = get_lightcurve_cached("TIC 261136679", 14)
real_t_pool = real_t_pool[:3000]
real_f_pool = real_f_pool[:3000]
real_noise_res = real_f_pool - 1.0
real_t_gpu = torch.tensor(real_t_pool, dtype=torch.float32, device=device)

TP_real, FN_real, TN_real, FP_real = 0, 0, 0, 0
probs_inj_planets, probs_inj_negatives = [], []
t0_inj = time.perf_counter()

for i in range(500):
    p = np.random.uniform(1.2, 9.0)
    dur = np.random.uniform(0.07, 0.16)
    depth = 10 ** np.random.uniform(np.log10(0.00035), np.log10(0.0070))
    impact = np.random.uniform(0.0, 0.75)
    t0_target = real_t_pool[0] + np.random.uniform(0.1, p)

    tr = generate_mandel_agol_transit(real_t_pool, p, t0_target, dur, depth, impact_b=impact)
    inj_flux = tr + real_noise_res
    f_gpu = torch.tensor(inj_flux, dtype=torch.float32, device=device)

    is_planet, prob, _, _ = evaluate_candidate_robust(real_t_pool, inj_flux, p, t0_target, dur, real_t_gpu, f_gpu)
    probs_inj_planets.append(prob)
    if is_planet: TP_real += 1
    else: FN_real += 1

for i in range(500):
    p = np.random.uniform(1.2, 9.0)
    dur = np.random.uniform(0.07, 0.16)
    depth = 10 ** np.random.uniform(np.log10(0.0010), np.log10(0.0150))
    t0_target = real_t_pool[0] + np.random.uniform(0.1, p)

    if i % 2 == 0:
        eb = generate_realistic_binary(real_t_pool, p, t0_target, dur, depth)
        inj_flux = eb + real_noise_res
    else:
        inj_flux = 1.0 + real_noise_res

    f_gpu = torch.tensor(inj_flux, dtype=torch.float32, device=device)
    is_planet, prob, _, _ = evaluate_candidate_robust(real_t_pool, inj_flux, p, t0_target, dur, real_t_gpu, f_gpu)
    probs_inj_negatives.append(prob)
    if is_planet: FP_real += 1
    else: TN_real += 1

lat_inj = (time.perf_counter() - t0_inj) * 1000
acc_real = (TP_real + TN_real) / 1000.0 * 100.0
prec_real = TP_real / (TP_real + FP_real + 1e-7) * 100.0
rec_real = TP_real / (TP_real + FN_real + 1e-7) * 100.0

print(f"--> [1000 GERÇEK GÜRÜLTÜ TESTİ BİTTİ - {lat_inj/1000:.2f} sn]:")
print(f"    * Doğru Pozitif (TP): {TP_real:3d} / 500  |  Yanlış Negatif (FN): {FN_real:3d} / 500")
print(f"    * Doğru Negatif (TN): {TN_real:3d} / 500  |  Yanlış Pozitif (FP): {FP_real:3d} / 500")
print(f"    * GERÇEK GÜRÜLTÜ RECOVERY DOĞRULUĞU : %{acc_real:.2f}")
print(f"    * KESİNLİK (PRECISION)              : %{prec_real:.2f}")
print(f"    * YAKALAMA ORANI (RECALL)           : %{rec_real:.2f}")

# =========================================================================
# KULVAR 3: 5 KANONİK GERÇEK NASA HEDEFI (DONMA-KORUMALI ÖNBELLEKLEME)
# =========================================================================
print("\n" + "="*85)
print("  KULVAR 3: 5 KANONIK GERCEK NASA HEDEFI (GROUND TRUTH BENCHMARK)")
print("="*85)

CANONICAL_TARGETS = [
    {"name": "WASP-18 b",  "tic": "TIC 100100827", "sector": 2,  "period": 0.941452, "t0": 1354.45,   "dur": 0.09, "type": "PLANET"},
    {"name": "L 98-59 c",  "tic": "TIC 307210830", "sector": 2,  "period": 3.690621, "t0": 1356.2032, "dur": 0.07, "type": "PLANET"},
    {"name": "WASP-126 b", "tic": "TIC 25155310",  "sector": 1,  "period": 3.288800, "t0": 1327.52,   "dur": 0.11, "type": "PLANET"},
    {"name": "TESS EB 1",  "tic": "TIC 339607421", "sector": 2,  "period": 1.258200, "t0": 1354.10,   "dur": 0.08, "type": "BINARY"},
    {"name": "Quiet Star", "tic": "TIC 261136679", "sector": 14, "period": 3.500000, "t0": 1683.00,   "dur": 0.10, "type": "NON_PLANET"}
]

real_hits = 0
for tgt in CANONICAL_TARGETS:
    t_v, f_v = get_lightcurve_cached(tgt["tic"], tgt["sector"])
    t_v_gpu = torch.tensor(t_v, dtype=torch.float32, device=device)
    f_v_gpu = torch.tensor(f_v, dtype=torch.float32, device=device)

    is_planet_pred, p_val, z_oe, z_sec = evaluate_candidate_robust(t_v, f_v, tgt["period"], tgt["t0"], tgt["dur"], t_v_gpu, f_v_gpu)

    expected_planet = (tgt["type"] == "PLANET")
    hit = (is_planet_pred == expected_planet)
    if hit: real_hits += 1

    status = "BAŞARILI (İSABET)" if hit else "ISKALADI"
    print(f"  * {tgt['name']:<18} | Gerçek: {tgt['type']:<10} | AI: %{p_val*100:5.1f} | Z_oe: {z_oe:4.1f}σ | Z_sec: {z_sec:4.1f}σ -> {status}")

# =========================================================================
# GENEL BİLİMSEL DENETİM RAPORU
# =========================================================================
print("\n" + "="*85)
print("                   GENEL BILIMSEL AUDIT BİLANÇOSU                        ")
print("="*85)
print(f"1. 2000 Hedeflik Adversarial Gauntlet : %{acc_syn:.2f} Doğruluk  (Precision: %{prec_syn:.2f} | Recall: %{rec_syn:.2f})")
print(f"2. 1000 Hedeflik Gerçek TESS Recovery : %{acc_real:.2f} Doğruluk  (Precision: %{prec_real:.2f} | Recall: %{rec_real:.2f})")
print(f"3. 5 Kanonik Hedef Doğrulaması        : {real_hits} / 5 Tam İsabet (%{real_hits/5.0*100:.1f})")
print("-"*85)
composite_score = (acc_syn * 0.40) + (acc_real * 0.40) + ((real_hits/5.0*100) * 0.20)
print(f"--> BİLEŞİK BİLİMSEL GÜVEN SKORU      : %{composite_score:.2f}")

if composite_score >= 90.0 and rec_real >= 85.0 and real_hits == 5:
    print("\n[BİLİMSEL HÜKÜM: %90 BİLEŞİK VE %85+ RECOVERY BARAJI BAŞARIYLA AŞILDI]:")
    print("Model; rezidüel beyaz gürültü normalizasyonu ve dinamik pencereleme ile")
    print("gerçek uzay teleskobu gürültüsünde sığ gezegenleri kaçırmadan keşif yapabilir duruma gelmiştir.")
else:
    print(f"\n[BİLİMSEL HÜKÜM: TEST TAMAMLANDI - DEĞERLENDİRME BİTTİ]")
print("="*85)