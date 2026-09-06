import warnings
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*85)
print("   OSTE-MoE V28: 4 KATMANLI BILIMSEL TEST VE DOGRULAMA SUITE")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} | MIMARI: 3-Sinifli Multi-Modal AstroNet")
print("="*85)

class MultiModalAstroNet(nn.Module):
    def __init__(self):
        super(MultiModalAstroNet, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
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
            nn.Dropout(0.15),
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
        self.phys_dense = nn.Sequential(
            nn.Linear(5, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 15 + 32 * 15 + 16, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.30),
            nn.Linear(64, 3)
        )

    def forward(self, g, l, phys):
        g_feat = self.global_conv(g)
        l_feat = self.local_conv(l)
        p_feat = self.phys_dense(phys)
        return self.classifier(torch.cat([g_feat, l_feat, p_feat], dim=1))

model = MultiModalAstroNet().to(device)
model.load_state_dict(torch.load("astronet_multimodal.pt", map_location=device))
model.eval()
print("--> [TAMAM]: Multi-Modal Nöral Ağ Belleğe Alındı.\n")

def safe_biweight_detrend(time_arr, flux_arr, window_days=0.75):
    t_clean = np.asarray(time_arr, dtype=np.float64)
    f_clean = np.asarray(flux_arr, dtype=np.float64)
    valid = np.isfinite(t_clean) & np.isfinite(f_clean)
    t_clean, f_clean = t_clean[valid], f_clean[valid]

    dt = np.median(np.diff(t_clean))
    win_pts = max(15, int(window_days / dt))
    if win_pts % 2 == 0: win_pts += 1

    pad = win_pts // 2
    f_padded = np.pad(f_clean, pad, mode='reflect')

    step = max(1, win_pts // 6)
    idx_samples = np.arange(0, len(f_clean), step)
    med_samples = [np.median(f_padded[i : i + win_pts]) for i in idx_samples]

    trend = np.interp(np.arange(len(f_clean)), idx_samples, med_samples)
    return t_clean, f_clean / (trend + 1e-8)

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

def phase_fold_spoc_gpu(time_t, flux_t, period, t0, duration):
    phase = ((time_t - t0 + 0.5 * period) % period) - (0.5 * period)
    sorted_idx = torch.argsort(phase)
    sorted_phase, sorted_flux = phase[sorted_idx], flux_t[sorted_idx]

    global_bins = torch.linspace(-0.5 * period, 0.5 * period, 201, device=device)
    g_raw = interp1d_gpu(global_bins, sorted_phase, sorted_flux)

    half_dur = max(duration * 2.5, period * 0.035)
    local_bins = torch.linspace(-half_dur, half_dur, 61, device=device)
    l_raw = interp1d_gpu(local_bins, sorted_phase, sorted_flux)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm.view(1, 1, 201).contiguous(), l_norm.view(1, 1, 61).contiguous()

def extract_advanced_tce(time_arr, flux_arr):
    bls = BoxLeastSquares(time_arr, flux_arr)
    periods = np.linspace(0.8, 8.5, 7000)
    durations = np.linspace(0.04, 0.16, 7)
    periodogram = bls.power(periods, durations)

    best_idx = np.argmax(periodogram.power)
    p_bls = float(periodogram.period[best_idx])
    t0_bls = float(periodogram.transit_time[best_idx])
    dur_bls = float(periodogram.duration[best_idx])
    snr_bls = float(periodogram.power[best_idx] / (np.std(periodogram.power) + 1e-7))

    # Alt-Harmonik / Yarı Periyot Analizi
    cand_half = p_bls * 0.5
    final_p, final_t0, final_dur = p_bls, t0_bls, dur_bls

    if cand_half >= 0.7:
        sub_grid = np.linspace(cand_half * 0.98, cand_half * 1.02, 100)
        sub_pow = bls.power(sub_grid, [dur_bls])
        sub_best = np.argmax(sub_pow.power)
        sub_snr = float(sub_pow.power[sub_best] / (np.std(periodogram.power) + 1e-7))
        if sub_snr >= snr_bls * 0.85:
            final_p = float(sub_pow.period[sub_best])
            final_t0 = float(sub_pow.transit_time[sub_best])

    phase = ((time_arr - final_t0 + 0.5 * final_p) % final_p) - 0.5 * final_p
    in_tr = np.abs(phase) < (final_dur / 2.0)
    tr_num = np.round((time_arr - final_t0) / final_p)

    odd_m = in_tr & (tr_num % 2 != 0)
    even_m = in_tr & (tr_num % 2 == 0)

    d_odd = float(1.0 - np.nanmedian(flux_arr[odd_m])) if np.sum(odd_m) > 3 else 1e-4
    d_even = float(1.0 - np.nanmedian(flux_arr[even_m])) if np.sum(even_m) > 3 else 1e-4
    max_d = max(d_odd, d_even, 1e-6)
    odd_even_ratio = abs(d_odd - d_even) / max_d

    sec_ph = ((time_arr - final_t0) % final_p) - 0.5 * final_p
    sec_m = np.abs(sec_ph) < (final_dur / 2.0)
    d_sec = float(1.0 - np.nanmedian(flux_arr[sec_m])) if np.sum(sec_m) > 3 else 0.0
    sec_ratio = abs(d_sec) / max_d

    std_scatter = np.std(flux_arr)
    d_to_scatter = max_d / (std_scatter + 1e-7)

    return {
        "p": final_p,
        "t0": final_t0,
        "dur": final_dur,
        "depth": max_d,
        "snr": snr_bls,
        "odd_even": odd_even_ratio,
        "sec_ratio": sec_ratio,
        "d_to_scatter": d_to_scatter
    }

def infer_target(time_arr, flux_arr):
    clean_t, clean_f = safe_biweight_detrend(time_arr, flux_arr)
    tce = extract_advanced_tce(clean_t, clean_f)

    t_t = torch.tensor(clean_t, dtype=torch.float32, device=device)
    f_t = torch.tensor(clean_f, dtype=torch.float32, device=device)
    g_v, l_v = phase_fold_spoc_gpu(t_t, f_t, tce["p"], tce["t0"], tce["dur"])

    phys = torch.tensor([[
        np.clip(tce["snr"] / 20.0, 0.0, 2.0),
        np.log10(max(tce["depth"], 1e-6)),
        tce["odd_even"],
        tce["sec_ratio"],
        np.clip(tce["d_to_scatter"] / 10.0, 0.0, 2.0)
    ]], device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(g_v, l_v, phys)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Sınıflar: 0=NOISE/NON_PLANET, 1=PLANET, 2=BINARY
    pred_idx = np.argmax(probs)
    class_map = {0: "NON_PLANET", 1: "PLANET", 2: "BINARY"}
    return class_map[pred_idx], probs, tce

# =========================================================================
# TEST 1: AGIR PARAZITLI, LEKELI VE FLARE'LI SENTETIK STRES TESTI (100 ADET)
# =========================================================================
print("--- [BOLUM 1/4]: AGIR PARAZITLI SENTETIK STRES TESTI (100 NUMUNE) ---")
np.random.seed(101)
p1_correct = 0

for i in range(100):
    t_sim = np.linspace(0, 27.4, 3000)
    p_sim = np.random.uniform(2.0, 6.5)
    dur_sim = 0.10
    noise = np.random.normal(0, 0.00045, 3000) + 0.0012 * np.sin(2 * np.pi * t_sim / 5.5)

    if i < 50:
        # Gerçek Gezegen + Flare
        u_p = 0.0018 * np.maximum(0.0, 1.0 - ((t_sim % p_sim - p_sim/2) / (dur_sim/2))**2)
        f_sim = 1.0 + noise - u_p
        f_sim[1200:1230] += 0.003 * np.exp(-np.linspace(0, 2, 30)) # Flare
        true_c = "PLANET"
    else:
        # İkili Yıldız
        v_p = 0.0035 * np.maximum(0.0, 1.0 - np.abs((t_sim % p_sim - p_sim/2) / (dur_sim/2)))
        f_sim = 1.0 + noise - v_p
        true_c = "BINARY"

    c_pred, _, _ = infer_target(t_sim, f_sim)
    if c_pred == true_c: p1_correct += 1

print(f"--> Bölüm 1 Doğruluğu: %{p1_correct:.1f} (100/100 Ağır Parazitli Sinyal)\n")

# =========================================================================
# TEST 2: GERCEK NASA VERILERININ SIMULE REKREASYONU (WASP-18, L 98-59, WASP-126)
# =========================================================================
print("--- [BOLUM 2/4]: GERCEK HEDEFLERIN SIMULE REKREASYON TESTI ---")
p2_targets = [
    {"name": "Sim-WASP-18b", "p": 0.9414, "depth": 0.0092, "type": "PLANET"},
    {"name": "Sim-L 98-59c",  "p": 3.6906, "depth": 0.00085,"type": "PLANET"},
    {"name": "Sim-WASP-126b","p": 3.2888, "depth": 0.0014, "type": "PLANET"},
    {"name": "Sim-TESS EB",  "p": 1.2582, "depth": 0.0250, "type": "BINARY"}
]

p2_correct = 0
for tgt in p2_targets:
    t_sim = np.linspace(0, 27.4, 3000)
    noise = np.random.normal(0, 0.00035, 3000)
    phase = (t_sim % tgt["p"]) - (tgt["p"] / 2.0)
    in_tr = np.abs(phase) < 0.05
    f_sim = 1.0 + noise
    if tgt["type"] == "PLANET":
        f_sim[in_tr] -= tgt["depth"] * np.sqrt(np.maximum(0.0, 1.0 - (phase[in_tr]/0.05)**2))
    else:
        f_sim[in_tr] -= tgt["depth"] * np.maximum(0.0, 1.0 - np.abs(phase[in_tr]/0.05))
        sec_m = np.abs(phase - tgt["p"]/2.0) < 0.05
        f_sim[sec_m] -= (tgt["depth"] * 0.5) * np.maximum(0.0, 1.0 - np.abs((phase[sec_m]-tgt["p"]/2.0)/0.05))

    c_pred, pr, _ = infer_target(t_sim, f_sim)
    is_hit = (c_pred == tgt["type"])
    if is_hit: p2_correct += 1
    print(f"  * {tgt['name']:<15} | Beklenen: {tgt['type']:<8} | Tahmin: {c_pred:<10} -> {'BASARILI' if is_hit else 'ISKALADI'}")

print(f"--> Bölüm 2 Doğruluğu: %{(p2_correct/len(p2_targets))*100:.1f}\n")

# =========================================================================
# TEST 3: BILMEDIGINE BILMIYORUM/GURULTU DEME TESTI (OUT-OF-DISTRIBUTION NOISE)
# =========================================================================
print("--- [BOLUM 3/4]: SIFIR TRANSIT VE DEDEKTER GURULTUSU TAHLIYE TESTI ---")
p3_correct = 0
for _ in range(50):
    t_sim = np.linspace(0, 27.4, 3000)
    # Tamamen rastgele sinüzoidal aletsel dalgalanmalar + kırmızı gürültü (Transit YOK)
    pure_noise = 1.0 + np.random.normal(0, 0.0005, 3000) + 0.0015 * np.sin(2 * np.pi * t_sim / np.random.uniform(2, 9))
    c_pred, pr, _ = infer_target(t_sim, pure_noise)
    if c_pred == "NON_PLANET":
        p3_correct += 1

print(f"--> Bölüm 3 Doğruluğu: %{(p3_correct/50)*100:.1f} (Saf Gürültüde Halüsinasyon Görmeme Oranı)\n")

# =========================================================================
# TEST 4: GERCEK NASA MAST CANLI DOGRULAMA TESTI
# =========================================================================
print("--- [BOLUM 4/4]: CANLI NASA TESS TELESKOP ARSIV DOGRULAMASI ---")
REAL_TARGETS = [
    {"name": "WASP-18 b",  "tic": "TIC 100100827", "sector": 2, "true": "PLANET"},
    {"name": "L 98-59 c",  "tic": "TIC 260128333", "sector": 2, "true": "PLANET"},
    {"name": "WASP-126 b", "tic": "TIC 25155310",  "sector": 1, "true": "PLANET"},
    {"name": "TESS EB Calib", "tic": "TIC 339607421", "sector": 2, "true": "BINARY"},
    {"name": "Quiet Control Star", "tic": "TIC 261136679", "sector": None, "true": "NON_PLANET"}
]

real_correct = 0
for tgt in REAL_TARGETS:
    try:
        search = lk.search_lightcurve(tgt["tic"], mission="TESS", sector=tgt["sector"])
        lc = search[0].download(quality_bitmask="hardest").remove_nans()
        raw_t = np.asarray(lc.time.value, dtype=np.float64)
        raw_f = np.asarray(lc.flux.value, dtype=np.float64)
        raw_f = raw_f / np.nanmedian(raw_f)

        c_pred, probs, tce = infer_target(raw_t, raw_f)
        is_hit = (c_pred == tgt["true"])
        if is_hit: real_correct += 1

        print(f"  * {tgt['name']:<20} | Gercek: {tgt['true']:<10} | Karar: {c_pred:<10} | Periyot: {tce['p']:6.3f}g -> {'[✓ GECTI]' if is_hit else '[✗ KALDI]'}")
    except Exception as e:
        print(f"  * {tgt['name']} Hatasi: {e}")

print(f"\n--> Bölüm 4 Gerçek Veri Doğruluğu: %{(real_correct/len(REAL_TARGETS))*100:.1f}")

print("\n" + "="*85)
print("                   GENEL NIHAI BILANCO RAPORU                            ")
print("="*85)
overall_score = (p1_correct + (p2_correct/len(p2_targets)*100) + (p3_correct/50*100) + (real_correct/len(REAL_TARGETS)*100)) / 4.0
print(f"1. Ağır Parazitli Stres Skoru     : %{p1_correct:.1f}")
print(f"2. Gerçek Hedef Simülasyon Skoru  : %{(p2_correct/len(p2_targets))*100:.1f}")
print(f"3. Anti-Halüsinasyon / OOD Skoru  : %{(p3_correct/50)*100:.1f}")
print(f"4. Canlı NASA TESS Doğruluğu      : %{(real_correct/len(REAL_TARGETS))*100:.1f}")
print("-"*85)
print(f"--> BİLEŞİK BİLİMSEL GÜVEN SKORU  : %{overall_score:.1f}")
print("="*85)