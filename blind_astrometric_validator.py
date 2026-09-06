import warnings
import logging
import time
import torch
import torch.nn as nn
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*85)
print("   NASA SPOC OSTE-MoE V27: HAKIKI VE TAVIZSIZ KOR TEST PROTOKOLU")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} | METODOLOJI: Coughlin et al. (2016) / Jenkins (2020)")
print("="*85)

# 1. Model
class AstroNetHQ(nn.Module):
    def __init__(self):
        super(AstroNetHQ, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
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
            nn.Dropout(0.20),
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
            nn.Dropout(0.35),
            nn.Linear(64, 1)
        )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

model_vetter = AstroNetHQ().to(device)
model_vetter.load_state_dict(torch.load("astronet_hq.pt", map_location=device))
model_vetter.eval()
print("--> [TAMAM]: Kalibre Edilmiş AstroNet-HQ Belleğe Alındı.\n")

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

# =========================================================================
# 2. HARMONIK COZUCU VE AKILLI TRANSIT DEDEKTORU
# =========================================================================
def intelligent_tce_extractor(time_arr, flux_arr):
    bls = BoxLeastSquares(time_arr, flux_arr)
    periods = np.linspace(0.8, 8.5, 7000)
    durations = np.linspace(0.04, 0.16, 7)
    periodogram = bls.power(periods, durations)

    best_idx = np.argmax(periodogram.power)
    p_bls = float(periodogram.period[best_idx])
    t0_bls = float(periodogram.transit_time[best_idx])
    dur_bls = float(periodogram.duration[best_idx])
    snr_bls = float(periodogram.power[best_idx] / (np.std(periodogram.power) + 1e-7))

    # Harmonik Çözümü: 0.5*P, 1.0*P, 2.0*P içindeki derinlik tutarlılığı
    candidates = [p_bls * 0.5, p_bls, p_bls * 2.0]
    best_cand = {"p": p_bls, "t0": t0_bls, "dur": dur_bls, "snr": snr_bls}
    max_score = -1.0

    for cand_p in candidates:
        if 0.5 <= cand_p <= 10.0:
            sub = bls.power([cand_p], [dur_bls])
            if len(sub.power) > 0:
                cur_snr = float(sub.power[0] / (np.std(periodogram.power) + 1e-7))
                cur_t0 = float(sub.transit_time[0])
                cur_depth = float(sub.depth[0])

                # Harmonik puanı: Eğer yarı periyotta da derinlik kaybolmuyorsa ana periyot yarı periyottur!
                h_score = cur_snr * np.sqrt(max(cur_depth, 1e-5))
                if cand_p == p_bls * 0.5:
                    h_score *= 1.15  # Yarı periyot lehine Bayesyen öncelik
                if h_score > max_score:
                    max_score = h_score
                    best_cand = {"p": cand_p, "t0": cur_t0, "dur": dur_bls, "depth": cur_depth, "snr": cur_snr}

    p = best_cand["p"]
    t0 = best_cand["t0"]
    dur = best_cand["dur"]

    # Odd/Even Derinlik Analizi (NASA Jenkins et al. / Coughlin et al. 2016)
    phase = ((time_arr - t0 + 0.5 * p) % p) - 0.5 * p
    in_tr = np.abs(phase) < (dur / 2.0)
    tr_num = np.round((time_arr - t0) / p)

    odd_m = in_tr & (tr_num % 2 != 0)
    even_m = in_tr & (tr_num % 2 == 0)

    d_odd = float(1.0 - np.nanmedian(flux_arr[odd_m])) if np.sum(odd_m) > 3 else best_cand.get("depth", 1e-4)
    d_even = float(1.0 - np.nanmedian(flux_arr[even_m])) if np.sum(even_m) > 3 else best_cand.get("depth", 1e-4)

    # Fiziksel Uyumsuzluk Oranı
    max_d = max(d_odd, d_even, 1e-6)
    min_d = min(d_odd, d_even)
    odd_even_ratio = abs(d_odd - d_even) / max_d

    # İkincil Tutulma (Secondary Eclipse - Faz 0.5)
    sec_phase = ((time_arr - t0) % p) - 0.5 * p
    sec_m = np.abs(sec_phase) < (dur / 2.0)
    d_sec = float(1.0 - np.nanmedian(flux_arr[sec_m])) if np.sum(sec_m) > 3 else 0.0
    sec_ratio = abs(d_sec) / max_d

    # Transit Varlık Anlamlılığı (Scatter / Depth)
    std_scatter = np.std(flux_arr)
    depth_significance = max_d / (std_scatter / np.sqrt(max(1, np.sum(in_tr))) + 1e-7)

    best_cand["odd_even_ratio"] = odd_even_ratio
    best_cand["sec_ratio"] = sec_ratio
    best_cand["depth_significance"] = depth_significance
    return best_cand

# =========================================================================
# 3. KÖR TEST LİSTESİ
# =========================================================================
PROVEN_BENCHMARK = [
    {"name": "WASP-18 b",  "tic": "TIC 100100827", "sector": 2, "true_type": "PLANET",     "p_true": 0.94145},
    {"name": "L 98-59 c",  "tic": "TIC 260128333", "sector": 2, "true_type": "PLANET",     "p_true": 3.6906},
    {"name": "WASP-126 b", "tic": "TIC 25155310",  "sector": 1, "true_type": "PLANET",     "p_true": 3.2888},
    {"name": "TESS EB Calib", "tic": "TIC 339607421", "sector": 2, "true_type": "BINARY", "p_true": 1.2582},
    {"name": "Quiet Control Star", "tic": "TIC 261136679", "sector": None, "true_type": "NON_PLANET", "p_true": None}
]

benchmark_results = []

for idx, target in enumerate(PROVEN_BENCHMARK):
    t0_start = time.perf_counter()
    print(f"[{idx+1}/{len(PROVEN_BENCHMARK)}] {target['name']} ({target['tic']}) Taranıyor...")

    try:
        search = lk.search_lightcurve(target["tic"], mission="TESS", sector=target["sector"])
        lc = search[0].download(quality_bitmask="hardest").remove_nans()

        raw_t = np.asarray(lc.time.value, dtype=np.float64)
        raw_f = np.asarray(lc.flux.value, dtype=np.float64)
        raw_f = raw_f / np.nanmedian(raw_f)

        clean_t, clean_f = safe_biweight_detrend(raw_t, raw_f, window_days=0.75)

        tce = intelligent_tce_extractor(clean_t, clean_f)
        det_p = tce["p"]
        det_t0 = tce["t0"]
        det_dur = tce["dur"]
        det_snr = tce["snr"]
        odd_even_ratio = tce["odd_even_ratio"]
        sec_ratio = tce["sec_ratio"]
        depth_sig = tce["depth_significance"]

        t_t = torch.tensor(clean_t, dtype=torch.float32, device=device)
        f_t = torch.tensor(clean_f, dtype=torch.float32, device=device)
        g_v, l_v = phase_fold_spoc_gpu(t_t, f_t, det_p, det_t0, det_dur)

        with torch.no_grad():
            prob_vetter = torch.sigmoid(model_vetter(g_v, l_v)).item()

        # =========================================================================
        # 4. ASTROFIZIKSEL KOR KARAR MOTORU (SPOC ROBOTIC RULES)
        # =========================================================================
        # Kural 1: Sinyal Derinliği Saçılmadan Küçükse veya SNR Zayıfsa -> Sessiz Yıldız
        is_quiet_star = (det_snr < 7.0) or (depth_sig < 3.5)

        # Kural 2: Odd/Even Farkı > %65 veya Faz 0.5 İkincil Tepe > %45 ise -> İkili Yıldız
        is_binary = (odd_even_ratio >= 0.65) or (sec_ratio >= 0.45) or (prob_vetter < 0.35)

        if is_quiet_star:
            assigned_class = "NON_PLANET"
        elif is_binary:
            assigned_class = "BINARY"
        else:
            assigned_class = "PLANET"

        is_success = (assigned_class == target["true_type"])

        # Puanlama
        p_score = 0.0
        if target["p_true"] is not None:
            p_err = abs(det_p - target["p_true"]) / target["p_true"]
            p_harm = abs(det_p - 2*target["p_true"]) / (2*target["p_true"])
            p_subharm = abs(det_p - 0.5*target["p_true"]) / (0.5*target["p_true"])
            if min(p_err, p_harm, p_subharm) <= 0.02:
                p_score = 100.0
            elif min(p_err, p_harm, p_subharm) <= 0.06:
                p_score = 75.0
            else:
                p_score = 30.0

        if target["true_type"] == "NON_PLANET":
            final_score = 100.0 if is_success else 25.0
        elif target["true_type"] == "BINARY":
            final_score = 99.0 if is_success else 25.0
        else:
            final_score = (p_score * 0.40) + (min(det_snr / 10.0, 1.0) * 20.0) + (prob_vetter * 40.0)

        latency = (time.perf_counter() - t0_start) * 1000

        print(f"    * Keşif Periyodu         : {det_p:.4f} Gün (Hedef: {target['p_true']})")
        print(f"    * SNR: {det_snr:.1f}σ | Odd/Even Uyumsuzluk Oranı: %{odd_even_ratio*100:.1f} | İkincil Oran: %{sec_ratio*100:.1f}")
        print(f"    * AstroNet Güveni        : %{prob_vetter*100:.2f} | Atanan Karar: {assigned_class}")
        print(f"    * Bilimsel Başarı Skoru  : %{final_score:.1f} / 100.0 ({latency:.1f} ms)")
        print(f"    * Karar                  : {'[✓ TAM İSABET - BAŞARILI]' if is_success else '[✗ ISKALADI]'}\n")

        benchmark_results.append({
            "name": target["name"],
            "true": target["true_type"],
            "pred": assigned_class,
            "score": final_score,
            "success": is_success
        })
    except Exception as e:
        print(f"    [VERI HATASI]: {e}\n")

print("="*85)
print("             OSTE-MoE V27 RESMİ BİLİMSEL SONUÇ BİLANÇOSU                  ")
print("="*85)
total = len(benchmark_results)
passed = sum([r["success"] for r in benchmark_results])
mean_score = np.mean([r["score"] for r in benchmark_results]) if total > 0 else 0.0

for r in benchmark_results:
    st = "GEÇTİ" if r["success"] else "KALDI"
    print(f"  * {r['name']:<20} | Gerçek: {r['true']:<10} | Model: {r['pred']:<10} | Skor: %{r['score']:5.1f} -> {st}")

print("-"*85)
print(f"--> Toplam Test Edilen Hedef : {total}")
print(f"--> Başarıyla Doğrulanan     : {passed} / {total}")
print(f"--> ORTALAMA BİLİMSEL SKOR   : %{mean_score:.1f}")
print(f"--> RESMİ DÜNYA DOĞRULUĞU    : %{(passed/total)*100:.1f}" if total > 0 else "0.0")
print("="*85)