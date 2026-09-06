import warnings
import logging
import time
import os
import torch
import torch.nn as nn
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*75)
print("  NASA KOR ASTRONOMIK DOGRULAMA V2 (HARMONIC-FREE & ROBUST DE-CORRELATION)")
print(f"--> DONANIM: {device.upper()} | GERCEK TELESKOP ZINCIRI")
print("="*75)

# En iyi modelin mimarisi
class AstroNetAdaptive(nn.Module):
    def __init__(self, is_hq=False):
        super(AstroNetAdaptive, self).__init__()
        self.is_hq = is_hq
        if is_hq:
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
        else:
            self.global_conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.MaxPool1d(2),
                nn.Flatten()
            )
            self.local_conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.MaxPool1d(2),
                nn.Flatten()
            )
            self.fc = nn.Sequential(
                nn.Linear(32 * 50 + 32 * 15, 64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

# Modeli yukle
best_model_name = "astronet_production.pt"
is_hq_mode = ("hq" in best_model_name)
model = AstroNetAdaptive(is_hq=is_hq_mode).to(device)
model.load_state_dict(torch.load(best_model_name, map_location=device))
model.eval()
print(f"--> Uretim Modeli Aktif: {best_model_name}\n")

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

def phase_fold_and_zscore(time_t, flux_t, period, t0):
    phase = ((time_t - t0 + 0.5 * period) % period) - (0.5 * period)
    sorted_idx = torch.argsort(phase)
    sorted_phase = phase[sorted_idx]
    sorted_flux = flux_t[sorted_idx]

    global_bins = torch.linspace(-0.5 * period, 0.5 * period, 201, device=device)
    g_raw = interp1d_gpu(global_bins, sorted_phase, sorted_flux)

    dur_approx = period * 0.05
    local_bins = torch.linspace(-dur_approx, dur_approx, 61, device=device)
    l_raw = interp1d_gpu(local_bins, sorted_phase, sorted_flux)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm.view(1, 1, 201).contiguous(), l_norm.view(1, 1, 61).contiguous()

def robust_harmonic_free_bls(time_arr, flux_arr):
    """
    BLS'in 2P ve P/2 harmonik tuzaklarina dusmesini engelleyen
    sub-harmonic derinlik karsilastirmali arama motoru (Kovacs et al. / SPOC standardi)
    """
    bls = BoxLeastSquares(time_arr, flux_arr)
    periods = np.linspace(0.8, 8.5, 3000)
    durations = np.linspace(0.04, 0.15, 6)
    periodogram = bls.power(periods, durations)

    best_idx = np.argmax(periodogram.power)
    cand_p = float(periodogram.period[best_idx])
    cand_t0 = float(periodogram.transit_time[best_idx])
    cand_depth = float(periodogram.depth[best_idx])
    cand_power = float(periodogram.power[best_idx])

    # Harmonik Kontrolü: cand_p / 2 gercek periyot mu?
    for test_p in [cand_p / 2.0, cand_p / 3.0]:
        if test_p >= 0.8:
            test_res = bls.power(np.array([test_p]), durations)
            test_power = float(np.max(test_res.power))
            if test_power >= 0.85 * cand_power:
                cand_p = test_p
                cand_t0 = float(test_res.transit_time[np.argmax(test_res.power)])
                cand_depth = float(test_res.depth[np.argmax(test_res.power)])
                break

    std_pow = np.std(periodogram.power)
    snr = cand_power / (std_pow if std_pow > 0 else 1e-7)
    return cand_p, cand_t0, cand_depth, snr

PROVEN_GROUND_TRUTH = [
    {"name": "WASP-18 b",  "tic": "TIC 100100827", "sector": 2, "true": "PLANET"},
    {"name": "L 98-59 c",   "tic": "TIC 260128333", "sector": 2, "true": "PLANET"},
    {"name": "WASP-126 b", "tic": "TIC 25155310",  "sector": 1, "true": "PLANET"},
    {"name": "TESS EB 1",  "tic": "TIC 339607421", "sector": 2, "true": "BINARY"},
    {"name": "TESS EB 2",  "tic": "TIC 48227288",  "sector": 2, "true": "BINARY"},
    {"name": "Quiet Star", "tic": "TIC 261136679", "sector": 2, "true": "NON_PLANET"} # Guvenli Sektor 2
]

results = []

for idx, tgt in enumerate(PROVEN_GROUND_TRUTH):
    t0_scan = time.perf_counter()
    print(f"[{idx+1}/{len(PROVEN_GROUND_TRUTH)}] {tgt['name']} ({tgt['tic']}) Taranıyor...")
    try:
        search = lk.search_lightcurve(tgt["tic"], mission="TESS", sector=tgt["sector"])
        if len(search) == 0:
            search = lk.search_lightcurve(tgt["tic"], mission="TESS")
        lc = search[0].download(quality_bitmask="hardest").remove_nans()

        # SPOC Eğim Temizliği
        flat = lc.flatten(window_length=101)
        t_arr = flat.time.value
        f_arr = flat.flux.value

        # 1. KÖR HARMONİKSİZ TRANSİT ARAMA (Modele periyot verilmez!)
        det_p, det_t0, det_d, det_snr = robust_harmonic_free_bls(t_arr, f_arr)
        print(f"    * Kör Sinyal Tespiti       : P = {det_p:.4f} Gün | Derinlik = {det_d*1e6:.1f} ppm | SNR = {det_snr:.1f}σ")

        # 2. SAF GPU FAZLAMA
        t_t = torch.tensor(t_arr, dtype=torch.float32, device=device)
        f_t = torch.tensor(f_arr, dtype=torch.float32, device=device)
        g_v, l_v = phase_fold_and_zscore(t_t, f_t, det_p, det_t0)

        # 3. YAPAY ZEKA ÇIKARIMI
        with torch.no_grad():
            prob = torch.sigmoid(model(g_v, l_v)).item()

        # 4. KARAR KAPISI
        if det_snr < 6.0:
            decision = "NON_PLANET"
        else:
            decision = "PLANET" if prob >= 0.50 else "BINARY"

        is_success = (decision == tgt["true"])
        lat = (time.perf_counter() - t0_scan) * 1000

        print(f"    * Yapay Zeka Güven Skoru  : %{prob*100:.2f} (Tahmin: {decision})")
        print(f"    * Karar Sonucu            : {'✓ BASARILI (ISABET)' if is_success else '✗ ISKALADI'} ({lat:.1f} ms)\n")

        results.append({
            "name": tgt["name"],
            "true": tgt["true"],
            "pred": decision,
            "prob": prob,
            "correct": is_success
        })
    except Exception as e:
        print(f"    [HATA]: {e}\n")

print("="*75)
print("            NASA RESMI KOR DOGRULAMA V2 SONUC TABLOSU                     ")
print("="*75)
total_tested = len(results)
correct_total = sum([r["correct"] for r in results])
acc = (correct_total / total_tested) * 100.0 if total_tested > 0 else 0.0

for r in results:
    status = "GECTI (ISABET)" if r["correct"] else "KALDI"
    print(f"  * {r['name']:<18} | Gercek: {r['true']:<11} | AI: {r['pred']:<11} | Prob: %{r['prob']*100:5.1f} -> {status}")

print("-"*75)
print(f"--> Toplam Gercek Hedef : {total_tested}")
print(f"--> Dogru Tahmin        : {correct_total} / {total_tested}")
print(f"--> GERCEK DUNYA ISABETI : %{acc:.1f}")
print("="*75)