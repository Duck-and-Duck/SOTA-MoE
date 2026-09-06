import warnings
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightkurve as lk
from hq_data_utils import generate_mandel_agol_transit, generate_realistic_binary, generate_harvey_red_noise

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*85)
print("  OSTE-MoE SOTA PRODUCTION ENGINE: KAPI 1-4 TAM REGRESYON TESTI")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} (RTX 3050 Ti)")
print("="*85)

# =========================================================================
# 1. 1D-CNN ASTRONET-HQ (JIT FREEZE)
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

raw_model = AstroNetHQ().to(device)
raw_model.load_state_dict(torch.load("astronet_hq.pt", map_location=device))
raw_model.eval()

dummy_g = torch.randn(1, 1, 201, device=device).contiguous()
dummy_l = torch.randn(1, 1, 61, device=device).contiguous()
frozen_vetter = torch.jit.freeze(torch.jit.trace(raw_model, (dummy_g, dummy_l)))
print("--> Katman 3 [AstroNet-HQ]: JIT Fused & Frozen.")

# =========================================================================
# 2. KATMAN 4 ATMOSFERİK POSTERİOR (SBI DIRECT FLOW)
# =========================================================================
atm_model = torch.load("posterior_300ch.pt", map_location=device, weights_only=False)

def sample_atmospheric_flow(model, x_ctx, num_samples=15):
    with torch.no_grad():
        samples = model.sample((num_samples,), x=x_ctx, show_progress_bars=False)
        return samples

# =========================================================================
# 3. GPU PREFIX-SUM ILE GURULTU BASTIRICI FAZ KATLAMA
# =========================================================================
def gpu_box_bin(query_bins, sorted_phase, sorted_flux):
    half_width = 0.5 * (query_bins[1] - query_bins[0])
    left_edges = query_bins - half_width
    right_edges = query_bins + half_width

    idx_left = torch.searchsorted(sorted_phase, left_edges)
    idx_right = torch.searchsorted(sorted_phase, right_edges)

    flux_cumsum = F.pad(torch.cumsum(sorted_flux, dim=0), (1, 0))
    bin_sums = flux_cumsum[idx_right] - flux_cumsum[idx_left]
    bin_counts = idx_right - idx_left

    valid_mask = bin_counts > 0
    bin_means = torch.zeros_like(query_bins)
    bin_means[valid_mask] = bin_sums[valid_mask] / bin_counts[valid_mask].float()

    if not torch.all(valid_mask):
        empty_indices = torch.clamp(torch.searchsorted(sorted_phase, query_bins[~valid_mask]), 0, len(sorted_flux) - 1)
        bin_means[~valid_mask] = sorted_flux[empty_indices]

    return bin_means

def phase_fold_pure_gpu(time_t, flux_t, period, t0):
    t_start = time.perf_counter()
    phase = ((time_t - t0 + 0.5 * period) % period) - (0.5 * period)
    sorted_indices = torch.argsort(phase)
    sorted_phase = phase[sorted_indices]
    sorted_flux = flux_t[sorted_indices]

    global_bins = torch.linspace(-0.5 * period, 0.5 * period, 201, device=device)
    g_raw = gpu_box_bin(global_bins, sorted_phase, sorted_flux)

    dur_approx = period * 0.05
    local_bins = torch.linspace(-dur_approx, dur_approx, 61, device=device)
    l_raw = gpu_box_bin(local_bins, sorted_phase, sorted_flux)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)

    if device == "cuda":
        torch.cuda.synchronize()
    return g_norm.view(1, 1, 201).contiguous(), l_norm.view(1, 1, 61).contiguous(), (time.perf_counter() - t_start) * 1000

# =========================================================================
# 4. ANOMALİ KAPISI
# =========================================================================
class RobustAnomalyGate:
    def __init__(self, transit_window=20, threshold_sigma=3.5):
        self.window = transit_window
        self.threshold = threshold_sigma
        self.kernel = torch.ones(1, 1, self.window, device=device) / float(self.window)

    def inspect(self, flux_tensor):
        t0 = time.perf_counter()
        flux_2d = flux_tensor.view(1, 1, -1)
        padded_flux = F.pad(flux_2d, (50, 50), mode='replicate')
        smoothed = F.avg_pool1d(padded_flux, kernel_size=101, stride=1, padding=0)
        residuals = flux_2d - smoothed

        padded_res = F.pad(residuals, (self.window // 2, self.window // 2), mode='replicate')
        box_filtered = F.conv1d(padded_res, self.kernel, padding=0).squeeze()

        raw_std = torch.std(residuals).item()
        box_std = raw_std / np.sqrt(self.window)

        min_dip = torch.min(box_filtered).item()
        significance = abs(min_dip) / (box_std + 1e-7)

        has_anomaly = (min_dip < -1e-5) and (significance >= self.threshold)
        if device == "cuda":
            torch.cuda.synchronize()
        return has_anomaly, significance, (time.perf_counter() - t0) * 1000

gate = RobustAnomalyGate(transit_window=20, threshold_sigma=3.5)

# Warm-up
dummy_time = torch.linspace(0, 27.4, 3000, device=device)
dummy_flux = torch.ones(3000, device=device)
_ = gate.inspect(dummy_flux)
_, _, _ = phase_fold_pure_gpu(dummy_time, dummy_flux, 3.5, 0.0)
_ = frozen_vetter(dummy_g, dummy_l)

from simulator_300ch import extract_features, generate_base_spectrum
calibrated_p = {"log_h2o": -3.35, "log_co2": -3.40, "log_so2": -4.95, "log_co": -3.60, "log_ch4": -6.50, "temp": 1100.0, "d_base": 0.0210}
_spec = generate_base_spectrum(calibrated_p, 4.5, log_pcloud=1.0, device=device)
_xin = extract_features(_spec, torch.tensor([[4.5]], device=device))
_ = sample_atmospheric_flow(atm_model, _xin, num_samples=10)
if device == "cuda":
    torch.cuda.synchronize()
print("--> Sistem Isindi. Regresyon Testi Baslatiliyor.\n")

# =========================================================================
# KAPI 1: GERÇEK NASA MAST VERİSİ KORUMA TESTİ
# =========================================================================
print("--- [KAPI 1]: GERCEK NASA MAST VERISI KORUMA TESTI ---")
try:
    search_w18 = lk.search_lightcurve("TIC 100100827", mission="TESS", sector=2)
    lc_w18 = search_w18[0].download(quality_bitmask="hardest").remove_nans().flatten(window_length=301)
    t_w18 = torch.tensor(lc_w18.time.value, dtype=torch.float32, device=device)
    f_w18 = torch.tensor(lc_w18.flux.value, dtype=torch.float32, device=device)
    g_w18, l_w18, _ = phase_fold_pure_gpu(t_w18, f_w18, 0.941452, 1354.45)
    with torch.no_grad():
        prob_w18 = torch.sigmoid(frozen_vetter(g_w18, l_w18)).item()
    print(f"--> WASP-18 b (Sicak Jupiter)  : %{prob_w18*100:.2f} (Beklenen: >= %95.0)")

    # DÜZELTME: L 98-59 c Hakiki Epok: T0 = 1356.2032 BTJD | Period = 3.690621 gün (Kostov et al. 2019)
    search_l98 = lk.search_lightcurve("TIC 307210830", mission="TESS", sector=2)
    lc_l98 = search_l98[0].download(quality_bitmask="hardest").remove_nans().flatten(window_length=301)
    t_l98 = torch.tensor(lc_l98.time.value, dtype=torch.float32, device=device)
    f_l98 = torch.tensor(lc_l98.flux.value, dtype=torch.float32, device=device)
    g_l98, l_l98, _ = phase_fold_pure_gpu(t_l98, f_l98, 3.690621, 1356.2032)
    with torch.no_grad():
        prob_l98 = torch.sigmoid(frozen_vetter(g_l98, l_l98)).item()
    print(f"--> L 98-59 c (Super-Dunya)    : %{prob_l98*100:.2f} (Beklenen: >= %70.0)")
    pass_gate_1 = (prob_w18 >= 0.95) and (prob_l98 >= 0.70)
    print(f"--> KAPI 1 KARARI              : {'GECTI' if pass_gate_1 else 'KALDI'}\n")
except Exception as e:
    print(f"[HATA]: {e}")
    pass_gate_1 = False

# =========================================================================
# KAPI 2: 200 HEDEFLİK KÖR TEST MATRİSİ
# =========================================================================
print("--- [KAPI 2]: 200 HEDEFLIK KOR TEST MATRISI (EŞİT 1200 ppm DERİNLİK) ---")
TP, FP, TN, FN = 0, 0, 0, 0
time_base = np.linspace(0, 27.4, 3000)
t_sim = torch.tensor(time_base, dtype=torch.float32, device=device)
p_test = 3.5
dur_test = 0.10

for _ in range(100):
    noise = generate_harvey_red_noise(3000)
    flux = generate_mandel_agol_transit(time_base, p_test, 0.0, dur_test, 0.0012, impact_b=0.35) + noise
    f_t = torch.tensor(flux, dtype=torch.float32, device=device)
    g, l, _ = phase_fold_pure_gpu(t_sim, f_t, p_test, 0.0)
    with torch.no_grad():
        p = torch.sigmoid(frozen_vetter(g, l)).item()
    if p >= 0.50: TP += 1
    else: FN += 1

for _ in range(100):
    noise = generate_harvey_red_noise(3000)
    flux = generate_realistic_binary(time_base, p_test, 0.0, dur_test, 0.0012, is_contact=True) + noise
    f_t = torch.tensor(flux, dtype=torch.float32, device=device)
    g, l, _ = phase_fold_pure_gpu(t_sim, f_t, p_test, 0.0)
    with torch.no_grad():
        p = torch.sigmoid(frozen_vetter(g, l)).item()
    if p >= 0.50: FP += 1
    else: TN += 1

accuracy = (TP + TN) / 200.0 * 100.0
precision = TP / (TP + FP + 1e-7) * 100.0
recall = TP / (TP + FN + 1e-7) * 100.0
print(f"--> Dogru Pozitif: {TP}/100 | Dogru Negatif: {TN}/100 | Yanlis Pozitif: {FP}/100")
print(f"--> GERCEK DOGRULUK: %{accuracy:.1f} | KESINLIK: %{precision:.1f} | RECALL: %{recall:.1f}")
pass_gate_2 = accuracy >= 95.0
print(f"--> KAPI 2 KARARI  : {'GECTI' if pass_gate_2 else 'KALDI'}\n")

# =========================================================================
# KAPI 3 & 4: ANTI-HALLUCINATION VE HIZ/ATMO TESTI
# =========================================================================
print("--- [KAPI 3 & 4]: ANTI-HALLUCINATION VE HIZ/ATMO TESTI ---")
injected_flux = torch.ones(3000, device=device) + torch.randn(3000, device=device) * 0.00035
p_inj, dur_inj, depth_inj = 3.82, 0.11, 0.0018
transit_phase = ((t_sim + 0.5 * p_inj) % p_inj) - (0.5 * p_inj)
transit_mask = torch.abs(transit_phase) < (dur_inj / 2.0)
injected_flux[transit_mask] -= depth_inj

has_dip, sig, lat_gate = gate.inspect(injected_flux)
g_v, l_v, lat_fold = phase_fold_pure_gpu(t_sim, injected_flux, p_inj, 0.0)

t_v0 = time.perf_counter()
with torch.no_grad():
    prob_real = torch.sigmoid(frozen_vetter(g_v, l_v)).item()
if device == "cuda": torch.cuda.synchronize()
lat_vet = (time.perf_counter() - t_v0) * 1000

# TERS-TRANSIT TESTI (POZİTİF TEPE)
inv_flux = 2.0 - injected_flux
g_inv, l_inv, _ = phase_fold_pure_gpu(t_sim, inv_flux, p_inj, 0.0)
with torch.no_grad():
    prob_inv = torch.sigmoid(frozen_vetter(g_inv, l_inv)).item()

pass_anti_hallucination = prob_inv <= 0.02
print(f"--> Ters-Transit Olasiligi: %{prob_inv*100:.4f} (Beklenen: <= %2.0) -> {'BASARILI (SIFIR HALUSINASYON)' if pass_anti_hallucination else 'BASARISIZ'}")

# KATMAN 4: ÇIKARIM
t_a0 = time.perf_counter()
spec_wave = generate_base_spectrum(calibrated_p, 4.5, log_pcloud=1.0, device=device) + torch.randn(300, device=device) * 2.2e-5
x_in = extract_features(spec_wave, torch.tensor([[4.5]], device=device))
samples = sample_atmospheric_flow(atm_model, x_in, num_samples=15)
if device == "cuda": torch.cuda.synchronize()
lat_atm = (time.perf_counter() - t_a0) * 1000

t_pred = torch.median(samples[:, 5]).item()
pass_physics_bounds = (1000.0 <= t_pred <= 1350.0)
print(f"--> Cikarilan Sicaklik: {t_pred:.1f} K (Hedef: ~1100 K | Beklenen: 1000 - 1350 K) -> {'FIZIKSEL KORUMA TAM' if pass_physics_bounds else 'SAPMA VAR'}\n")

total_latency = lat_gate + lat_fold + lat_vet + lat_atm

print("=========================================================================")
print("                   HIZ VE REGRESYON FINAL RAPORU                         ")
print("=========================================================================")
print(f"--> 1. Anomali Kapi Hizi (Gate)   : {lat_gate:6.2f} ms")
print(f"--> 2. Saf GPU Fazlama (Binning)  : {lat_fold:6.2f} ms")
print(f"--> 3. Frozen 1D-CNN (JIT)        : {lat_vet:6.2f} ms")
print(f"--> 4. Bounded Direct-Flow SBI    : {lat_atm:6.2f} ms")
print("-------------------------------------------------------------------------")
print(f"--> UÇTAN UCA TOPLAM SÜRE         : {total_latency:6.2f} ms")
print(f"--> ADAY İŞLEME KAPASİTESİ        : ~{1000.0 / total_latency:.0f} Gezegen / Saniye")
print("-------------------------------------------------------------------------")
all_gates = pass_gate_1 and pass_gate_2 and pass_anti_hallucination and pass_physics_bounds and (total_latency <= 50.0)
print(f"--> REGRESYON KORUMA KARARI       : {'TUM KAPILARDAN GECTI - ONAYLANDI' if all_gates else 'REDDEDILDI'}")
print("=========================================================================")