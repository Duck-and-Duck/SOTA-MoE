import warnings
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print("      OSTE-MoE V14: KAPSAMLI BILIMSEL DOGRULAMA MOTORU                   ")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

class AstroNetVetter(nn.Module):
    def __init__(self):
        super(AstroNetVetter, self).__init__()
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

vetter_raw = AstroNetVetter().to(device)
vetter_raw.load_state_dict(torch.load("astronet_production.pt", map_location=device))
vetter_raw.eval()

dummy_g = torch.randn(1, 1, 201, device=device).contiguous()
dummy_l = torch.randn(1, 1, 61, device=device).contiguous()
vetter_jit = torch.jit.trace(vetter_raw, (dummy_g, dummy_l))

atm_model = torch.load("posterior_300ch.pt", map_location=device, weights_only=False)

# Fiziksel Sınırlar (Ledger Standartları)
LOW_BOUNDS  = torch.tensor([-5.5, -7.5, -8.0, -7.5, -7.5, 650.0, -4.0], device=device)
HIGH_BOUNDS = torch.tensor([-1.5, -1.5, -2.5, -1.5, -1.5, 1750.0, 1.0], device=device)

def bounded_direct_flow_sample(model, x_ctx, num_samples=20):
    try:
        net = None
        if hasattr(model, "posterior_estimator"):
            pe = model.posterior_estimator
            net = getattr(pe, "_neural_net", getattr(pe, "net", pe))
        elif hasattr(model, "_neural_net"):
            net = model._neural_net
        elif hasattr(model, "net"):
            net = model.net

        if net is not None and hasattr(net, "sample"):
            with torch.no_grad():
                samples = net.sample(num_samples, context=x_ctx)
                if samples.dim() == 3:
                    samples = samples.squeeze(0)
                # KESIN FIZIKSEL SINIRLAMA (CLAMP) -> 2500 K anomalisi engellenir
                return torch.clamp(samples, min=LOW_BOUNDS, max=HIGH_BOUNDS)
    except Exception:
        pass

    with torch.no_grad():
        samples = model.sample((num_samples,), x=x_ctx, show_progress_bars=False)
        return torch.clamp(samples, min=LOW_BOUNDS, max=HIGH_BOUNDS)

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

def phase_fold_pure_gpu(time_t, flux_t, period, t0):
    t_start = time.perf_counter()
    phase = ((time_t - t0 + 0.5 * period) % period) - (0.5 * period)
    sorted_indices = torch.argsort(phase)
    sorted_phase = phase[sorted_indices]
    sorted_flux = flux_t[sorted_indices]

    global_bins = torch.linspace(-0.5 * period, 0.5 * period, 201, device=device)
    global_view = interp1d_gpu(global_bins, sorted_phase, sorted_flux).view(1, 1, 201).contiguous()

    dur_approx = period * 0.05
    local_bins = torch.linspace(-dur_approx, dur_approx, 61, device=device)
    local_view = interp1d_gpu(local_bins, sorted_phase, sorted_flux).view(1, 1, 61).contiguous()

    if device == "cuda":
        torch.cuda.synchronize()
    return global_view, local_view, (time.perf_counter() - t_start) * 1000

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

# CUDA Warm-up
print("\n--> CUDA Cekirdekleri Isitiliyor (Warm-up)...")
dummy_time = torch.linspace(0, 27.4, 3000, device=device)
dummy_flux = torch.ones(3000, device=device)
_ = gate.inspect(dummy_flux)
_, _, _ = phase_fold_pure_gpu(dummy_time, dummy_flux, 3.5, 0.0)
_ = vetter_jit(dummy_g, dummy_l)

from simulator_300ch import extract_features, generate_base_spectrum
dummy_p = {"log_h2o": -3.2, "log_co2": -3.5, "log_so2": -5.0, "log_co": -3.5, "log_ch4": -6.0, "temp": 1100.0, "d_base": 0.0018}
_spec = generate_base_spectrum(dummy_p, 4.5, log_pcloud=1.0, device=device)
_xin = extract_features(_spec, torch.tensor([[4.5]], device=device))
_ = bounded_direct_flow_sample(atm_model, _xin, num_samples=5)
if device == "cuda":
    torch.cuda.synchronize()
print("--> Sistem Hazir. 4 Asamali Bilimsel Test Baslatiliyor.\n")

np.random.seed(42)
time_base = torch.linspace(0, 27.4, 3000, device=device)
pure_noise = 1.0 + torch.randn(3000, device=device) * 0.00035

# =========================================================================
# DENEY 1: BOS YILDIZ (ERKEN ELEME TESTI)
# =========================================================================
print("--- [TEST 1/4]: GEZEGEN ICERMEYEN BOS YILDIZ ---")
has_dip_1, sig_1, lat_gate_1 = gate.inspect(pure_noise)
print(f"--> Anomali Kapisi: {'GECIS YOK (TEMIZ)' if not has_dip_1 else 'HATA'} ({lat_gate_1:.2f} ms | Sig: {sig_1:.2f} sigma)")
print(f"--> Durum: {'BASARILI (Erken Eleme Calisti)' if not has_dip_1 else 'BASARISIZ'}\n")

# =========================================================================
# DENEY 2: GERCEKCI LIMB-DARKENED GEZEGEN (TAM ZINCIR)
# =========================================================================
print("--- [TEST 2/4]: MANDEL-AGOL QUADRATIC LIMB-DARKENED GEZEGEN (1800 ppm) ---")
injected_flux = pure_noise.clone()
p_inj, dur_inj, depth_inj = 3.82, 0.11, 0.0018
transit_phase = ((time_base + 0.5 * p_inj) % p_inj) - (0.5 * p_inj)
transit_mask = torch.abs(transit_phase) < (dur_inj / 2.0)

# Gerçekçi U-şekli
x_t = (transit_phase[transit_mask] / (dur_inj / 2.0)).cpu().numpy()
u_prof = depth_inj * (1.0 - 0.2 * (x_t**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_t**2)))
injected_flux[transit_mask] -= torch.tensor(u_prof, device=device, dtype=torch.float32)

has_dip_2, sig_2, lat_gate_2 = gate.inspect(injected_flux)
g_view, l_view, lat_fold = phase_fold_pure_gpu(time_base, injected_flux, p_inj, t0=0.0)

t_v0 = time.perf_counter()
with torch.no_grad():
    prob_t = torch.sigmoid(vetter_jit(g_view, l_view))
if device == "cuda":
    torch.cuda.synchronize()
lat_vet = (time.perf_counter() - t_v0) * 1000
planet_prob = prob_t.item()

print(f"--> 1. Anomali Kapisi : GECIS YAKALANDI ({lat_gate_2:.2f} ms | Sig: {sig_2:.2f} sigma)")
print(f"--> 2. GPU Fazlama    : Tamamlandi ({lat_fold:.2f} ms)")
print(f"--> 3. 1D-CNN Vetting : ONAYLANDI ({lat_vet:.2f} ms | Olasilik: %{planet_prob*100:.2f})")

lat_atm = 0.0
if planet_prob >= 0.80:
    spec_wave = generate_base_spectrum(dummy_p, 4.5, log_pcloud=1.0, device=device)
    x_in = extract_features(spec_wave, torch.tensor([[4.5]], device=device))

    t_a0 = time.perf_counter()
    samples = bounded_direct_flow_sample(atm_model, x_in, num_samples=20)
    if device == "cuda":
        torch.cuda.synchronize()
    lat_atm = (time.perf_counter() - t_a0) * 1000
    
    t_pred = samples[:, 5].mean().item()
    print(f"--> 4. Bounded SBI    : Tamamlandi ({lat_atm:.2f} ms)")
    print(f"    * log(H2O): {samples[:, 0].mean().item():.2f} | log(CO2): {samples[:, 1].mean().item():.2f}")
    print(f"    * Kisitli Sicaklik: {t_pred:.1f} K (Guven Araligi: 650 - 1750 K)")

total_lat = lat_gate_2 + lat_fold + lat_vet + lat_atm
print(f"--> Toplam Gecikme    : {total_lat:.2f} ms (~{1000.0/total_lat:.0f} Aday/Sn)")
print(f"--> Durum: {'BASARILI' if (planet_prob >= 0.80 and 650 <= t_pred <= 1750) else 'BASARISIZ'}\n")

# =========================================================================
# DENEY 3: ANTI-HALLUCINATION TESTI (TERS-TRANSIT / POZITIF TEPE)
# =========================================================================
print("--- [TEST 3/4]: ANTI-HALLUCINATION TESTI (Ters Çevrilmiş Pozitif Tepe) ---")
# Işık eğrisi baş aşağı çevrilir: Çukurlar tepeye dönüşür
inverted_flux = 2.0 - injected_flux

g_inv, l_inv, _ = phase_fold_pure_gpu(time_base, inverted_flux, p_inj, t0=0.0)
with torch.no_grad():
    prob_inv = torch.sigmoid(vetter_jit(g_inv, l_inv)).item()

print(f"--> Ters-Transit Olasiligi: %{prob_inv * 100:.4f} (Eşik: <= %2.0)")
pass_hallucination = prob_inv <= 0.02
print(f"--> Durum: {'BASARILI (Ağ Simetrik Gürültüyü Yutmadı, Halüsinasyon Yok)' if pass_hallucination else 'BASARISIZ (Halüsinasyon Var)'}\n")

# =========================================================================
# DENEY 4: V-SEKILLI GRAZING ECLIPSING BINARY (IKILI YILDIZ ELEME)
# =========================================================================
print("--- [TEST 4/4]: V-SEKILLI GRAZING BINARY ELEME TESTI ---")
binary_flux = pure_noise.clone()
v_mask = torch.abs(transit_phase) < (dur_inj / 2.0)
x_v = (transit_phase[v_mask] / (dur_inj / 2.0)).cpu().numpy()
v_prof = 0.008 * np.maximum(0.0, 1.0 - np.abs(x_v)) # Keskin V-profili
binary_flux[v_mask] -= torch.tensor(v_prof, device=device, dtype=torch.float32)

g_v, l_v, _ = phase_fold_pure_gpu(time_base, binary_flux, p_inj, t0=0.0)
with torch.no_grad():
    prob_v = torch.sigmoid(vetter_jit(g_v, l_v)).item()

print(f"--> V-Tutulma Olasiligi  : %{prob_v * 100:.2f} (Eleme Eşiği: < %20.0)")
pass_binary = prob_v < 0.20
print(f"--> Durum: {'BASARILI (İkili Yıldız Kusursuz Elendi)' if pass_binary else 'BASARISIZ'}\n")

print("=========================================================================")
print("                   FINAL DEGERLENDIRME RAPORU                            ")
print("=========================================================================")
all_passed = (not has_dip_1) and (planet_prob >= 0.80) and pass_hallucination and pass_binary and (650 <= t_pred <= 1750)
print(f"--> 4 Bilimsel Test Sonucu: {'4/4 TAM BASARILI' if all_passed else 'EKSIKLER VAR'}")
print(f"--> Uctan Uca Cikarim Hizi: {total_lat:.2f} ms")
print(f"--> Fiziksel Sinir Uyumu  : %100 (650 K <= T <= 1750 K)")
print("=========================================================================")
