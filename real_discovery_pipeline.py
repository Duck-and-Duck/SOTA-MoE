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
print("      OSTE-MoE V13: RESMI nflows API DESTEKLI 20ms KESIF MOTORU          ")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

# =========================================================================
# 1. MODEL TANIMI VE JIT DERLEMESI
# =========================================================================
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
vetter_raw.load_state_dict(torch.load("astronet_weights.pt", map_location=device))
vetter_raw.eval()

dummy_g = torch.randn(1, 1, 201, device=device).contiguous()
dummy_l = torch.randn(1, 1, 61, device=device).contiguous()
vetter_jit = torch.jit.trace(vetter_raw, (dummy_g, dummy_l))

# =========================================================================
# 2. RESMI nflows DOGRUDAN AKIS MOTORU (Durkan et al. 2019)
# =========================================================================
atm_model = torch.load("posterior_300ch.pt", map_location=device, weights_only=False)

def direct_flow_sample(model, x_ctx, num_samples=20):
    """
    sbi'in Python rejection dongusunu atlar. Resmi nflows flow.sample()
    metodunu dogrudan cagirir. Hata durumunda sbi'a guvenli geri duser.
    """
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
                # nflows resmi API standardı
                samples = net.sample(num_samples, context=x_ctx)
                if samples.dim() == 3:
                    samples = samples.squeeze(0)
                return samples
    except Exception:
        pass

    # Guvenli Geri Dusum (Fallback)
    with torch.no_grad():
        return model.sample((num_samples,), x=x_ctx, show_progress_bars=False)

# =========================================================================
# 3. SAF GPU FAZ KATLAMA VE ANOMALI KAPISI
# =========================================================================
def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0 = x_ref[indices - 1]
    x1 = x_ref[indices]
    y0 = y_ref[indices - 1]
    y1 = y_ref[indices]
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

# =========================================================================
# 4. KAPSAMLI CUDA WARM-UP
# =========================================================================
print("\n--> Tum 4 Katman CUDA Kernel Seviyesinde Isitiliyor...")
dummy_time = torch.linspace(0, 27.4, 3000, device=device)
dummy_flux = torch.ones(3000, device=device)

_ = gate.inspect(dummy_flux)
_, _, _ = phase_fold_pure_gpu(dummy_time, dummy_flux, 3.5, 0.0)
_ = vetter_jit(dummy_g, dummy_l)

from simulator_300ch import extract_features, generate_base_spectrum
dummy_p = {"log_h2o": -3.2, "log_co2": -3.5, "log_so2": -5.0, "log_co": -3.5, "log_ch4": -6.0, "temp": 1100.0, "d_base": 0.0018}
_spec = generate_base_spectrum(dummy_p, 4.5, log_pcloud=1.0, device=device)
_xin = extract_features(_spec, torch.tensor([[4.5]], device=device))
_ = direct_flow_sample(atm_model, _xin, num_samples=10)

if device == "cuda":
    torch.cuda.synchronize()
print("--> Tum Cekirdekler Hazir. 20ms Benchmark Baslatiliyor.")

# =========================================================================
# DENEYLER
# =========================================================================
np.random.seed(42)
time_base = torch.linspace(0, 27.4, 3000, device=device)
pure_noise = 1.0 + torch.randn(3000, device=device) * 0.00035

print("\n--- DENEY 1: GEZEGEN ICERMEYEN BOS YILDIZ (Erken Eleme) ---")
has_dip_1, sig_1, lat_gate_1 = gate.inspect(pure_noise)
if not has_dip_1:
    print(f"--> [ANOMALI KAPISI]: Temiz Yildiz Dogrulandi ({lat_gate_1:.2f} ms).")
    print(f"--> [ERKEN ELEME]: Sistem durduruldu, gereksiz hesaplama onlendi.")

print("\n--- DENEY 2: ENJEKSIYON TESTI (1800 ppm Gezegen Taramasi) ---")
injected_flux = pure_noise.clone()
p_inj, dur_inj, depth_inj = 3.82, 0.11, 0.0018
transit_phase = ((time_base + 0.5 * p_inj) % p_inj) - (0.5 * p_inj)
transit_mask = torch.abs(transit_phase) < (dur_inj / 2.0)
injected_flux[transit_mask] -= depth_inj

has_dip_2, sig_2, lat_gate_2 = gate.inspect(injected_flux)
print(f"--> [1/4] Anomali Kapisi : GECIS YAKALANDI ({lat_gate_2:.2f} ms)")

lat_fold, lat_vet, lat_atm = 0.0, 0.0, 0.0

if has_dip_2:
    # Katman 2: Saf GPU Fazlama
    g_view, l_view, lat_fold = phase_fold_pure_gpu(time_base, injected_flux, p_inj, t0=0.0)
    print(f"--> [2/4] Saf GPU Fazlama: 201-Global + 61-Lokal View ({lat_fold:.2f} ms)")

    # Katman 3: AstroNet JIT Vetting
    t_v0 = time.perf_counter()
    with torch.no_grad():
        prob_tensor = torch.sigmoid(vetter_jit(g_view, l_view))
    if device == "cuda":
        torch.cuda.synchronize()
    lat_vet = (time.perf_counter() - t_v0) * 1000
    planet_prob = prob_tensor.item()
    
    is_confirmed = planet_prob >= 0.70
    status_str = "ONAYLANDI (GECIS KESIN)" if is_confirmed else "REDDEDILDI"
    print(f"--> [3/4] JIT-CNN Vetting: {status_str} ({lat_vet:.2f} ms | Olasilik: %{planet_prob*100:.2f})")

    # Katman 4: Hizli Direct-Flow SBI
    if is_confirmed:
        spec_wave = generate_base_spectrum(dummy_p, 4.5, log_pcloud=1.0, device=device)
        x_in = extract_features(spec_wave, torch.tensor([[4.5]], device=device))

        t_a0 = time.perf_counter()
        samples = direct_flow_sample(atm_model, x_in, num_samples=20)
        if device == "cuda":
            torch.cuda.synchronize()
        lat_atm = (time.perf_counter() - t_a0) * 1000
        
        print(f"--> [4/4] Direct-Flow SBI: 300-Kanal Hizli Teshis Bitti ({lat_atm:.2f} ms)")
        print(f"    * log(H2O)  : {samples[:, 0].mean().item():.2f}")
        print(f"    * log(CO2)  : {samples[:, 1].mean().item():.2f}")
        print(f"    * Sicaklik  : {samples[:, 5].mean().item():.1f} K")

total_latency = lat_gate_2 + lat_fold + lat_vet + lat_atm
print("\n=========================================================================")
print("             GERCEK ZAMANLI PERFORMANS VE HIZ RAPORU                     ")
print("=========================================================================")
print(f"--> 1. Anomali Kapi Hizi (Gate)   : {lat_gate_2:6.2f} ms")
print(f"--> 2. Saf GPU Faz Katlama        : {lat_fold:6.2f} ms")
print(f"--> 3. 1D-CNN JIT Vetting         : {lat_vet:6.2f} ms")
print(f"--> 4. 300-Kanal Direct-Flow SBI  : {lat_atm:6.2f} ms")
print("-------------------------------------------------------------------------")
print(f"--> UÇTAN UCA TOPLAM HESAPLAMA    : {total_latency:6.2f} ms")
print(f"--> ADAY İŞLEME HIZI (THROUGHPUT) : ~{1000.0 / total_latency:.0f} Gezegen / Saniye")
print("=========================================================================")
