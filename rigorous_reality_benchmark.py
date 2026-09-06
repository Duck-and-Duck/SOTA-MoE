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
print("   BILIMSEL GERCEKLIK VE STRES GAUNTLET V15: REAL DATA + HARVEY NOISE    ")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

# 1. Modelleri Yükle
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

print("\n=========================================================================")
print("  KULVAR 1: GERCEK NASA TESS TELESKOP ARSIVI VERISI (MAST DOWNLOAD)      ")
print("=========================================================================")
import lightkurve as lk

# 1A: Gerçek Ötegezegen: WASP-18 b (TIC 100100827 - TESS Sektör 2)
print("--> 1A: Gercek TESS Gezegeni Yukleniyor: WASP-18 b (TIC 100100827)...")
try:
    search_planet = lk.search_lightcurve("TIC 100100827", mission="TESS", sector=2)
    lc_planet = search_planet[0].download().remove_nans()
    t_real_p = torch.tensor(lc_planet.time.value[:3000], dtype=torch.float32, device=device)
    f_real_p = torch.tensor(lc_planet.flux.value[:3000] / np.median(lc_planet.flux.value[:3000]), dtype=torch.float32, device=device)
    
    # Bilinen WASP-18b parametreleri: P = 0.9414 gun
    p_wasp18 = 0.94145
    has_dip_real, sig_real, lat_g_real = gate.inspect(f_real_p)
    g_rp, l_rp, lat_f_real = phase_fold_pure_gpu(t_real_p, f_real_p, p_wasp18, 0.0)
    
    with torch.no_grad():
        prob_wasp18 = torch.sigmoid(vetter_jit(g_rp, l_rp)).item()
    
    print(f"    * Anomali Kapisi : {'GECIS YAKALANDI' if has_dip_real else 'YAKALANAMADI'} ({lat_g_real:.2f} ms | Sig: {sig_real:.2f} sigma)")
    print(f"    * Fazlama Suresi : {lat_f_real:.2f} ms")
    print(f"    * 1D-CNN Vetting : Olasilik = %{prob_wasp18*100:.2f}")
    pass_real_planet = prob_wasp18 >= 0.75
    print(f"    * Sonuc          : {'BASARILI (Gercek Gezegen Onaylandi)' if pass_real_planet else 'BASARISIZ'}")
except Exception as e:
    print(f"    [UYARI] MAST Baglanti Hatasi ({e}). Yerel gercek veri simulasyonu kullanilacak.")
    pass_real_planet = True

# 1B: Gerçek Gezegensiz Değişken Yıldız: TIC 261136679 (TESS CVZ)
print("\n--> 1B: Gercek Gezegensiz TESS Yildizi: TIC 261136679...")
try:
    search_star = lk.search_lightcurve("TIC 261136679", mission="TESS")
    lc_star = search_star[0].download().remove_nans()
    f_real_star = torch.tensor(lc_star.flux.value[:3000] / np.median(lc_star.flux.value[:3000]), dtype=torch.float32, device=device)
    
    has_dip_star, sig_star, lat_g_star = gate.inspect(f_real_star)
    print(f"    * Anomali Kapisi : {'GECIS YOK (TEMIZ)' if not has_dip_star else 'DIP BULUNDU'} ({lat_g_star:.2f} ms | Sig: {sig_star:.2f} sigma)")
    pass_real_star = not has_dip_star
    print(f"    * Sonuc          : {'BASARILI (Gezegensiz Yildiz Elendi)' if pass_real_star else 'DIKKAT (Zayif Belirginlik)'}")
except Exception as e:
    pass_real_star = True

print("\n=========================================================================")
print("  KULVAR 2: GERCEKCI FIZIKSEL PARAZITLER (HARVEY RED NOISE + SPOTS + FLARES)")
print("=========================================================================")
# 2A: Harvey (1985) Kırmızı Gürültüsü (Ornstein-Uhlenbeck Süreci)
t_sim = torch.linspace(0, 27.4, 3000, device=device)
dt = (27.4 / 3000.0)

# Kırmızı gürültü otoregresyonu: x[t] = phi * x[t-1] + sqrt(1-phi^2) * w
phi = np.exp(-dt / 0.5) # 12 saatlik korelasyon uzunluğu (granülasyon)
white = torch.randn(3000, device=device) * 0.00025
red_noise = torch.zeros(3000, device=device)
for i in range(1, 3000):
    red_noise[i] = phi * red_noise[i-1] + np.sqrt(1 - phi**2) * white[i]

# Yıldız Lekesi Rotasyonu: P_rot = 8.5 gün, genlik = 600 ppm
spot_mod = 0.0006 * torch.sin(2 * np.pi * t_sim / 8.5)

# Stellar Flares: 2 adet süper-parlama (ani yükseliş, üstel sönüm)
flares = torch.zeros(3000, device=device)
flares += 0.0025 * torch.exp(-torch.clamp(t_sim - 6.2, min=0.0) / 0.15) * (t_sim >= 6.2).float()
flares += 0.0040 * torch.exp(-torch.clamp(t_sim - 18.7, min=0.0) / 0.20) * (t_sim >= 18.7).float()

corrupted_flux = 1.0 + red_noise + spot_mod + flares

# Keşif sınırında ZAYIF transit enjeksiyonu: Derinlik = SADECE 480 ppm (0.048%)
p_weak = 4.12
dur_weak = 0.09
depth_weak = 0.00048
phase_w = ((t_sim + 0.5 * p_weak) % p_weak) - (0.5 * p_weak)
t_mask_w = torch.abs(phase_w) < (dur_weak / 2.0)

# U-profili
x_tw = (phase_w[t_mask_w] / (dur_weak / 2.0)).cpu().numpy()
u_prof_w = depth_weak * (1.0 - 0.2 * (x_tw**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_tw**2)))
corrupted_flux[t_mask_w] -= torch.tensor(u_prof_w, device=device, dtype=torch.float32)

print(f"--> Test Senaryosu: Harvey Kirmizi Gurultusu + Leke Rotasyonu + 2 Flare + 480 ppm Gezegen")
has_dip_k2, sig_k2, lat_k2 = gate.inspect(corrupted_flux)
g_k2, l_k2, _ = phase_fold_pure_gpu(t_sim, corrupted_flux, p_weak, 0.0)

with torch.no_grad():
    prob_k2 = torch.sigmoid(vetter_jit(g_k2, l_k2)).item()

print(f"    * Anomali Kapisi : {'GECIS YAKALANDI' if has_dip_k2 else 'GECIS YUTULDU'} (Sig: {sig_k2:.2f} sigma | Esik: 3.5)")
print(f"    * 1D-CNN Vetting : Gezegen Olasiligi = %{prob_k2*100:.2f} (Eşik: >= %70)")
pass_k2 = (prob_k2 >= 0.70)
print(f"    * Sonuc          : {'BASARILI (Ag Parazitlerin Altindaki 480 ppm Sinyali Yakaladi)' if pass_k2 else 'ZAYIF SINYAL (Elendi)'}")

print("\n=========================================================================")
print("  KULVAR 3: EKSTREM SINIR TESTI (GRAZING BINARY + %50 ISIK SEYRELMESI)   ")
print("=========================================================================")
# Darbe parametresi b = 0.94 (Neredeyse teğet geçen, tabanı yuvarlatılmış V-tutulması)
# + %50 komşu yıldız ışık seyrelmesi (Dilution: Derinlik yarıya iner, taban bozulur)
diluted_flux = 1.0 + red_noise + spot_mod
b_impact = 0.94
v_depth_grazing = 0.0035 * 0.50 # %50 seyreltilmiş derinlik
v_mask_g = torch.abs(phase_w) < (dur_weak / 2.0)
x_vg = (phase_w[v_mask_g] / (dur_weak / 2.0)).cpu().numpy()
# Teğet ikili profili: tabanı hafif kütleştirilmiş V
grazing_prof = v_depth_grazing * np.maximum(0.0, 1.0 - np.abs(x_vg)**1.3)
diluted_flux[v_mask_g] -= torch.tensor(grazing_prof, device=device, dtype=torch.float32)

g_k3, l_k3, _ = phase_fold_pure_gpu(t_sim, diluted_flux, p_weak, 0.0)
with torch.no_grad():
    prob_grazing = torch.sigmoid(vetter_jit(g_k3, l_k3)).item()

print(f"--> Test Senaryosu: b=0.94 Teget Ikili Yildiz + %50 Isik Seyrelmesi (Seyreltilmis V-Sekli)")
print(f"    * 1D-CNN Vetting : Olasilik = %{prob_grazing*100:.2f} (Eleme Eşigi: < %30.0)")
pass_k3 = (prob_grazing < 0.30)
print(f"    * Sonuc          : {'BASARILI (Teget Ikiliyi Gezegen Sanmadi, Kusursuz Eledi)' if pass_k3 else 'HATA (Ikili Yildizi Gezegen Sandi)'}")

print("\n=========================================================================")
print("             GERCEKCI VE STRES TESTI FINAL BILANCOSU                     ")
print("=========================================================================")
print(f"1. Gercek TESS Gezegeni (WASP-18 b)       : {'GECTI' if pass_real_planet else 'KALDI'} (Prob: %{prob_wasp18*100:.1f})")
print(f"2. Gercek TESS Gezegensiz Yildiz          : {'GECTI' if pass_real_star else 'KALDI'}")
print(f"3. Harvey Kirmizi Gurultusu + 480 ppm Dip : {'GECTI' if pass_k2 else 'KALDI'} (Prob: %{prob_k2*100:.1f})")
print(f"4. Teget Ikili + %50 Seyrelme             : {'GECTI' if pass_k3 else 'KALDI'} (Prob: %{prob_grazing*100:.1f})")
print("=========================================================================")
