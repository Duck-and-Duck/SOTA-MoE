import warnings
import logging
import time
import torch
import torch.nn as nn
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print("      GERCEKCI VE TAVIZSIZ BILANCO V17 (HIBRIT MORFOLOJI TESTI)          ")
print(f"--> DONANIM: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

class HybridAstroNet(nn.Module):
    def __init__(self):
        super(HybridAstroNet, self).__init__()
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
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 50 + 32 * 30 + 3, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

    def forward(self, g, l, phys):
        g_feat = self.global_conv(g)
        l_feat = self.local_conv(l)
        combined = torch.cat([g_feat, l_feat, phys], dim=1)
        return self.fc(combined)

vetter = HybridAstroNet().to(device)
vetter.load_state_dict(torch.load("astronet_hybrid.pt", map_location=device))
vetter.eval()

def compute_physical_metrics(local_view_1d):
    min_val = torch.min(local_view_1d)
    if min_val >= -1e-5:
        return torch.tensor([0.0, 0.0, 0.0], device=local_view_1d.device)
    threshold_10 = min_val * 0.10
    threshold_90 = min_val * 0.90
    width_10 = torch.sum(local_view_1d <= threshold_10).float()
    width_90 = torch.sum(local_view_1d <= threshold_90).float()
    flatness_ratio = width_90 / (width_10 + 1e-5)
    curvature = local_view_1d[27] - 2.0 * local_view_1d[30] + local_view_1d[33]
    return torch.tensor([flatness_ratio.item(), curvature.item(), min_val.item()], device=local_view_1d.device)

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

def phase_fold_and_zscore(time_t, flux_t, period, t0):
    phase = ((time_t - t0 + 0.5 * period) % period) - (0.5 * period)
    sorted_indices = torch.argsort(phase)
    sorted_phase = phase[sorted_indices]
    sorted_flux = flux_t[sorted_indices]

    global_bins = torch.linspace(-0.5 * period, 0.5 * period, 201, device=device)
    g_raw = interp1d_gpu(global_bins, sorted_phase, sorted_flux)

    dur_approx = period * 0.05
    local_bins = torch.linspace(-dur_approx, dur_approx, 61, device=device)
    l_raw = interp1d_gpu(local_bins, sorted_phase, sorted_flux)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    phys = compute_physical_metrics(l_norm).view(1, 3)

    return g_norm.view(1, 1, 201).contiguous(), l_norm.view(1, 1, 61).contiguous(), phys

# =========================================================================
# TEST 1: GERÇEK NASA TESS VERİSİ (WASP-18 b)
# =========================================================================
print("\n--- 1. GERCEK NASA TESS VERISI (WASP-18 b) ---")
import lightkurve as lk
try:
    search = lk.search_lightcurve("TIC 100100827", mission="TESS", sector=2)
    lc = search[0].download(quality_bitmask="hardest").remove_nans()
    flat_lc = lc.flatten(window_length=101)
    t_val = torch.tensor(flat_lc.time.value[:3000], dtype=torch.float32, device=device)
    f_val = torch.tensor(flat_lc.flux.value[:3000], dtype=torch.float32, device=device)
    
    g_w18, l_w18, p_w18 = phase_fold_and_zscore(t_val, f_val, 0.94145, 1354.45)
    with torch.no_grad():
        prob_w18 = torch.sigmoid(vetter(g_w18, l_w18, p_w18)).item()
    
    print(f"--> WASP-18 b Olasılığı : %{prob_w18 * 100:.2f}")
    print(f"--> Karar                : {'ONAYLANDI (Gezegen)' if prob_w18 >= 0.50 else 'REDDEDILDI'}")
except Exception as e:
    print(f"[HATA]: {e}")

# =========================================================================
# TEST 2: 200 HEDEFLİK KÖR TEST MATRİSİ (BİREBİR AYNI 1200 ppm DERİNLİKTE)
# =========================================================================
print("\n--- 2. 200 HEDEFLIK KOR TEST MATRISI (EŞİT 1200 ppm DERİNLİKTE) ---")
print("--> 100 Gezegen (U-Tipi) ve 100 İkili Yıldız (V-Tipi) taranıyor...")

TP, FP, TN, FN = 0, 0, 0, 0
prob_planets = []
prob_binaries = []

t_sim = torch.linspace(0, 27.4, 3000, device=device)
p_test = 3.5
dur_test = 0.10

# 100 Gezegen Testi (1200 ppm, U-şekli)
for _ in range(100):
    noise = torch.randn(3000, device=device) * 0.00035
    phase = ((t_sim + 0.5 * p_test) % p_test) - (0.5 * p_test)
    mask = torch.abs(phase) < (dur_test / 2.0)
    
    x_t = (phase[mask] / (dur_test / 2.0)).cpu().numpy()
    u_prof = 0.0012 * (1.0 - 0.2 * (x_t**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_t**2)))
    noise[mask] -= torch.tensor(u_prof, device=device, dtype=torch.float32)
    
    g, l, phys = phase_fold_and_zscore(t_sim, noise, p_test, 0.0)
    with torch.no_grad():
        p = torch.sigmoid(vetter(g, l, phys)).item()
    prob_planets.append(p)
    if p >= 0.50:
        TP += 1
    else:
        FN += 1

# 100 İkili Yıldız Testi (BİREBİR AYNI 1200 ppm, V-şekli)
for _ in range(100):
    noise = torch.randn(3000, device=device) * 0.00035
    phase = ((t_sim + 0.5 * p_test) % p_test) - (0.5 * p_test)
    mask = torch.abs(phase) < (dur_test / 2.0)
    
    x_v = (phase[mask] / (dur_test / 2.0)).cpu().numpy()
    v_prof = 0.0012 * np.maximum(0.0, 1.0 - np.abs(x_v))
    noise[mask] -= torch.tensor(v_prof, device=device, dtype=torch.float32)
    
    g, l, phys = phase_fold_and_zscore(t_sim, noise, p_test, 0.0)
    with torch.no_grad():
        p = torch.sigmoid(vetter(g, l, phys)).item()
    prob_binaries.append(p)
    if p >= 0.50:
        FP += 1
    else:
        TN += 1

accuracy = (TP + TN) / 200.0 * 100.0
precision = TP / (TP + FP + 1e-7) * 100.0
recall = TP / (TP + FN + 1e-7) * 100.0

print("\n=========================================================================")
print("               DURUST VE GERCEKCI BILIMSEL BILANCO (V17)                 ")
print("=========================================================================")
print(f"--> Doğru Pozitif (TP) [Gezegeni Bildi]   : {TP:3d} / 100")
print(f"--> Yanlış Negatif (FN) [Gezegeni Kaçırdı] : {FN:3d} / 100")
print(f"--> Doğru Negatif (TN) [İkiliyi Eledi]     : {TN:3d} / 100")
print(f"--> Yanlış Pozitif (FP) [İkiliyi Yuttu]    : {FP:3d} / 100")
print("-------------------------------------------------------------------------")
print(f"--> GERÇEK DOĞRULUK (ACCURACY) : %{accuracy:.1f}")
print(f"--> KESİNLİK (PRECISION)       : %{precision:.1f}")
print(f"--> YAKALAMA ORANI (RECALL)    : %{recall:.1f}")
print(f"--> Ortalama Gezegen Olasılığı : %{np.mean(prob_planets)*100:.1f}")
print(f"--> Ortalama İkili Olasılığı   : %{np.mean(prob_binaries)*100:.1f}")
print("=========================================================================")
