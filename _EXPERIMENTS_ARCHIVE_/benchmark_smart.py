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
print("      GERCEKCI VE TAVIZSIZ BILANCO V20 (MULTI-SCALE RESNET TESTI)        ")
print(f"--> DONANIM: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiScaleBlock, self).__init__()
        mid_ch = out_ch // 3
        rem_ch = out_ch - mid_ch * 2
        self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_ch, mid_ch, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(in_ch, rem_ch, kernel_size=15, padding=7)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        return self.act(self.bn(out) + res)

class SmartAstroNet(nn.Module):
    def __init__(self):
        super(SmartAstroNet, self).__init__()
        self.global_net = nn.Sequential(
            MultiScaleBlock(1, 24),
            nn.MaxPool1d(2),
            MultiScaleBlock(24, 48),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(12),
            nn.Flatten()
        )
        self.local_net = nn.Sequential(
            MultiScaleBlock(1, 24),
            nn.MaxPool1d(2),
            MultiScaleBlock(24, 48),
            nn.AdaptiveAvgPool1d(12),
            nn.Flatten()
        )
        self.phys_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(48 * 12 + 48 * 12 + 16, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.30),
            nn.Linear(64, 1)
        )

    def forward(self, g, l, phys):
        g_f = self.global_net(g)
        l_f = self.local_net(l)
        p_f = self.phys_net(phys)
        return self.classifier(torch.cat([g_f, l_f, p_f], dim=1))

vetter = SmartAstroNet().to(device)
vetter.load_state_dict(torch.load("astronet_smart.pt", map_location=device))
vetter.eval()

def extract_dynamic_physics(l_raw):
    l_smooth = F.avg_pool1d(l_raw.view(1, 1, -1), kernel_size=5, stride=1, padding=2).squeeze()
    min_val = torch.min(l_smooth)
    if min_val >= -1e-5:
        return torch.tensor([0.0, 0.0, 0.0], device=l_raw.device)
    t20 = torch.sum(l_smooth <= min_val * 0.20).float()
    t80 = torch.sum(l_smooth <= min_val * 0.80).float()
    flatness = t80 / (t20 + 1e-5)
    deep_pts = l_smooth[l_smooth <= min_val * 0.70]
    bottom_var = torch.std(deep_pts) if len(deep_pts) > 2 else torch.tensor(0.0, device=l_raw.device)
    return torch.tensor([flatness.item(), bottom_var.item() * 10.0, min_val.item()], device=l_raw.device)

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
    phys = extract_dynamic_physics(l_norm).view(1, 3)

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
# TEST 2: 200 HEDEFLİK KÖR TEST MATRİSİ (EŞİT 1200 ppm DERİNLİKTE)
# =========================================================================
print("\n--- 2. 200 HEDEFLIK KOR TEST MATRISI (EŞİT 1200 ppm DERİNLİKTE) ---")
print("--> 100 Gezegen (U-Tipi) ve 100 İkili Yıldız (V-Tipi) taranıyor...")

TP, FP, TN, FN = 0, 0, 0, 0
prob_planets = []
prob_binaries = []

t_sim = torch.linspace(0, 27.4, 3000, device=device)
p_test = 3.5
dur_test = 0.10

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
print("               DURUST VE GERCEKCI BILIMSEL BILANCO (V20)                 ")
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
