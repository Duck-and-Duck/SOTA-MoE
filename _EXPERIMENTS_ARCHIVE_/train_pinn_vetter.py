import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> FIZIK BILGILENDIRMELI (PINN) ASTRONET V18 | BIRIM: {device.upper()}")
print("=========================================================================")

# En Küçük Kareler Parabolik Ağırlıkları (Merkez 15 nokta için katsayılar)
# x = [-7..+7] -> sum(x^2)=280, var=18.667 -> Analytical Savitzky-Golay kernel
x_idx = np.arange(-7, 8)
w_curv = (x_idx**2 - 18.6667) / 3458.67
W_CURV_TENSOR = torch.tensor(w_curv, dtype=torch.float32, device=device).view(1, 15)

class PhysicsInformedAstroNet(nn.Module):
    def __init__(self):
        super(PhysicsInformedAstroNet, self).__init__()
        # Global Kol: 201 Nokta
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
        # Local Kol: 61 Nokta (Taban detayını koruyan mimari)
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
        # Karar Katmanı: Global + Local + 4 Fiziksel Metrik
        self.fc = nn.Sequential(
            nn.Linear(32 * 50 + 32 * 30 + 4, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.20),
            nn.Linear(64, 1)
        )

    def forward(self, g, l, phys):
        g_feat = self.global_conv(g)
        l_feat = self.local_conv(l)
        combined = torch.cat([g_feat, l_feat, phys], dim=1)
        return self.fc(combined)

def extract_advanced_physics_metrics(g_raw, l_raw):
    """
    Gürültüden arındırılmış 4 fiziksel morfoloji metriği
    """
    # 1. 5-Noktalı Kayan Ortalama ile Yumuşatma (Gürültü iğneleri elenir)
    l_smooth = F.avg_pool1d(l_raw.view(1, 1, -1), kernel_size=5, stride=1, padding=2).squeeze()
    
    min_val = torch.min(l_smooth)
    if min_val >= -1e-5:
        return torch.tensor([0.0, 0.0, 0.0, 0.0], device=l_raw.device)
    
    # 2. Dayanıklı Taban Genişlik Oranı (T85 / T15)
    t15_thresh = min_val * 0.15
    t85_thresh = min_val * 0.85
    width_15 = torch.sum(l_smooth <= t15_thresh).float()
    width_85 = torch.sum(l_smooth <= t85_thresh).float()
    flatness_ratio = width_85 / (width_15 + 1e-5)
    
    # 3. Analitik En Küçük Kareler Parabolik Taban Eğriliği (a katsayısı)
    center_15 = l_smooth[23:38].view(1, 15)
    curvature_a = torch.sum(center_15 * W_CURV_TENSOR) * 100.0  # V-için yüksek, U-için ~0
    
    # 4. İkincil Tutulma / Faz 0.5 Dip Tespiti (Global pencerenin uçları)
    g_smooth = F.avg_pool1d(g_raw.view(1, 1, -1), kernel_size=7, stride=1, padding=3).squeeze()
    edges = torch.cat([g_smooth[:20], g_smooth[-20:]])
    secondary_dip = torch.min(edges) * 10.0
    
    return torch.tensor([flatness_ratio.item(), curvature_a.item(), min_val.item(), secondary_dip.item()], device=l_raw.device)

# 5000 Örnek Üretimi (Genişletilmiş Fizik Havuzu)
N = 5000
print(f"1. {N} Adet Gelismis Morfolojik Egitim Havuzu Olusturuluyor...")
np.random.seed(42)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
phys_tensors = torch.zeros(N, 4, device=device)
labels = torch.zeros(N, 1, device=device)

t_sim = torch.linspace(0, 27.4, 3000, device=device)

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

for i in range(N):
    p_sim = np.random.uniform(2.2, 7.0)
    dur_sim = np.random.uniform(0.08, 0.16)
    depth_sim = np.random.uniform(0.0006, 0.0030)
    
    noise = torch.randn(3000, device=device) * 0.00035
    phase = ((t_sim + 0.5 * p_sim) % p_sim) - (0.5 * p_sim)
    mask = torch.abs(phase) < (dur_sim / 2.0)
    x_val = (phase[mask] / (dur_sim / 2.0)).cpu().numpy()
    
    if i < N // 2:
        # GEZEGEN: U-Şekli (Mandel-Agol Düz Taban)
        u_prof = depth_sim * (1.0 - 0.2 * (x_val**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_val**2)))
        noise[mask] -= torch.tensor(u_prof, device=device, dtype=torch.float32)
        labels[i] = 1.0
    else:
        # İKİLİ YILDIZ: V-Şekli (Sivri Apeks) + Bazen İkincil Tutulma
        v_prof = depth_sim * np.maximum(0.0, 1.0 - np.abs(x_val))
        noise[mask] -= torch.tensor(v_prof, device=device, dtype=torch.float32)
        if i % 3 == 0:
            # İkincil tutulma ekle (Faz 0.5 civarı)
            sec_phase = ((t_sim + p_sim) % p_sim) - (0.5 * p_sim)
            sec_mask = torch.abs(sec_phase) < (dur_sim / 2.5)
            noise[sec_mask] -= (depth_sim * 0.45)
        labels[i] = 0.0
        
    sorted_idx = torch.argsort(phase)
    s_phase = phase[sorted_idx]
    s_flux = noise[sorted_idx]
    
    g_raw = interp1d_gpu(torch.linspace(-0.5 * p_sim, 0.5 * p_sim, 201, device=device), s_phase, s_flux)
    l_raw = interp1d_gpu(torch.linspace(-p_sim * 0.05, p_sim * 0.05, 61, device=device), s_phase, s_flux)
    
    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    
    g_tensors[i, 0] = g_norm
    l_tensors[i, 0] = l_norm
    phys_tensors[i] = extract_advanced_physics_metrics(g_norm, l_norm)

dataset = TensorDataset(g_tensors, l_tensors, phys_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = PhysicsInformedAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)
criterion = nn.BCEWithLogitsLoss()

print("2. PINN Modeli Egitiliyor (24 Epoch, Cosine Annealing)...")
t0 = time.perf_counter()
model.train()
for epoch in range(24):
    total_loss = 0.0
    for g_b, l_b, p_b, y_b in loader:
        optimizer.zero_grad()
        logits = model(g_b, l_b, p_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 6 == 0:
        print(f"    * Epoch [{epoch+1:02d}/24] -> Loss: {total_loss / len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_pinn.pt")
print("--> Model 'astronet_pinn.pt' olarak kaydedildi.")
