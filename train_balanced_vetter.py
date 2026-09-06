import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> DENGELI DARBOGAZLI PINN V19 EGITIMI | BIRIM: {device.upper()}")
print("=========================================================================")

class BalancedPhysicsAstroNet(nn.Module):
    def __init__(self):
        super(BalancedPhysicsAstroNet, self).__init__()
        # Global Kol: 201 Nokta -> 1600 Boyut
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
        # Local Kol: 61 Nokta -> 960 Boyut
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
        # 1. CNN Darboğazı (Bottleneck): 2560 Boyutu 32'ye Sıkıştırır!
        self.cnn_bottleneck = nn.Sequential(
            nn.Linear(32 * 50 + 32 * 30, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )
        # 2. Fiziksel Metrik Yansıtması: 5 Boyutu 16'ya Çıkarır (Karar Gücü %33 Olur)
        self.phys_layer = nn.Sequential(
            nn.Linear(5, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)
        )
        # 3. Nihai Sınıflandırıcı: 32 + 16 = 48 Boyut
        self.classifier = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, g, l, phys):
        g_feat = self.global_conv(g)
        l_feat = self.local_conv(l)
        cnn_raw = torch.cat([g_feat, l_feat], dim=1)
        
        cnn_embed = self.cnn_bottleneck(cnn_raw)
        phys_embed = self.phys_layer(phys)
        
        combined = torch.cat([cnn_embed, phys_embed], dim=1)
        return self.classifier(combined)

def extract_5_physics_metrics(g_raw, l_raw):
    """
    Duvar bulaşmasını engelleyen taban-içi eğim ve 5 fiziksel metrik
    """
    l_smooth = F.avg_pool1d(l_raw.view(1, 1, -1), kernel_size=5, stride=1, padding=2).squeeze()
    min_val = torch.min(l_smooth)
    if min_val >= -1e-5:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=l_raw.device)
    
    # 1. Düzlük Oranı (T80 / T20)
    w20 = torch.sum(l_smooth <= min_val * 0.20).float()
    w80 = torch.sum(l_smooth <= min_val * 0.80).float()
    flatness_ratio = w80 / (w20 + 1e-5)
    
    # 2. Taban-İçi Eğim (Duvarlar hariç, sadece tabandaki 8 noktanın düzlüğü)
    bottom_points = l_smooth[26:35]
    bottom_slope = torch.mean(torch.abs(torch.diff(bottom_points))) * 50.0  # V için yüksek, U için ~0
    
    # 3. İkincil Tutulma Dip Seviyesi (Global Faz 0.5)
    g_smooth = F.avg_pool1d(g_raw.view(1, 1, -1), kernel_size=7, stride=1, padding=3).squeeze()
    edges = torch.cat([g_smooth[:20], g_smooth[-20:]])
    secondary_dip = torch.min(edges) * 10.0
    
    # 4. Giriş/Çıkış Asimetrisi
    left_half = l_smooth[10:30]
    right_half = torch.flip(l_smooth[31:51], dims=[0])
    asymmetry = torch.mean(torch.abs(left_half - right_half)) * 10.0
    
    return torch.tensor([flatness_ratio.item(), bottom_slope.item(), min_val.item(), secondary_dip.item(), asymmetry.item()], device=l_raw.device)

# 6000 Örnek Üretimi (Genişletilmiş Set)
N = 6000
print(f"1. {N} Adet Dengeli ve Taban-Eğimli Veri Havuzu Olusturuluyor...")
np.random.seed(42)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
phys_tensors = torch.zeros(N, 5, device=device)
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
        # GEZEGEN (U-Şekli, Mandel-Agol Düz Taban)
        u_prof = depth_sim * (1.0 - 0.2 * (x_val**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_val**2)))
        noise[mask] -= torch.tensor(u_prof, device=device, dtype=torch.float32)
        labels[i] = 1.0
    else:
        # İKİLİ YILDIZ (V-Şekli, Keskin Taban)
        v_prof = depth_sim * np.maximum(0.0, 1.0 - np.abs(x_val))
        noise[mask] -= torch.tensor(v_prof, device=device, dtype=torch.float32)
        if i % 3 == 0:
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
    phys_tensors[i] = extract_5_physics_metrics(g_norm, l_norm)

dataset = TensorDataset(g_tensors, l_tensors, phys_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = BalancedPhysicsAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=26)
criterion = nn.BCEWithLogitsLoss()

print("2. Model Egitiliyor (26 Epoch, Dengeli Darbogaz)...")
t0 = time.perf_counter()
model.train()
for epoch in range(26):
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
        print(f"    * Epoch [{epoch+1:02d}/26] -> Loss: {total_loss / len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_balanced.pt")
print("--> Model 'astronet_balanced.pt' olarak kaydedildi.")
