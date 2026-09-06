import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> HIBRIT FIZIKSEL ASTRONET EGITIMI (V17) | BIRIM: {device.upper()}")
print("=========================================================================")

class HybridAstroNet(nn.Module):
    def __init__(self):
        super(HybridAstroNet, self).__init__()
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
        # Local Kol: 61 Nokta (TABAN COZUNURLUGUNU KORUYAN DILATED CONV)
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
        # Karar Katmanı: Global (32*50) + Local (32*30) + 3 Fiziksel Metrik = 2563 Boyut
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

def compute_physical_metrics(local_view_1d):
    """
    Seager & Mallen-Ornelas (2003) Trapezoid Taban Orani (T90 / T10) ve Dip Egriligi
    """
    min_val = torch.min(local_view_1d)
    if min_val >= -1e-5:
        return torch.tensor([0.0, 0.0, 0.0], device=local_view_1d.device)
    
    threshold_10 = min_val * 0.10
    threshold_90 = min_val * 0.90
    
    width_10 = torch.sum(local_view_1d <= threshold_10).float()
    width_90 = torch.sum(local_view_1d <= threshold_90).float()
    
    flatness_ratio = width_90 / (width_10 + 1e-5)
    
    # Dip eğriliği (Apeks açısı)
    center_idx = 30
    curvature = local_view_1d[center_idx - 3] - 2.0 * local_view_1d[center_idx] + local_view_1d[center_idx + 3]
    
    return torch.tensor([flatness_ratio.item(), curvature.item(), min_val.item()], device=local_view_1d.device)

# 3600 Örnek: TEST İLE BİREBİR AYNI FAZLAMA HATTIYLA ÜRETİLİR
N = 3600
print(f"1. {N} Adet Ozdes Fazlamali ve Fizik Metrikli Set Uretiliyor...")
np.random.seed(42)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
phys_tensors = torch.zeros(N, 3, device=device)
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
    p_sim = np.random.uniform(2.5, 6.0)
    dur_sim = np.random.uniform(0.08, 0.15)
    depth_sim = np.random.uniform(0.0006, 0.0028)
    
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
        # İKİLİ YILDIZ (V-Şekli, Keskin Apeks)
        v_prof = depth_sim * np.maximum(0.0, 1.0 - np.abs(x_val))
        noise[mask] -= torch.tensor(v_prof, device=device, dtype=torch.float32)
        labels[i] = 0.0
        
    # Test ile Birebir Aynı Faz Katlama
    sorted_idx = torch.argsort(phase)
    s_phase = phase[sorted_idx]
    s_flux = noise[sorted_idx]
    
    g_raw = interp1d_gpu(torch.linspace(-0.5 * p_sim, 0.5 * p_sim, 201, device=device), s_phase, s_flux)
    l_raw = interp1d_gpu(torch.linspace(-p_sim * 0.05, p_sim * 0.05, 61, device=device), s_phase, s_flux)
    
    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    
    g_tensors[i, 0] = g_norm
    l_tensors[i, 0] = l_norm
    phys_tensors[i] = compute_physical_metrics(l_norm)

dataset = TensorDataset(g_tensors, l_tensors, phys_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = HybridAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print("2. Hibrit Model Egitiliyor (16 Epoch)...")
t0 = time.perf_counter()
model.train()
for epoch in range(16):
    total_loss = 0.0
    for g_b, l_b, p_b, y_b in loader:
        optimizer.zero_grad()
        logits = model(g_b, l_b, p_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 4 == 0:
        print(f"    * Epoch [{epoch+1:02d}/16] -> Loss: {total_loss / len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_hybrid.pt")
print("--> Model 'astronet_hybrid.pt' olarak basariyla kaydedildi.")
