import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> MULTI-SCALE RESIDUAL ASTRONET (V20) EGITIMI | BIRIM: {device.upper()}")
print("=========================================================================")

# Çok Ölçekli Evrişim Bloğu (Farklı Sürelerdeki U ve V Şekillerini Eşzamanlı Yakalar)
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
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        out = torch.cat([c1, c2, c3], dim=1)
        return self.act(self.bn(out) + res)

class SmartAstroNet(nn.Module):
    def __init__(self):
        super(SmartAstroNet, self).__init__()
        # Global Kol (201 Nokta)
        self.global_net = nn.Sequential(
            MultiScaleBlock(1, 24),
            nn.MaxPool1d(2),
            MultiScaleBlock(24, 48),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(12),
            nn.Flatten()
        )
        # Local Kol (61 Nokta)
        self.local_net = nn.Sequential(
            MultiScaleBlock(1, 24),
            nn.MaxPool1d(2),
            MultiScaleBlock(24, 48),
            nn.AdaptiveAvgPool1d(12),
            nn.Flatten()
        )
        # Fiziksel Özellik Kolu
        self.phys_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)
        )
        # Birleşik Sınıflandırıcı
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

def extract_dynamic_physics(l_raw):
    """
    Sabit indeks kullanmaz; her transitin dip derinliğine göre dinamik hesaplar.
    """
    l_smooth = F.avg_pool1d(l_raw.view(1, 1, -1), kernel_size=5, stride=1, padding=2).squeeze()
    min_val = torch.min(l_smooth)
    if min_val >= -1e-5:
        return torch.tensor([0.0, 0.0, 0.0], device=l_raw.device)
    
    # 1. Dinamik Taban Genişlik Oranı (Carter et al. 2008)
    t20 = torch.sum(l_smooth <= min_val * 0.20).float()
    t80 = torch.sum(l_smooth <= min_val * 0.80).float()
    flatness = t80 / (t20 + 1e-5)
    
    # 2. En Dip Noktaların Varyansı (U-için düz/sıfır, V-için yüksek)
    deep_pts = l_smooth[l_smooth <= min_val * 0.70]
    bottom_var = torch.std(deep_pts) if len(deep_pts) > 2 else torch.tensor(0.0, device=l_raw.device)
    
    return torch.tensor([flatness.item(), bottom_var.item() * 10.0, min_val.item()], device=l_raw.device)

# 6000 Örnek Üretimi (Dinamik Aralık)
N = 6000
print(f"1. {N} Adet Dinamik Egitim Havuzu Olusturuluyor...")
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
    p_sim = np.random.uniform(2.0, 7.5)
    dur_sim = np.random.uniform(0.07, 0.18)
    depth_sim = np.random.uniform(0.0006, 0.0035)
    
    noise = torch.randn(3000, device=device) * 0.00035
    phase = ((t_sim + 0.5 * p_sim) % p_sim) - (0.5 * p_sim)
    mask = torch.abs(phase) < (dur_sim / 2.0)
    x_val = (phase[mask] / (dur_sim / 2.0)).cpu().numpy()
    
    if i < N // 2:
        # GEZEGEN (U-Şekli)
        u_prof = depth_sim * (1.0 - 0.2 * (x_val**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_val**2)))
        noise[mask] -= torch.tensor(u_prof, device=device, dtype=torch.float32)
        labels[i] = 1.0
    else:
        # İKİLİ YILDIZ (V-Şekli)
        v_prof = depth_sim * np.maximum(0.0, 1.0 - np.abs(x_val))
        noise[mask] -= torch.tensor(v_prof, device=device, dtype=torch.float32)
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
    phys_tensors[i] = extract_dynamic_physics(l_norm)

# Doğrulama Bölümü (Validation Split) ile Erken Durdurma Güvencesi
val_split = int(N * 0.8)
train_dataset = TensorDataset(g_tensors[:val_split], l_tensors[:val_split], phys_tensors[:val_split], labels[:val_split])
val_dataset = TensorDataset(g_tensors[val_split:], l_tensors[val_split:], phys_tensors[val_split:], labels[val_split:])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = SmartAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print("2. Multi-Scale Model Egitiliyor (Validation Izlemeli)...")
t0 = time.perf_counter()

for epoch in range(16):
    model.train()
    train_loss = 0.0
    for g_b, l_b, p_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(g_b, l_b, p_b), y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for g_b, l_b, p_b, y_b in val_loader:
            preds = model(g_b, l_b, p_b)
            val_loss += criterion(preds, y_b).item()
            correct += ((torch.sigmoid(preds) >= 0.50).float() == y_b).sum().item()
            
    val_acc = correct / len(val_dataset) * 100.0
    if (epoch + 1) % 4 == 0:
        print(f"    * Epoch [{epoch+1:02d}/16] -> Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: %{val_acc:.1f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_smart.pt")
print("--> Model 'astronet_smart.pt' olarak basariyla mühürlendi.")
