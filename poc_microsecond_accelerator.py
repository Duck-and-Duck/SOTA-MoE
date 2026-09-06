import warnings
import time
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*85)
print("  OSTE-MoE PROOF-OF-CONCEPT: MIKROSANIYE (µs) CIKARIM MOTORU")
print(f"--> DONANIM: {device.upper()} (NVIDIA RTX 3050 Ti Tensor Cores) | HEDEF: SUB-100 µs")
print("="*85)

# 1D-CNN ASTRONET-HQ MİMARİSİ
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

# Modeli Yükle ve FP16 (Yarı-Hassasiyet) Tensor Core Moduna Al
model = AstroNetHQ().to(device).half()
model.eval()

# 1. TEST: TEKİL ADAY GECİKMESİ (LATENCY TEST - JIT WARMUP)
dummy_g_single = torch.randn(1, 1, 201, device=device).half()
dummy_l_single = torch.randn(1, 1, 61, device=device).half()

# CUDA Isınma (Warmup)
for _ in range(100):
    _ = model(dummy_g_single, dummy_l_single)
torch.cuda.synchronize()

# Donanımsal CUDA Timer ile Tekil Ölçüm
start_ev = torch.cuda.Event(enable_timing=True)
end_ev = torch.cuda.Event(enable_timing=True)

start_ev.record()
for _ in range(500):
    _ = model(dummy_g_single, dummy_l_single)
end_ev.record()
torch.cuda.synchronize()

single_total_ms = start_ev.elapsed_time(end_ev)
single_us_per_candidate = (single_total_ms / 500.0) * 1000.0

print(f"--> [TEST 1: TEKİL ADAY ÇIKARIM GECİKMESİ]:")
print(f"    * Aday Başına Süre : {single_us_per_candidate:.2f} MİKROSANİYE (µs) [{single_us_per_candidate/1000.0:.4f} ms]")
print(f"    * Eşik             : < 1000 µs (Milisaniyenin Çok Altında)")

# 2. TEST: TOPLU İŞLEME KAPASİTESİ (BATCH PIPELINING THROUGHPUT)
# 256'lık bloklar halinde teleskop verisi beslendiğinde:
BATCH_SIZE = 256
dummy_g_batch = torch.randn(BATCH_SIZE, 1, 201, device=device).half()
dummy_l_batch = torch.randn(BATCH_SIZE, 1, 61, device=device).half()

start_ev.record()
for _ in range(100):
    _ = model(dummy_g_batch, dummy_l_batch)
end_ev.record()
torch.cuda.synchronize()

batch_total_ms = start_ev.elapsed_time(end_ev)
total_candidates = BATCH_SIZE * 100
batch_us_per_candidate = (batch_total_ms / total_candidates) * 1000.0
throughput_per_sec = total_candidates / (batch_total_ms / 1000.0)

print(f"\n--> [TEST 2: PARALEL BATCH VERİMİ (AMORTİZE EDİLMİŞ HIZ)]:")
print(f"    * 256'lık Paralel Blokta Aday Başına Süre : {batch_us_per_candidate:.2f} MİKROSANİYE (µs)")
print(f"    * Saniyedeki Gezegen Tarama Kapasitesi     : ~{throughput_per_sec:.0f} Aday / Saniye")
print(f"    * Tüm TESS Gökyüzü Arşivini (400.000 Yıldız) Tarama Süresi: ~{400000.0/throughput_per_sec:.1f} Saniye (~{400000.0/throughput_per_sec/60:.1f} Dakika!)")
print("="*85)