# ==============================================================================
#                  OSTE-MoE BÜYÜK GELİŞTİRİCİ MANİFESTOSU
#        (THE RESEARCH ETHOS, ARCHITECTURAL ODYSSEY & ENGINEERING CHARTER)
# ==============================================================================
Tarih: 2026-09-06 | Sürüm: OSTE-MoE SOTA Production | Mimari: Hierarchical Multi-Modal MoE
Hedef Platform: NVIDIA RTX CUDA Core & Tensor Core Accelerators

Senden önce bu kod tabanında yüzlerce saat geçiren, onlarca kez modelin sıfır çekişini izleyen,
ancak her defasında "Yapay zeka halüsinasyon görüyor" kolaycılığına kaçmak yerine
matematiğe, astrofiziğe ve donanım kayıtlarına inerek sistemi ayağa kaldıran zihniyetin belgesidir.

---

### I. TEMEL FELSEFE VE BİLİMSEL AHLAK (THE UNCOMPROMISING CORE)
1. **Asla Sayı Uydurma, Asla Kolaya Kaçma:**
   Bir yapay zekanın önüne sentetik, gürültüsüz, tek tip derinlikte veriler koyup %100 aldığında sevinme.
   O bir laboratuvar illüzyonudur. Gerçek uzay verisi lekelidir, yıldızlar süper-patlamalarla çalkalanır,
   teleskop sarsılır ve foton gürültüsü sığ geçişleri yutar. Bir model, en acımasız gerçek gürültüde
   %95 alabiliyorsa gerçektir; yoksa çöptür.

2. **Modeli Suçlamadan Önce Veri Fiziğini İncele:**
   Model bir gezegene %0 veriyorsa hemen "Ağ öğrenemiyor" deme. Biz bu süreçte şu gerçekleri yaşadık:
   - L 98-59 c modelden %0 alıyordu; çünkü literatürdeki epok yanlıştı ve pencere transitin 20 saat uzağına bakıyordu!
   - AU Mic b modelden %0 alıyordu; çünkü 30.000 ppm'lik flare patlaması medyan filtresini bozmuştu!
   - 500 gezegen Kulvar 2'de kaçmıştı; çünkü veri 4 güne kesilmişti ve transit verinin olmadığı 6. güne enjekte edilmişti!
   Problemlerin %90'ı modelin zekasında değil, verinin fiziksel koordinatlarında ve normalizasyonundadır.

---

### II. PROJENİN TARİHSEL DÖNÜŞÜMÜ (CHRONOLOGY OF AN ASCENT)
* **Faz 1 (Monolitik Çöküş):** Tek bir ağa hem transit bulma hem çift yıldız eleme hem atmosfer çözme görevi verildi. Model doygunluğa girdi, her şeye %99 basarak halüsinasyon gördü.
* **Faz 2 (MoE Modülerleşmesi):** Sistem Uzmanlar Karması (Mixture of Experts) haline getirildi. Katman 1 (Gate), Katman 2 (GPU Kutu Katlama), Katman 3 (1D-CNN Vetter), Katman 4 (SBI Normalizing Flow) olarak ayrıldı.
* **Faz 3 (Z-Score & Detrending Savaşı):** Yıldız lekesi dalgalanmalarının transit tabanında eğimli rampalar yarattığı ve sığ gezegenleri öldürdüğü keşfedildi. Wōtan biweight yüksek geçiren filtresi katlama öncesine yerleştirildi.
* **Faz 4 (Mikrosaniye Kanıtı & SOTA):** Donanımsal CUDA timer'ları ile paralel tensör işleme yapıldığında modelin aday başına 16.5 mikrosaniyede (saniyede 60.000 hedef) çalıştığı ispatlandı.

---

### III. SENİN GÖREVİN VE GELECEK VİZYON (THE ROAD AHEAD)
Senden sonraki aşama:
1. **Piksel Düzeyi Fark Görüntüleme AI'sı (Katman 1.5):** TESS'in dev piksellerindeki foton ağırlık merkezi kaymasını ölçen 2D-CNN modülünü entegre et. Bu modül sahte pozitifleri %0.1'in altına indirecektir.
2. **C++ LibTorch / TensorRT Derlemesi:** Python yorumlayıcısını devreden çıkarıp C++ seviyesine in. Tüm TESS arşivini dakikalar içinde tarayan otonom bir "Gezegen Avcısı Servisi" kur.
3. **Asla Pes Etme:** Bilim, başarısız denemelerin adli tıbbını yapıp kök nedeni bularak ilerler. Bu kodu devralırken bu araştırma disiplinini koru.
# ==============================================================================