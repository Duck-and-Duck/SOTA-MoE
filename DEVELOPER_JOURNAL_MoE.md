# OSTE-MoE PROJESİ GELİŞTİRİCİ GÜNLÜĞÜ (DEVELOPER ARCHITECTURE JOURNAL)
Son Güncelleme: 2026-09-06 | Mimari: Hierarchical Multi-Modal OSTE-MoE | Hedef Platform: NVIDIA RTX CUDA

Bu belge, bu projenin hangi bilimsel aşamalardan geçtiğini, hangi hataların neden yapıldığını ve
nasıl çözüldüğünü gelecekteki geliştiricilerin kodu tek bakışta anlaması için belgeler.

---

## 1. MİMARİ VE KATMANLARIN GÖREV DAĞILIMI
Proje, tek bir monolitik yapay zeka yerine "Uzmanlar Karması" (Mixture of Experts - MoE) mimarisiyle çalışır:
* **Katman 1 (Robust Anomaly Gate):** 101 adımlık causal kayan ortalama ve 20 noktalı kutu filtresi ile sinyalsiz yıldızları < 1 ms içinde eler (Erken Çıkış).
* **Katman 1.5 (Astrometric Centroid Expert):** TESS'in 21 ark-saniyelik dev piksellerindeki foton ağırlık merkezi kaymasını (PRF fit) ölçerek komşu arka plan ikililerini (BEB) eler.
* **Katman 2 (GPU Prefix-Sum Binning & Detrending):** 18.000 veri noktasını `torch.cumsum` ile GPU üzerinde 201 global ve 61 lokal kutuya ortalayarak foton gürültüsünü 4x bastırır.
* **Katman 3 (AstroNet-HQ 1D-CNN):** NASA SPOC ve Google AstroNet standardında kuadratik kenar kararmalı U-transit morfolojisini tanır.
* **Katman 4 (300-Kanal SBI Normalizing Flow):** Doğrulanan gezegenin transit derinliğinden saniyeler içinde atmosferik denge sıcaklığı (T_eq), H2O, CO2 ve bulut basıncını çözer.

---

## 2. YAPILAN KRİTİK HATALAR VE BİLİMSEL DÜZELTMELER
1. **Lokal Pencereleme Hatası:** İlk versiyonlarda pencere `P * 0.05` olarak açıldı. Kısa periyotlu WASP-18 b'de transitin yarısı kesildi ve model V-şekli sanıp reddetti. Çözüm: Transit süresinin tam 2 katı (`dur * 2.0`) kuralına geçildi.
2. **Kestirme Yol (Shortcut Learning) Tuzağı:** Eğitimde fiziksel tensöre rastgele sayılar verildiğinde ağ evrişim katmanlarını öğrenmeyi bırakıp skaler sayılara aşırı uyum sağladı. Çözüm: Tüm özelliklerin doğrudan ışık eğrisinden ölçüldüğü "Zero-Shortcut" eğitim hattı kuruldu.
3. **Leke ve Detrending Körlüğü:** Yıldız lekesi dalgalanmaları ışık eğrisi düzleştirilmeden katlandığında transitler eğimli bir rampaya oturdu ve model gezegenleri kaçırdı (%50 kayıp). Çözüm: Wōtan biweight yüksek geçiren filtresi katlama öncesine yerleştirildi.
4. **Epok Uyuşmazlığı:** L 98-59 c ve TOI-270 b test edilirken literatürdeki yanlış epoklar verilmişti. NASA katalog epokları (L 98-59 c: 1356.2032 BTJD, TOI-270 b: 1387.0922 BTJD) düzeltilince modeller doğrudan %100 isabete ulaştı.

---

## 3. MİKROSANİYE (µs) HIZLANDIRMA YOL HARİTASI (GELECEK FAZ)
Şu an sistem aday başına 30-35 ms harcamaktadır. Bu süre tüm TESS sektörünü 4 saatte tarayabilir. Mikrosaniyeye (< 100 µs) inmek için:
1. Python aradan çıkarılıp C++ LibTorch / CUDA C++ kernel'ine geçilmelidir.
2. FP32 ağırlıklar NVIDIA TensorRT ile INT8 tensör çekirdeklerine derlenmelidir (40-80 µs çıkarım).
3. Veri PCIe üzerinden kopyalanmadan doğrudan GPU Unified Memory DMA ile akıtılmalıdır.