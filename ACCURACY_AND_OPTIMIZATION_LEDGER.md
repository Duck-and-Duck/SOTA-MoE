# OSTE-MoE OPTİMİZASYON VE DOĞRULUK DEFTERİ (ACCURACY LEDGER)
Tarih: 2026-09-05 | Mimari: OSTE-MoE V14 | Donanım: NVIDIA RTX 3050 Ti

Bu belge, gelecekte yapılacak hiçbir hızlandırma veya sıkıştırma (quantization) işleminin 
modelin bilimsel doğruluğunu düşürmemesi için tanımlanmış kırmızı çizgileri içerir.

---

## 1. HASSASİYET VE GÜVENLİK EŞİKLERİ
1. **AstroNet Sınıflandırma Güveni:**
   - Gezegen Kabul Eşiği: P >= 0.80
   - Eclipsing Binary / Sahte Pozitif Eleme Eşiği: P < 0.20
   - İzin verilen maksimum FPR (False Positive Rate): <= %1.5
2. **Anti-Hallucination (Ters-Transit) Kuralı:**
   - Işık eğrisi baş aşağı çevrildiğinde (F_inv = 2.0 - F), ağ kesinlikle P <= 0.02 üretmelidir.
   - Bu eşiğin üzerine çıkan modeller "halüsinasyon görüyor" sayılarak doğrudan reddedilir.
3. **Katman 4 Fiziksel Aralık Güvencesi:**
   - Sıcaklık (Temp): [650.0, 1750.0] K aralığında kesin kısıtlıdır.
   - log(H2O): [-5.5, -1.5]
   - log(CO2): [-7.5, -1.5]
   - log(SO2): [-8.0, -2.5]
   - Normalizing Flow serbest kuyruk sapmaları hiçbir koşulda bu sınırları aşamaz.

---

## 2. HIZ VE DONANIM BÜTÇESİ
- Katman 1 (Anomali Kapısı): <= 1.0 ms
- Katman 2 (Saf GPU Faz Katlama): <= 1.5 ms
- Katman 3 (1D-CNN JIT Vetting): <= 2.0 ms
- Katman 4 (Direct-Flow SBI): <= 20.0 ms
- **Uçtan Uca Toplam Aday Süresi: <= 25.0 ms**
- **Boş Yıldız Erken Çıkış Süresi: <= 1.5 ms**

---

## 3. MODEL GELİŞTİRME PROTOKOLÜ
Ağ ağırlıkları FP16 veya INT8'e sıkıştırılacaksa, doğrulanmış 500 Monte Carlo testinde 
2-Sigma başarı oranı en az %98.0 seviyesinde sabit kalmalıdır.

---

## 4. OSTE-MoE KULVAR 2 (%34 RECALL) HATASININ ADLİ ANALİZİ VE DÜZELTME KAYDI
Tarih: 2026-09-05 | Denetim Sürümü: OSTE-MoE V31.4

### A. Kök Neden (Root Cause Analysis):
1. **Zaman Aralığı - Epok Uyuşmazlığı:** `real_t_pool` dizisinden yalnızca `[:3000]` veri noktası (4.16 gün) kesilmiş, ancak simülasyon periyotları $P \in [1.2, 9.0]\text{ gün}$ olarak üretilerek $t_0 \in [0.1, P]$ atanmıştır. $P > 4.16\text{ gün}$ olan hedeflerde $t_0 > 4.16\text{ gün}$ çıktığında transit gözlem penceresinin tamamen dışına düşmüş; model transiti olmayan boş bir gürültüyü taramıştır (%34'lük sözde başarısızlığın ana sebebi veride transit olmamasıdır).
2. **Çift Filtreleme Yıkımı:** Zaten düzleştirilmiş TESS verisine ikinci kez uygulanan 0.5 günlük kayan medyan filtresi sığ transitlerin derinliğini %30 aşındırmıştır.

### B. Uygulanan Düzeltme:
1. $t_0$ zamanı gözlem aralığı içinde kalacak şekilde ($t_0 \in [t_{\text{start}}, t_{\text{start}} + 0.8 \times \Delta t_{\text{obs}}]$) sınırlandırıldı.
2. Gürültü saçılması, yıldız eğimlerinden arındırılmış yüksek frekanslı CDPP rezidüellerinden hesaplandı.
3. Katman 1.5 Difference Image Centroid analizi ön eleme kalkanı olarak devreye alındı.
