# 📊 ABC2026 BLE Indoor Localization — Báo Cáo EDA Dữ Liệu

> **Ngày**: 12 tháng 12, 2025  
> **Phiên bản**: 1.0  
> **Tác giả**: Khoa Minh (tổng hợp từ EDA + nhật ký nghiên cứu)

---

## Giới thiệu

Hi team, mình viết báo cáo EDA này để bạn đọc nhanh trước khi bàn pipeline/model.

Mình tập trung vào các số liệu chính, nguyên nhân Macro F1 thấp, và các điểm cần tránh (đặc biệt là rò rỉ dữ liệu). Nếu bạn ít thời gian, cứ đọc mục 1–5 và mục 8.

## 📋 Mục lục

1. [Tổng quan dữ liệu](#1-tổng-quan-dữ-liệu)
2. [Tiền xử lý & Windowing](#2-tiền-xử-lý--windowing)
3. [Mất cân bằng lớp (Class Imbalance)](#3-mất-cân-bằng-lớp-class-imbalance)
4. [Beacon infrastructure & độ phủ (Coverage)](#4-beacon-infrastructure--độ-phủ-coverage)
5. [Chất lượng tín hiệu RSSI](#5-chất-lượng-tín-hiệu-rssi)
6. [Temporal patterns](#6-temporal-patterns)
7. [Chất lượng đặc trưng sau windowing](#7-chất-lượng-đặc-trưng-sau-windowing)
8. [Kiểm soát rò rỉ dữ liệu (Data Leakage)](#8-kiểm-soát-rò-rỉ-dữ-liệu-data-leakage)
9. [Kết luận EDA & khuyến nghị](#9-kết-luận-eda--khuyến-nghị)
10. [Tài liệu liên quan](#10-tài-liệu-liên-quan)

## 📋 Mục tiêu & phạm vi

Mục tiêu của báo cáo này là giúp bạn nắm 3 thứ:
1) dữ liệu trông như thế nào (BLE RSSI + nhãn vị trí),
2) vì sao Macro F1 thấp (mất cân bằng lớp, tín hiệu yếu, độ phủ thưa),
3) các rủi ro/ràng buộc bắt buộc phải tuân thủ khi làm pipeline (đặc biệt là **không rò rỉ dữ liệu**).

**Nguồn số liệu chính:** các số liệu EDA trong báo cáo này ưu tiên trích từ notebook `ABC2026_Streamlined_v3.ipynb`.

**Nguồn phụ (chỉ để ghi chú/rủi ro):**
- `memory/phases/research_journal.md` (bài học + kiểm soát data leakage)
- `ABC2026 Sozolab Challenge/data_summary_raw.json` (chỉ để đối chiếu, không dùng làm chuẩn)

---

## 1) Tổng quan dữ liệu

### 1.1 Dữ liệu thô (Raw BLE)

Mình trích trực tiếp các con số tổng quan từ notebook v3:

| Hạng mục | Giá trị |
|---|---:|
| Tổng mẫu BLE thô | 1,888,019 |
| Số file BLE đọc vào | 4,107 |
| Khoảng thời gian | 2023-04-10 14:21:46+09:00 → 2023-04-13 12:50:50+09:00 |
| Số vị trí (locations) | 44 |
| Số beacon | 25 |
| Số user có nhãn | 2 (user 91, user 97) |

### 1.2 Thống kê từ `data_summary_raw.json` (lưu ý khác biệt)

File `ABC2026 Sozolab Challenge/data_summary_raw.json` ghi nhận (để team đối chiếu nhanh):

| Hạng mục | Giá trị |
|---|---:|
| Tổng mẫu (trong file thống kê này) | 1,117,361 |
| Số locations | 22 |
| Số beacon | 23 |
| Time range | 2023-04-10 14:21:46+09:00 → 2023-04-13 12:48:16+09:00 |

**Lưu ý:** Các con số (22 locations, 23 beacons, 1,117,361 samples) **không trùng** với notebook v3 (44, 25, 1,888,019). Trường hợp này thường do:
- thống kê được tạo trên **tập con** (ví dụ: chỉ user có nhãn, hoặc chỉ một số file hợp lệ), hoặc
- có bước **lọc** (lọc beacon/phiên đo, lọc theo time range, loại file lỗi/thiếu cột).

=> Nếu bạn cần ra quyết định nhanh, mình khuyên bạn bám notebook v3; `data_summary_raw.json` chỉ dùng để đối chiếu.

---

## 2) Tiền xử lý & Windowing

### 2.1 Sliding window (5 giây)

**Notebook v3 đang dùng cửa sổ 5 giây. Vì sao 5s?** (ghi chú từ nhật ký nghiên cứu)
- đủ để gom 50–100 readings/beacon
- giảm nhiễu, ổn định RSSI
- phù hợp nhịp di chuyển

**Kết quả sau windowing** (từ notebook v3):

| Hạng mục | Giá trị |
|---|---:|
| Raw samples | 1,888,019 |
| Sau windowing 5s | 11,543 |
| Tỉ lệ nén | 163.56× |
| Số đặc trưng gốc (1/beacon) | 25 |

**Gợi ý quy ước fill**:
- noise floor xấp xỉ `-110 dBm` (đã được dùng như default cho missing trong các pipeline sạch)
- binary threshold thường dùng quanh `-105 dBm` (đánh dấu “có thấy beacon hay không”)

---

## 3) Mất cân bằng lớp (Class Imbalance)

### 3.1 Chỉ số tổng quan

| Chỉ số | Giá trị | Mức độ |
|---|---:|---|
| Gini coefficient | 0.7832 | SEVERE (>0.7) |
| Imbalance ratio | 2,302× | EXTREME |
| Trung bình mẫu/location | 42,910 | mean trong notebook |
| Lớp lớn nhất | 28.52% | nurse station |
| Lớp nhỏ nhất | 0.01% | WC |

**Hệ quả trực tiếp (mình nói ngắn để bạn bắt nhanh):**
- head classes kéo gradient ⇒ Macro F1 rất khó lên
- tail classes ít mẫu ⇒ kết quả dao động, dễ overfit

### 3.2 Top locations (từ notebook v3)

| Rank | Location | Samples | % |
|---:|---|---:|---:|
| 1 | nurse station | 538,553 | 28.52% |
| 2 | Office Small | 210,400 | 11.14% |
| 3 | cafeteria | 198,877 | 10.53% |
| 4 | Office Large | 193,737 | 10.26% |
| 5 | kitchen | 167,730 | 8.88% |
| 6 | Cafeteria D | 157,583 | 8.35% |
| 7 | hallway | 116,266 | 6.16% |

### 3.3 Lorenz curve

Notebook v3 có Lorenz curve (Gini ≈ 0.783). Nhìn đường cong là thấy ngay: mẫu tập trung mạnh vào một vài location.

---

## 4) Beacon infrastructure & độ phủ (Coverage)

### 4.1 Tần suất phát hiện beacon (Top/Bottom)

Mình trích trực tiếp từ notebook v3:

| Beacon | Detections | % |
|---|---:|---:|
| B4 | 400,432 | 21.21% |
| B9 | 382,287 | 20.25% |
| B14 | 266,152 | 14.10% |
| B7 | 153,467 | 8.13% |
| B19 | 131,798 | 6.98% |
| … | … | … |
| B25 | 1,107 | 0.06% |

### 4.2 Coverage matrix

| Chỉ số | Giá trị |
|---|---:|
| Mean coverage | 4% per (location, beacon) pair |
| Zero-coverage pairs | 413/1100 (37.55%) |

**Hệ quả:** fingerprint RSSI bị thưa (sparse), nhiều cặp location-beacon không bao giờ quan sát ⇒ phân tách lớp khó, đặc biệt ở các location có tín hiệu tương tự.

---

## 5) Chất lượng tín hiệu RSSI

### 5.1 Thống kê RSSI

| Chỉ số | Giá trị |
|---|---:|
| Mean | -93.00 dBm |
| Std | 4.23 dBm |

### 5.2 Phân bố cường độ

| Nhóm | Điều kiện | % |
|---|---|---:|
| Strong | > -70 | 0.00% |
| Moderate | -70 → -85 | 4.57% |
| Weak | -85 → -95 | 55.03% |
| Very Weak | ≤ -95 | 40.41% |

**Kết luận:** dataset gần như không có tín hiệu mạnh; đa số weak/very weak ⇒ nhiều location chồng lấp trong không gian RSSI.

### 5.3 SNR theo location

Notebook v3 chưa có mục SNR theo location. Nếu bạn cần, mình sẽ thêm cell tính SNR theo một định nghĩa mà team thống nhất trước.

---

## 6) Temporal patterns

Notebook v3 hiện chưa có EDA theo thời gian (theo ngày/giờ/gaps). Nếu bạn muốn đưa phần này vào báo cáo, nói mình “muốn soi cái gì” (shift theo ngày? theo giờ? theo user?), mình sẽ thêm cell và trích số liệu từ notebook.

---

## 7) Chất lượng đặc trưng sau windowing

### 7.1 PCA

| Metric | Value |
|---|---:|
| PC1 variance | 34.23% |
| PC2 variance | 12.99% |
| PC1+PC2 | 47.22% |
| Components for 80% | 9 |
| Components for 90% | 16 |

### 7.2 Dataset cho modeling (từ notebook v3)

Mình để lại mấy con số “setup modeling” để bạn đối chiếu nhanh:

| Hạng mục | Giá trị |
|---|---:|
| X shape | (11,543, 92) |
| y shape | (11,543,) |
| Classes | 44 |
| Feature sets | mean_only (25), mean+binary (51), full (92) |
| DRSS | 16 features (top by variance) |
| CV | Stratified 5-fold (Val ~2,308–2,309/fold) |

---

## 8) Kiểm soát rò rỉ dữ liệu (Data Leakage)

### 8.1 Red flags

Từ `memory/phases/research_journal.md`, lỗi nghiêm trọng đã gặp:

- **Matrix Completion** làm feature bằng `groupby('location')` (dùng target label) ⇒ leakage gần như hoàn toàn
- dấu hiệu nhận biết: Macro F1 nhảy vọt ~0.996 trong khi baseline thật ~0.15

**Quy tắc bắt buộc:**
- mọi biến đổi phụ thuộc thống kê (mean/encoder/MC/normalizer/…) phải được **fit trong train fold** và **apply sang val fold**
- tuyệt đối tránh mọi thao tác kiểu `groupby(y)` trên toàn dataset trước khi CV

---

## 9) Kết luận EDA & khuyến nghị

### 9.1 Kết luận chính

1. **Mất cân bằng lớp cực độ** (Gini 0.7832, IR 2302×) là rào cản lớn nhất cho Macro F1.
2. **Độ phủ beacon thưa** (mean coverage 4%, 37.55% zero-pairs) làm fingerprint RSSI rất thưa.
3. **Tín hiệu yếu** (mean -93 dBm; ~95% weak/very weak) làm nhiều location dễ chồng lấp.
4. **Temporal patterns**: notebook v3 chưa có số liệu, nên mình chưa kết luận trong báo cáo này.

### 9.2 Khuyến nghị kỹ thuật (định hướng cho pipeline)

- Phần này là gợi ý từ literature/journal (không phải số liệu EDA trong notebook).
- Nếu bạn nhắm tăng Macro F1, mình ưu tiên hướng extreme imbalance: **LDAM + DRW**, cosine classifier, logit adjustment.
- Về feature, ưu tiên các đặc trưng ổn định với tín hiệu yếu: binary presence, ranking, DRSS.
- Về đánh giá, mình muốn bạn bám **Stratified K-Fold** + checklist kiểm tra leakage.

---

## 10) Tài liệu liên quan

- Notebook/pipeline: `ABC2026_Streamlined_v3.ipynb`
- Báo cáo phân tích/pipeline: `docs/ABC2026_Analysis_Report.md`
- Nhật ký nghiên cứu & bài học leakage: `memory/phases/research_journal.md`

---

*Báo cáo tổng hợp bởi Khoa Minh (có hỗ trợ GitHub Copilot) — 12 tháng 12, 2025*
