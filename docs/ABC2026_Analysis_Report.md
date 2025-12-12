# 📊 ABC2026 Định Vị Trong Nhà Bằng BLE - Báo Cáo Phân Tích

> **Ngày**: 12 tháng 12, 2025  
> **Phiên bản**: 3.0 (Quy trình/pipeline tinh gọn)  
> **Tác giả**: Phân tích hỗ trợ bởi AI

---

## 📋 Mục Lục

1. [Tóm Tắt Tổng Quan](#1-tóm-tắt-tổng-quan)
2. [Tổng Quan Tập Dữ Liệu](#2-tổng-quan-tập-dữ-liệu)
3. [Các Phát Hiện Quan Trọng](#3-các-phát-hiện-quan-trọng)
4. [Kết Quả Pipeline Hiện Tại](#4-kết-quả-pipeline-hiện-tại)
5. [Chẩn Đoán Vấn Đề](#5-chẩn-đoán-vấn-đề)
6. [Các Kỹ Thuật Nâng Cao Đề Xuất](#6-các-kỹ-thuật-nâng-cao-đề-xuất)
7. [Đề Xuất Pipeline V4](#7-đề-xuất-pipeline-v4)
8. [Lộ Trình Triển Khai](#8-lộ-trình-triển-khai)

---

## 1. Tóm Tắt Tổng Quan

### 🎯 Mô Tả Bài Toán
Định vị trong nhà sử dụng tín hiệu Bluetooth Low Energy (BLE) từ beacon (đèn hiệu) trong môi trường y tế/văn phòng với 44 vị trí khác nhau và 25 beacon.

### 📈 Hiệu Suất Hiện Tại
| Chỉ số | Giá trị | Trạng thái |
|--------|---------|------------|
| **F1 macro tốt nhất** | 0.1417 | ⚠️ Thấp |
| **Bộ đặc trưng tốt nhất** | `mean_only` (25 đặc trưng) | ✅ Đơn giản thắng |
| **Đánh giá** | CV phân tầng 5-fold | ✅ Đúng chuẩn |
| **Rò rỉ dữ liệu** | Không có | ✅ Sạch |

### ⚠️ Các Thách Thức Chính Được Xác Định
1. **Mất cân bằng lớp cực độ**: Gini = 0.7832, IR = 2302x
2. **Chất lượng tín hiệu kém**: RSSI trung bình = -93 dBm (toàn bộ yếu/rất yếu)
3. **Phủ sóng beacon thưa thớt**: Chỉ 4% độ phủ (coverage) trung bình
4. **Chiều cao**: 44 lớp với số lượng mẫu hạn chế mỗi lớp

---

## 2. Tổng Quan Tập Dữ Liệu

### 2.1 Thống Kê Dữ Liệu Thô

```
┌─────────────────────────────────────────────────────────────┐
│                   TÓM TẮT TẬP DỮ LIỆU                       │
├─────────────────────────────────────────────────────────────┤
│  Mẫu BLE thô             │  1,888,019                       │
│  Sau chia cửa sổ (5s)    │  11,543                          │
│  Số lượng vị trí         │  44                              │
│  Số lượng beacon         │  25                              │
│  Số lượng người dùng     │  2 (User 91, User 97)            │
│  Khoảng thời gian        │  2023-04-10 (một ngày)           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Phân Bố Vị Trí (Location)

| Thứ hạng | Vị trí | Số mẫu | Phần trăm |
|----------|--------|--------|-----------|
| 1 | nurse station | 538,553 | 28.52% |
| 2 | Office Small | 210,400 | 11.14% |
| 3 | cafeteria | 198,877 | 10.53% |
| 4 | Office Large | 193,737 | 10.26% |
| 5 | kitchen | 167,730 | 8.88% |
| 6 | Cafeteria D | 157,583 | 8.35% |
| 7 | hallway | 116,266 | 6.16% |
| ... | ... | ... | ... |
| 44 | WC | 234 | 0.01% |

**Nhận xét quan trọng**: Top 7 vị trí (location) chứa ~84% tổng dữ liệu. 37 vị trí (location) còn lại chỉ chiếm ~16%.

### 2.3 Tần Suất Phát Hiện Beacon (Beacon)

```
Top 5 beacon (đèn hiệu):
  Beacon 4:  400,432 (21.21%)
  Beacon 9:  382,287 (20.25%)
  Beacon 14: 266,152 (14.10%)
  Beacon 7:  153,467 (8.13%)
  Beacon 19: 131,798 (6.98%)

Bottom 5 beacon (đèn hiệu):
  Beacon 3:  8,902 (0.47%)
  Beacon 2:  7,993 (0.42%)
  Beacon 1:  6,192 (0.33%)
  Beacon 24: 4,275 (0.23%)
  Beacon 25: 1,107 (0.06%)
```

---

## 3. Các Phát Hiện Quan Trọng

### 3.1 🔴 Mất Cân Bằng Lớp (NGHIÊM TRỌNG)

```
Hệ số Gini: 0.7832
├── Diễn giải: Mất cân bằng NGHIÊM TRỌNG
├── Tỷ lệ mất cân bằng: 2,302x (lớp lớn nhất/nhỏ nhất)
└── Tác động: Mô hình thiên vị về các lớp đa số
```

**Phân tích đường cong Lorenz**:
- 20% vị trí (location) chứa ~80% mẫu
- Cân bằng hoàn hảo sẽ có Gini = 0
- Gini = 0.78 của chúng ta cho thấy sự bất bình đẳng cực độ

### 3.2 🔴 Chất Lượng Tín Hiệu (KÉM)

```
Phân bố RSSI:
├── RSSI trung bình: -93.0 dBm
├── Độ lệch chuẩn RSSI: 4.23 dBm
├── Mạnh (> -70 dBm): 0.00% ❌
├── Trung bình (-70 đến -85): 4.57%
├── Yếu (-85 đến -95): 55.03%
└── Rất yếu (<= -95): 40.41%
```

**Tác động**: 
- Không có tín hiệu mạnh nào trong toàn bộ tập dữ liệu
- 95%+ tín hiệu là yếu hoặc rất yếu
- Tỷ lệ nhiễu/tín hiệu cao
- Khó phân biệt vị trí (location) chỉ bằng RSSI

### 3.3 🔴 Phủ Sóng Beacon (THƯA THỚT)

```
Phân tích ma trận độ phủ (coverage):
├── Độ phủ trung bình: 4.00%
├── Cặp không có dữ liệu: 37.55% (413/1100)
└── Hầu hết beacon chỉ phủ 2-3 vị trí (location)
```

**Tác động**:
- Ma trận đặc trưng thưa (chủ yếu là giá trị thiếu)
- Nhiều cặp beacon-location không có quan sát
- Khả năng phân biệt hạn chế cho mỗi beacon

### 3.4 🟡 Phân Tích Đặc Trưng (PCA)

```
Kết quả PCA:
├── PC1 giải thích: ~15% phương sai
├── PC1+PC2: ~25% phương sai
├── Số component cho 80%: ~20 component
└── Số component cho 90%: ~30 component
```

**Nhận xét**: Dư thừa cao trong đặc trưng. 92 đặc trưng gốc có thể giảm xuống 30 mà không mất nhiều thông tin.

---

## 4. Kết Quả Pipeline Hiện Tại

### 4.1 Tóm Tắt Kỹ Thuật Tạo Đặc Trưng

| Loại đặc trưng | Số lượng | Mô tả |
|--------------|----------|-------|
| Trung bình RSSI | 25 | RSSI trung bình mỗi beacon (cửa sổ 5s) |
| Nhị phân | 25 | Cờ phát hiện (RSSI > -105) |
| Thứ hạng | 25 | Xếp hạng cường độ tín hiệu |
| DRSS | 16 | RSSI vi sai (cặp mạnh nhất) |
| Độ phủ | 1 | Số beacon được phát hiện |
| **Tổng** | **92** | Tất cả đặc trưng kết hợp |

### 4.2 So Sánh Mô Hình (XGBoost, CV 5-fold)

| Bộ đặc trưng | Số đặc trưng | F1 macro | Độ lệch chuẩn | Kết luận |
|-------------|-------------|----------|-----|----------|
| **`mean_only`** | 25 | **0.1417** | 0.0055 | ✅ Tốt nhất |
| `mean+binary` | 51 | 0.1416 | 0.0036 | Tương đương |
| `full` | 92 | 0.1326 | 0.0041 | ❌ Quá khớp |

### 4.3 Các Quan Sát Chính

1. **Đơn giản tốt hơn**: 25 đặc trưng vượt trội 92 đặc trưng
2. **Nguy cơ quá khớp**: Nhiều đặc trưng hơn → khả năng khái quát kém hơn
3. **Kết quả ổn định**: Độ lệch chuẩn thấp giữa các fold (đánh giá đúng chuẩn)
4. **Trọng số lớp hữu ích**: Đã áp dụng trọng số cân bằng (balanced)

---

## 5. Chẩn Đoán Vấn Đề

### 5.1 Tại Sao F1 Chỉ ~14%?

```
Phân Tích Nguyên Nhân Gốc:
│
├── [1] Mất Cân Bằng Cực Độ (Chính)
│   ├── 44 lớp với IR = 2302x
│   ├── Lớp thiểu số có <10 mẫu sau chia cửa sổ (windowing)
│   └── Mô hình không thể học từ quá ít ví dụ
│
├── [2] Phân Biệt Tín Hiệu Kém
│   ├── Tất cả tín hiệu đều yếu (-93 dBm trung bình)
│   ├── Phân bố RSSI chồng chéo giữa các vị trí (location)
│   └── Beacon không xác định duy nhất một vị trí (location)
│
├── [3] Độ Phủ Thưa Thớt
│   ├── 37.5% cặp beacon-location không có dữ liệu
│   ├── Nhiều vị trí (location) "trông giống nhau" trong không gian RSSI
│   └── Độ phân giải không gian hạn chế
│
└── [4] Đa Dạng Dữ Liệu Hạn Chế
    ├── Chỉ 1 ngày dữ liệu
    ├── Chỉ 2 user
    └── Không nắm bắt được pattern thời gian
```

### 5.2 Những Gì Chúng Ta Tránh (Rò Rỉ Dữ Liệu)

❌ **Matrix Completion** KHÔNG được sử dụng vì:
```python
# Điều này gây RÒ RỈ DỮ LIỆU:
location_means = df.groupby('location').mean()  # Sử dụng nhãn đích (target)!
df_filled = df.fillna(location_means)           # Rò rỉ thông tin tương lai!

# Kết quả: F1 giả = 0.996 (không phải hiệu suất thực)
```

✅ **Cách tiếp cận của chúng ta**: Sử dụng -110 dBm làm giá trị mặc định cho giá trị khuyết thiếu (missing values) (không rò rỉ)

---

## 6. Các Kỹ Thuật Nâng Cao Đề Xuất

### 6.1 Cho Mất Cân Bằng Lớp

| Kỹ thuật | Mô tả | Cải thiện kỳ vọng |
|----------|-------|-------------------|
| **SMOTE-ENN** | Lấy mẫu vượt (oversampling) tổng hợp + làm sạch nhiễu | +5-10% F1 |
| **Focal Loss** | Hàm mất mát tiêu điểm: giảm trọng số mẫu dễ, tập trung mẫu khó | +5-8% F1 |
| **Gom cụm vị trí** | Gộp vị trí (location) tương tự (44→10-15) | +15-20% F1 |
| **Phân loại phân cấp** | Tầng → Phòng → Vị trí chính xác | +10-15% F1 |

### 6.2 Cho Tín Hiệu Thưa/Yếu

| Kỹ thuật | Mô tả | Cải thiện kỳ vọng |
|----------|-------|-------------------|
| **Fingerprinting xác suất** | Sử dụng phân phối (distribution) thay vì trung bình (mean) | +5-10% F1 |
| **Mạng nơ-ron đồ thị (GNN)** | Mô hình hóa đồ thị beacon-location | +10-15% F1 |
| **Cơ chế chú ý (Attention)** | Tập trung vào beacon giàu thông tin | +5-10% F1 |
| **Quá trình Gaussian (GP)** | Nội suy không gian | +5-8% F1 |

### 6.3 Cho Học Multi-class

| Kỹ thuật | Mô tả | Cải thiện kỳ vọng |
|----------|-------|-------------------|
| **Học metric (Metric Learning)** | Mất mát triplet/contrastive | +10-15% F1 |
| **Mạng nguyên mẫu (Prototypical Networks)** | Nguyên mẫu lớp trong embedding | +10-15% F1 |
| **CatBoost** | Xử lý biến phân loại (categorical) + hỗ trợ mất cân bằng tốt | +3-5% F1 |

---

## 7. Đề Xuất Pipeline V4

### 7.1 Tổng Quan Chiến Lược

```
┌─────────────────────────────────────────────────────────────┐
│                    CHIẾN LƯỢC PIPELINE V4                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GIAI ĐOẠN 1: Gom Cụm Location                              │
│  ├── Gom 44 location → 8-10 super-location                 │
│  ├── Dựa trên độ tương tự RSSI fingerprint                  │
│  └── Kỳ vọng: F1 cải thiện từ 0.14 → 0.40+                 │
│                                                             │
│  GIAI ĐOẠN 2: Lấy Mẫu Nâng Cao                              │
│  ├── SMOTE-ENN cho cụm thiểu số                             │
│  ├── Giảm mẫu ngẫu nhiên (undersampling) cho đa số          │
│  └── Mục tiêu: Phân bố lớp cân bằng                         │
│                                                             │
│  GIAI ĐOẠN 3: Cải Thiện Mô Hình                             │
│  ├── CatBoost với Focal Loss                                │
│  ├── Tối ưu siêu tham số (Optuna)                           │
│  └── Tổ hợp mô hình (ensemble) với LightGBM                 │
│                                                             │
│  GIAI ĐOẠN 4: Tinh Chỉnh Phân Cấp                           │
│  ├── Thô: Dự đoán super-location (8-10 lớp)                │
│  ├── Tinh: Dự đoán location chính xác trong cụm            │
│  └── Kỳ vọng: F1 tổng thể 0.30-0.50                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Kế Hoạch Triển Khai Chi Tiết

#### Bước 1: Gom Cụm Location

```python
# Cách tiếp cận: Gom cụm location theo độ tương tự fingerprint RSSI ("vân tay" RSSI)
from sklearn.cluster import AgglomerativeClustering

# Tạo fingerprint (vân tay) cho location (RSSI trung bình mỗi beacon)
location_fingerprints = df_windowed.groupby('location')[beacon_cols].mean()

# Gom cụm sử dụng phân cụm phân cấp (hierarchical clustering)
clustering = AgglomerativeClustering(n_clusters=10, linkage='ward')
location_clusters = clustering.fit_predict(location_fingerprints)

# Ánh xạ: location → cluster_id
location_to_cluster = dict(zip(location_fingerprints.index, location_clusters))
```

**Kết quả kỳ vọng**:
- Giảm 44 lớp → 10 super-lớp
- Mỗi super-lớp có nhiều mẫu hơn
- Dễ học hơn, F1 cao hơn

#### Bước 2: Lấy Mẫu SMOTE-ENN

```python
from imblearn.combine import SMOTEENN

# Áp dụng chỉ cho dữ liệu huấn luyện (training data) (mỗi fold)
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
```

**Kết quả kỳ vọng**:
- Phân bố lớp cân bằng
- Loại bỏ mẫu nhiễu gần ranh giới quyết định
- Recall lớp thiểu số tốt hơn

#### Bước 3: Focal Loss với CatBoost

```python
import catboost as cb

model = cb.CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiClass',
    class_weights='Balanced',
    use_best_model=True,
    early_stopping_rounds=50,
)
```

#### Bước 4: Phân Loại Phân Cấp

```python
# Giai đoạn 1: Bộ phân loại thô (10 super-location)
coarse_model = train_model(X, y_cluster)
predicted_cluster = coarse_model.predict(X_test)

# Giai đoạn 2: Bộ phân loại tinh (mỗi cụm)
for cluster_id in range(10):
    mask = (predicted_cluster == cluster_id)
    fine_model = cluster_models[cluster_id]
    final_predictions[mask] = fine_model.predict(X_test[mask])
```

### 7.3 Hiệu Suất Kỳ Vọng

| Giai đoạn | Số lớp | F1 kỳ vọng | Ghi chú |
|-----------|--------|------------|---------|
| Hiện tại (V3) | 44 | 0.14 | Mốc chuẩn (baseline) |
| Gom cụm (10) | 10 | 0.40-0.50 | Cải thiện lớn |
| + SMOTE-ENN | 10 | 0.45-0.55 | Tốt hơn cho lớp thiểu số |
| + CatBoost | 10 | 0.50-0.60 | Mô hình tốt hơn |
| Phân cấp | 44 | 0.30-0.40 | Mức chi tiết (fine-grained) cuối cùng |

---

## 8. Lộ Trình Triển Khai

### 8.1 Hành Động Ngay (V4)

```
Gom Cụm Location
├── [ ] Triển khai phân cụm phân cấp (hierarchical clustering)
├── [ ] Trực quan hóa chất lượng cụm (silhouette, dendrogram)
├── [ ] Ánh xạ location vào cụm
└── [ ] Huấn luyện lại XGBoost trên nhãn gom cụm
Lấy Mẫu & Mô Hình
├── [ ] Triển khai quy trình (pipeline) SMOTE-ENN
├── [ ] Thêm CatBoost với trọng số lớp (class weights)
├── [ ] Tối ưu siêu tham số (hyperparameter) với Optuna
└── [ ] So sánh với mốc chuẩn (baseline)

Quy Trình (Pipeline) Phân Cấp
├── [ ] Huấn luyện bộ phân loại thô
├── [ ] Huấn luyện bộ phân loại tinh mỗi cụm
├── [ ] Xây dựng quy trình dự đoán từ đầu đến cuối (end-to-end)
└── [ ] Đánh giá cuối cùng và báo cáo
```

### 8.2 Cải Thiện Tương Lai (V5+)

| Ưu tiên | Kỹ thuật | Độ phức tạp | Cải thiện kỳ vọng |
|---------|----------|-------------|-------------------|
| Cao | Đặc trưng thời gian (trend, variance) | Thấp | +5% |
| Cao | Mô hình riêng theo người dùng (user) | Trung bình | +5-10% |
| Trung bình | Mạng nơ-ron đồ thị (GNN) | Cao | +10-15% |
| Trung bình | Học tương phản (Contrastive Learning) | Cao | +10-15% |
| Thấp | Học chuyển giao (Transfer Learning) | Cao | +5-10% |

---

## 📌 Kết Luận


1. **Chất lượng dữ liệu > Độ phức tạp mô hình**: Các thách thức cơ bản nằm ở dữ liệu (mất cân bằng, tín hiệu yếu), không phải lựa chọn mô hình.

2. **Tránh rò rỉ dữ liệu**: Hoàn thiện ma trận (Matrix Completion) trông rất tốt (F1=0.996) nhưng về cơ bản là sai. F1=0.14 trung thực của chúng ta mới là mốc chuẩn (baseline) thực.

3. **Tiết kiệm đặc trưng**: 25 đặc trưng thắng 92 đặc trưng. Quá khớp (overfitting) là rủi ro thực với dữ liệu hạn chế.

4. **Mất cân bằng lớp là chính**: Với Gini=0.78, không mô hình nào có thể học lớp thiểu số hiệu quả mà không có can thiệp.

### Các Bước Tiếp Theo Được Khuyến Nghị

1. **Gom cụm Location** (ROI cao nhất, triển khai đầu tiên)
2. **SMOTE-ENN** (kỹ thuật đã được chứng minh, dễ thêm)
3. **CatBoost** (tốt hơn XGBoost cho dữ liệu mất cân bằng)
4. **Phân loại phân cấp** (chia để trị)

### Lưu Ý Cuối

Đây là một **tập dữ liệu thực tế đầy thách thức**. F1 từ 0.30-0.40 cho phân loại 44 lớp với mức độ mất cân bằng và chất lượng tín hiệu này sẽ là một **kết quả mạnh**. Điều quan trọng là đánh giá (validation) đúng cách và tránh rò rỉ dữ liệu.

---

## Phụ Lục

### A. File Đầu Ra

| File | Mô tả |
|------|-------|
| `df_features_v3.parquet` | Đặc trưng đã xử lý (92 cột) |
| `pipeline_summary_v3.json` | Chỉ số đánh giá pipeline đầy đủ |
| `xgb_model_v3.json` | Mô hình XGBoost đã huấn luyện |

### B. Tài Liệu Tham Khảo

1. SMOTE: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
2. Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
3. BLE Fingerprinting: Faragher & Harle, "Location Fingerprinting with Bluetooth Low Energy Beacons" (2015)

### C. Mã Nguồn Repository

```
Tên repository: ABC_challenge_2026_MyLab
Chủ sở hữu: KhoaMinhPMK
Nhánh: main
Notebook: ABC2026_Streamlined_v3.ipynb
```

---


