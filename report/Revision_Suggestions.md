# 📝 ĐỀ XUẤT CHỈNH SỬA — Silent Exit Project

> **Ngày:** 09/05/2026  
> **Người review:** Nguyễn Tuấn Phong  
> **Files được review:**  
> - `SilentExit_Phase0_Academic_Foundation (1).md`  
> - `SilentExit_Phase2_IEEE_Paper (1).tex`  
> - `Phase2_Churn_Notebook.ipynb`

---

## TỔNG QUAN

Cả 3 file mới đều **cải thiện đáng kể** so với bản cũ. AUC realistic (0.748), H4 nhất quán giữa notebook và paper, Phase 0 mở rộng tốt. Tuy nhiên còn một số vấn đề cần sửa **trước khi nộp**.

---

## 🔴 PHẢI SỬA TRƯỚC KHI NỘP

### 1. IEEE Paper — Thiếu Figures hoàn toàn

**File:** `SilentExit_Phase2_IEEE_Paper (1).tex`

**Vấn đề:** Paper 443 dòng LaTeX mà không có bất kỳ figure/image nào. Notebook có rất nhiều hình chất lượng cao nhưng không figure nào được include vào paper. Đối với IEEE conference paper, đây là thiếu sót nghiêm trọng.

**Yêu cầu:** Thêm **ít nhất 4 figures** vào paper:

| Figure | Vị trí đề xuất | Lý do |
|--------|----------------|-------|
| **SHAP Feature Importance Bar Plot** | Section IV.D (Feature Importance) — sau Table IV | Minh họa trực quan H2 — behavioral features dominate |
| **ROC Curves (LR vs RF)** | Section IV.C (Model Performance) — sau Table III | So sánh trực quan 2 models |
| **Churn Rate by Membership Tier** | Section IV.B (Churn Concentration) — sau Table II | Minh họa non-monotonic pattern (H1) |
| **Priority Save Zone / Retention Matrix** | Section IV.E (Risk Segmentation) | Minh họa PSZ — core managerial output |

**Cách làm:** Export hình từ notebook dưới dạng PNG/PDF, thêm vào paper bằng:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{figure_name.png}
\caption{Caption mô tả.}
\label{fig:label_name}
\end{figure}
```

---

### 2. IEEE Paper — Điền Author Name

**File:** `SilentExit_Phase2_IEEE_Paper (1).tex`  
**Dòng:** 23

**Hiện tại:**
```latex
\IEEEauthorblockN{Author Name}
```

**Sửa thành:** Tên thật của tác giả/nhóm.

---

### 3. IEEE Paper — Thiếu Acknowledgements Section

**File:** `SilentExit_Phase2_IEEE_Paper (1).tex`  
**Vị trí:** Trước `\begin{thebibliography}`

**Thêm:**
```latex
\section*{Acknowledgements}
The authors would like to thank [tên giảng viên hướng dẫn] for guidance throughout this project.
```

---

## 🟡 NÊN SỬA (TĂNG CHẤT LƯỢNG ĐÁNG KỂ)

### 4. IEEE Paper — Bổ sung References

**File:** `SilentExit_Phase2_IEEE_Paper (1).tex`

**Hiện tại:** 14 references — hơi ít cho IEEE conference paper.

**Đề xuất thêm 4–6 references**, ưu tiên:
- 1–2 papers về churn prediction **trong e-commerce cụ thể** (không chỉ telecom)
- 1 paper về CLV validation/comparison methods
- 1 paper về SHAP interpretation best practices
- Integrate `Rößler & Schoder (2022)` sâu hơn vào Discussion (hiện chỉ cite ở Section I)

---

### 5. Notebook — Kiểm tra Conclusions Section

**File:** `Phase2_Churn_Notebook.ipynb`

**Cần verify** phần Conclusions (Module cuối) đảm bảo:

| Nội dung | Phải khớp với |
|----------|---------------|
| H1 verdict | "Partially supported" — Gold 9.9%, Silver 7.1%, non-monotonic |
| H2 verdict | "Supported" — behavioral features dominate SHAP top-10 |
| H3 verdict | "Supported in per-capita terms" — PSZ CLV $772 |
| H4 verdict | **"Rejected for return rate, not supported for discount"** — r = −0.571, p = 0.033 |
| SHAP demographic weight | **~6%** (KHÔNG PHẢI "less than 30%") |
| Revenue at risk | $0.87M (7% of total) |
| PSZ size | 56 customers, mean CLV $772 |

> ⚠️ **Lưu ý:** Bản cũ từng mắc lỗi H4 "supported" trong conclusions trong khi data nói "not supported". Đây là lỗi nghiêm trọng — cần kiểm tra kỹ bản mới không lặp lại.

---

### 6. Notebook — Thêm Cross-Validation

**File:** `Phase2_Churn_Notebook.ipynb`

**Hiện tại:** Chỉ 1 train/test split (80/20 stratified).

**Đề xuất:** Thêm 1 cell chạy **5-fold Stratified CV**, report mean ± std cho cả ROC-AUC và PR-AUC:
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf_cv_roc = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')
rf_cv_pr  = cross_val_score(rf_model, X, y, cv=cv, scoring='average_precision')
print(f"RF ROC-AUC: {rf_cv_roc.mean():.4f} ± {rf_cv_roc.std():.4f}")
print(f"RF PR-AUC:  {rf_cv_pr.mean():.4f} ± {rf_cv_pr.std():.4f}")
```

---

### 7. Notebook — Thêm Feature Correlation Check

**File:** `Phase2_Churn_Notebook.ipynb`

**Đề xuất:** Thêm 1 cell trước modeling section:
- Correlation heatmap giữa các features
- VIF (Variance Inflation Factor) check cho multicollinearity
- Đặc biệt kiểm tra: `total_orders` ↔ `total_spend_usd` ↔ `CLV_3yr` ↔ `orders_per_yr`

---

## 🟢 NẾU CÒN THỜI GIAN

### 8. CLV Estimation nâng cao

**File:** `Phase2_Churn_Notebook.ipynb`

CLV hiện tại = `avg_aov × orders_per_yr × 3` — quá đơn giản. Phase 0 đã cite Fader et al. (2005) BG/NBD + Gamma-Gamma framework nhưng chưa implement.

**Nếu có thời gian:**
```python
# pip install lifetimes
from lifetimes import BetaGeoFitter, GammaGammaFitter
```
So sánh CLV estimates giữa simple formula vs BG/NBD.

### 9. Churn Threshold Sensitivity Analysis

**File:** `Phase2_Churn_Notebook.ipynb`

Thêm 1 cell test sensitivity của kết quả khi thay đổi churn definition threshold.

### 10. Temporal Train/Test Split

**File:** `Phase2_Churn_Notebook.ipynb`

Hiện tại dùng random stratified split. Trong production, nên train trên orders trước cutoff date T, test trên hành vi sau T.

---

## CHECKLIST TRƯỚC KHI NỘP

- [ ] Thêm 4 figures vào IEEE paper
- [ ] Điền author name trong paper
- [ ] Thêm Acknowledgements section
- [ ] Verify notebook conclusions khớp paper numbers
- [ ] Verify H4 = "rejected/not supported" trong notebook conclusions
- [ ] Verify SHAP demographic weight = ~6% (không phải 30%)
- [ ] Bổ sung references (target: 18–20)
- [ ] (Optional) Thêm cross-validation cell
- [ ] (Optional) Thêm feature correlation check

---

*Review completed: 09/05/2026*
