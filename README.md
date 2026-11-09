# S04 – LIME và SHAP: thực nghiệm trên Breast Cancer Wisconsin

Thực nghiệm so sánh hai phương pháp giải thích mô hình **LIME** và **SHAP (KernelSHAP)** trên bộ dữ liệu Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer`), với hai mô hình: Logistic Regression và Random Forest. Mọi giải thích đều là cho **P(malignant)** (lớp 0 = ác tính).

## Cấu trúc thư mục

```
AI Seminar/
├── S04_LIME_SHAP_experiments.ipynb   # notebook thực nghiệm chính
├── S04_Seminar_LIME.pdf              # slide seminar
├── 1705.07874v2.pdf                  # tài liệu tham khảo (bài báo)
├── 2409.00265v2.pdf                  # tài liệu tham khảo (bài báo)
├── 2939672.2939778.pdf               # tài liệu tham khảo (bài báo)
├── figs/                             # hình sinh ra khi chạy notebook
└── results/                          # bảng CSV sinh ra khi chạy notebook
```

## Cài đặt

Cần Python 3 và các thư viện sau:

```bash
pip install numpy pandas matplotlib scipy scikit-learn lime shap jupyter
```

Nếu Python hệ thống báo lỗi `externally-managed-environment` (Homebrew), hãy dùng môi trường ảo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy scikit-learn lime shap jupyter
```

## Cách chạy

Chạy từ thư mục này để `figs/` và `results/` được tạo đúng chỗ:

```bash
jupyter notebook S04_LIME_SHAP_experiments.ipynb
```

Hoặc chạy không cần giao diện:

```bash
jupyter nbconvert --to notebook --execute S04_LIME_SHAP_experiments.ipynb --output S04_executed.ipynb
```

Các tham số (seed, số mẫu nhiễu, số background...) nằm trong dict `CFG` ở cell cấu hình đầu notebook.

## Nội dung notebook

| Mục | Nội dung |
|---|---|
| 0 | Cấu hình và các quyết định thiết kế |
| 1 | Dữ liệu và mô hình |
| 2 | Ví dụ tính tay: SHAP trên mô hình tuyến tính (2a), LIME từng bước trên mô hình phi tuyến nhỏ (2b) |
| 3 | Chọn 3 mẫu theo quy tắc cố định: đúng chắc chắn, sát ranh giới, dự đoán sai |
| 4 | Chạy LIME (nhiều seed) và SHAP (nhiều background) |
| 5 | Biểu đồ đóng góp đặc trưng |
| 6 | Bảng so sánh LIME vs SHAP |
| 7 | Độ ổn định, ví dụ giải thích bất ổn, độ nhạy theo background / kernel width |
| 8 | Kiểm tra faithfulness bằng deletion test |
| 9 | Checklist chuyển vào báo cáo |

## Kết quả đầu ra

**`figs/`**
- `contrib_LogReg.png`, `contrib_RandomForest.png`: đóng góp đặc trưng của LIME và SHAP cho 3 mẫu
- `unstable_example.png`: ví dụ giải thích bất ổn
- `deletion_test.png`: đường cong deletion test

**`results/`**
- `compare_lime_shap.csv`: overlap@k, Spearman, tỉ lệ đồng dấu, top-1, runtime
- `stability.csv`: Jaccard top-k và tỉ lệ top-1 chiếm ưu thế khi đổi seed / background
- `sensitivity.csv`: độ nhạy theo kích thước background (SHAP) và kernel width (LIME)
- `deletion_curves.csv`: xác suất dự đoán khi lần lượt xoá đặc trưng (LIME, SHAP, ngẫu nhiên)

## Lưu ý

- Thực nghiệm chỉ dùng một dataset, ba mẫu và hai mô hình, nên nhận xét cần dựa trên số liệu và nêu rõ giới hạn.
- SHAP được tính trên hàm xác suất (không phải log-odds), nên tổng đóng góp cộng tuyến tính trên xác suất.
- LIME mặc định rời rạc hoá đặc trưng liên tục, nên trọng số LIME là hiệu ứng của khoảng giá trị mà mẫu rơi vào.
