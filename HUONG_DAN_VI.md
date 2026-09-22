# Hướng dẫn tái hiện FedGAGAN

Đây là bản cài đặt độc lập dựa trên bài báo *Enhancing IoT Intrusion Detection
Systems Through Horizontal Federated Learning and Optimized WGAN-GP*. Bài báo
không phát hành source code, vì vậy dự án này không phải mã gốc của tác giả.

## Tài liệu nên đọc

- `COLAB.md`: cài đặt và chạy trên Google Colab, lưu kết quả vào Google Drive.
- `TROUBLESHOOTING.md`: xử lý lỗi TensorFlow, GPU, RAM, dataset, label, NaN,
  GA chậm, Colab ngắt phiên và các lỗi đánh giá.
- `scripts/diagnose.py`: kiểm tra môi trường và dữ liệu trước khi train.

## 1. Luồng xử lý

1. Đọc một hoặc nhiều file CSV của một bộ dữ liệu IDS.
2. Chuyển nhãn gốc thành nhị phân: `0 = benign`, `1 = attack`.
3. Chia train/validation/test trước khi học bộ biến đổi dữ liệu.
4. Điền giá trị thiếu, one-hot biến phân loại, đưa đặc trưng về `[0,1]`.
5. Lấy riêng các mẫu attack trong tập train để huấn luyện WGAN-GP.
6. Nếu bật GA, tìm bộ siêu tham số có KL divergence thấp nhất.
7. Chia dữ liệu attack theo chiều ngang cho nhiều client.
8. Mỗi client nhận cùng trọng số global, train cục bộ và chỉ trả trọng số.
9. Server dùng FedAvg có trọng số theo số mẫu để cập nhật generator và critic.
10. Global generator sinh thêm mẫu attack.
11. Ghép mẫu sinh với dữ liệu thật để tạo tập hybrid.
12. Đánh giá chất lượng dữ liệu sinh và hiệu quả của mô hình IDS.

## 2. Vai trò từng module

### `config.py`

Đọc YAML vào các cấu hình có kiểu dữ liệu rõ ràng. Module kiểm tra các điều kiện
như tỉ lệ chia tập, số client, batch size và kiểu partition trước khi chạy.

### `data.py`

- Ghép nhiều CSV.
- Loại cột nhãn và các cột do người dùng chỉ định.
- Loại cột hằng.
- Chia tập có stratify.
- Fit preprocessing chỉ trên train để tránh data leakage.
- Tạo partition IID hoặc Dirichlet theo quy mô client.
- Lưu `splits.npz` và `preprocessor.joblib` để tái sử dụng.

### `models.py`

- Tạo generator từ nhiễu 32 chiều.
- Tạo critic đầu ra tuyến tính, không dùng sigmoid.
- Tính Wasserstein critic loss.
- Tính gradient penalty trên điểm nội suy thật/giả.
- Train critic `n_critic` lần cho mỗi lần train generator.

### `genetic.py`

Mỗi cá thể gồm learning rate, batch size, epochs, beta 1, beta 2, `n_critic` và
trọng số gradient penalty. Module dùng roulette selection, single-point
crossover, real mutation và elitism. Fitness là KL divergence, càng thấp càng tốt.

### `aggregation.py`

Chứa phép FedAvg độc lập với TensorFlow. Trọng số client được nhân theo số mẫu
cục bộ, thay vì lấy trung bình đều.

### `federated.py`

Mô phỏng Horizontal FL trong một tiến trình. Mỗi client có cùng không gian đặc
trưng nhưng giữ một tập hàng khác nhau. Generator và critic đều được train cục
bộ rồi tổng hợp riêng.

Nếu `federated.local_epochs` là `null`, số epoch do GA chọn sẽ được dùng. Nếu
đặt một số nguyên, giá trị đó sẽ ghi đè để thuận tiện cho thử nghiệm nhỏ.

### `metrics.py`

Tính KL theo histogram từng cột, sai khác ma trận tương quan, số dòng trùng tuyệt
đối và khoảng cách tới hàng thật gần nhất.

### `evaluation.py`

So sánh ba kịch bản train IDS:

- `real`: chỉ dữ liệu train thật;
- `synthetic_attacks_plus_real_benign`: attack sinh + benign thật;
- `hybrid`: toàn bộ train thật + attack sinh.

Ba classifier được dùng là Decision Tree, Logistic Regression và MLP. Mọi mô
hình được chấm trên cùng một tập test thật chưa từng tham gia huấn luyện.

### `pipeline.py`

Điều phối năm stage: `preprocess`, `search`, `train`, `generate`, `evaluate`.
Kết quả của từng stage được lưu để có thể chạy tiếp mà không làm lại từ đầu.

### `cli.py`

Cung cấp hai lệnh: tạo dữ liệu demo và chạy pipeline theo YAML.

## 3. Chạy nhanh

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

python -m fedgagan make-demo-data --output data/raw/demo.csv --rows 2400
python scripts/diagnose.py --config configs/demo.yaml
python -m fedgagan run --config configs/demo.yaml \
  --stages preprocess,train,generate,evaluate
```

Trên Windows, dùng `.venv\Scripts\activate` thay cho lệnh `source`.

## 4. Chạy cấu hình gần bài báo

Đặt CSV UNSW-NB15 vào `data/raw/UNSW-NB15/`, kiểm tra lại tên cột nhãn, rồi chạy:

```bash
python -m fedgagan run --config configs/paper.yaml \
  --stages preprocess,search,train,generate,evaluate
```

Cấu hình này rất nặng: GA của paper có 250 cá thể, 20 thế hệ và mỗi lần đánh giá
train 300 bước. Nên giảm population/generation/step để thử trước.

## 5. Kết quả đầu ra chính

| File | Ý nghĩa |
|---|---|
| `resolved_config.yaml` | Cấu hình tuyệt đối đã dùng cho lần chạy |
| `data_summary.json` | Số hàng, số đặc trưng và phân bố nhãn |
| `ga_best.json` | Cá thể GA tốt nhất |
| `ga_history.json` | KL tốt nhất/trung bình theo thế hệ |
| `models/generator.keras` | Global generator cuối cùng |
| `models/critic.keras` | Global critic cuối cùng |
| `federated_history.json` | Client được chọn và loss theo round |
| `synthetic_attacks.npy` | Mẫu sinh ở dạng ma trận encoded |
| `synthetic_attacks_encoded.csv` | Mẫu sinh có tên cột encoded |
| `distribution_metrics.json` | Metric thống kê, tương quan và privacy proxy |
| `classifier_metrics.csv` | Accuracy, precision, recall, F1 và ROC AUC |

## 6. Những điểm chưa thể sao chép tuyệt đối

Bài báo không cho biết chính xác số client, số round, local epoch, tỉ lệ chọn
client, cách chia dữ liệu client, toàn bộ kích thước từng layer và file dữ liệu
đầu vào. Những phần đó được cấu hình hóa và ghi rõ là lựa chọn tái hiện. Vì thiếu
các thông tin này, mục tiêu hợp lý là tái hiện phương pháp và xu hướng kết quả,
không cam kết trùng từng con số trong bảng của paper.

Horizontal FL ở đây cũng không đồng nghĩa với bảo mật tuyệt đối. Nếu cần tuyên
bố privacy chính thức, phải bổ sung secure aggregation hoặc differential privacy
và đánh giá riêng.

## 7. Kiểm tra và xử lý sự cố

Trước mỗi lần chạy dataset thật:

```bash
python scripts/diagnose.py --config configs/my_run.yaml
```

Trên Colab, thêm `--require-gpu`. Không bắt đầu train nếu dòng JSON cuối báo
`"status": "FAIL"`. Tra lỗi cụ thể trong `TROUBLESHOOTING.md`.
