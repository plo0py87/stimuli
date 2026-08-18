# REFED 電極配置研究

用 [REFED 資料集](https://huggingface.co/datasets/REFED2025/REFED-dataset)（Ning et al.,
NeurIPS 2025 Datasets & Benchmarks）研究**頭戴式裝置的電極擺放**對連續情緒辨識的影響：
在同一套前處理與模型下比較 9 種電極配置，並用 [SEED](http://bcmi.sjtu.edu.cn/~seed/)
（Zheng & Lu）當對照組。

本目錄只放**實際跑模型的程式碼**。圖表、儀表板、結果 JSON 都不在 repo 裡（見最後一節）。

---

## 1. 環境

所有腳本都在專案共用的 LibEER venv 底下跑：

```bash
"C:/Dev/BCI/LibEER/.venv39/Scripts/python.exe" <script>.py
```

底下所有指令為了好讀都寫成 `python xxx.py`，**`python` 一律指上面那個 venv 的直譯器**
（換成你自己的環境路徑）。腳本都用 `Path(__file__)` 定位，所以要在 `refed/` 目錄底下執行。

需要 `torch`（CUDA 版）、`scikit-learn`、`scipy`、`numpy`。模型與前處理函式
（`DGCNN`、`data_utils.preprocess.lds` / `libeer_bandpass`）來自本 repo 上一層的
`emotion_model/`（vendored LibEER 副本）；每個腳本開頭的 `sys.path.insert(0, ...)`
就是在指這個路徑。換機器時**不需要**改這行，但要確認 `emotion_model/` 跟本目錄同層。

GPU 建議必備。跑 DGCNN 的腳本預設 `--device cuda`，用 CPU 會慢十幾倍。

## 2. 資料（不在 repo 裡，要自己下載）

| 資料集 | 預設路徑 | 內容 |
|---|---|---|
| REFED | `D:/EEG dataset/REFED` | 32 位受試者，每人 15 段影片，64ch @ 1000Hz，**每秒**連續 valence/arousal 搖桿標註。約 32GB。 |
| SEED | `D:/EEG dataset/SEED/SEED/SEED_EEG/Preprocessed_EEG` | 15 位受試者 × 3 session，62ch @ 200Hz，每個 trial 一個離散三分類標籤。 |

REFED 目錄結構需為 `data/<subject>/EEG_videos.mat` 與 `annotations/<subject>_label.mat`。

**換機器時要改路徑。** 三支抽取腳本吃 `--refed-root` 參數，可以直接在命令列覆蓋：

```bash
python refed_extract_all.py --refed-root "/your/path/REFED"
```

但有兩支**沒有**參數，路徑寫死在程式裡，要手動改：

| 檔案 | 位置 | 要改的東西 |
|---|---|---|
| `seed_montage_classify_persubj_trunc.py` | 第 36 行 | `SEED_DIR = Path("D:/EEG dataset/SEED/...")` |
| `svm_reproduce.py` | 第 244 行 | `root = Path("D:/EEG dataset/REFED")` |

（沒有 `--seed-root` 這個參數，別找了。）

## 3. 完整重現流程

依序執行。每一步都會產出 `.npz` / `.json`，被下一步吃掉。特徵檔約 250MB，已 gitignore。

### 3.1 特徵抽取（三種，用途不同，都要跑）

```bash
# (a) 全 64 通道的因果（realtime-legal）逐秒特徵 -> 主力電極配置研究用
python refed_extract_all.py                # -> refed_features_all64.npz

# (b) 完全照 LibEER 的 SEED 前處理流程 -> SVM 基線與 SEED 比較用
python refed_extract_libeer.py             # -> refed_features_libeer.npz

# (c) 早期的因果 DE+PSD 特徵 -> 跨受試者正規化實驗用
python refed_extract_features.py           # -> refed_features.npz
```

三個檔案的差別只在前處理，不要混用 —— 下面每個實驗都標了它吃哪一個。

### 3.2 時間平滑（在 (b) 之上）

```bash
python add_lds.py         # -> refed_features_libeer_lds.npz   非因果 LDS（LibEER 原版）
python add_kalman.py      # -> refed_features_libeer_kalman.npz 因果 Kalman（即時可用）
```

`add_lds.py` 的平滑**不是因果的**（`lds()` 在遞迴開始前就用了整段 trial 的平均），
只為了對齊 SEED benchmark 慣例而保留，**不能用在即時系統**。

### 3.3 主實驗：9 種電極配置（吃 `refed_features_all64.npz`）

```bash
# 連續 valence/arousal 回歸，指標是 CCC
python refed_montage_regress.py --modes per_subject

# 三分類版本
python refed_montage_classify.py

# 把預測值夾回合理範圍後重算指標（只處理迴歸，不處理分類）
python refed_clip_metrics.py
```

`refed_clip_metrics.py` 讀的是 `refed_montage_regress.py` 存的 `montage_predictions.npz`，
輸出 `montage_results_clipped.json`。分類那支存的是 `montage_predictions_cls.npz`，
不經過這一步。

9 種配置定義在 `refed_montage_regress.py` 的 `MONTAGES`：
`whole`（62ch，等同 SEED）、`quick20`（19ch，CGX Quick-20）、`headtop`（21ch，全擠頭頂）、
`glasses`（11ch，眼鏡腳）、`glasses_ear` / `glasses_mastoid` / `glasses_wide`（眼鏡腳逐步往後延伸）、
`vr_ring`（20ch，VR 綁帶橫跨額到枕）、`vr_ring_low`（綁帶再往下）。

**切折方式是照情緒平衡的，不是隨機**：15 段影片依目標情緒分成 5 組（`TARGET_GROUPS`），
3 折各取每組一段。改折數前先看懂這段邏輯。

> ⚠️ `--modes` 預設是 `["pooled", "per_subject"]`，但 **pooled 會洩漏**（見陷阱 1）。
> 一律加 `--modes per_subject`。

### 3.4 SVM 基線（吃 `refed_features_libeer*.npz`）

```bash
python svm_persubj_montage.py --features refed_features_libeer_lds.npz --out svm_persubj_montage.json
python svm_persubj_montage.py --features refed_features_libeer.npz     --out svm_persubj_montage_de.json
```

### 3.5 跨受試者正規化比較（吃 `refed_features.npz`）

```bash
python refed_train_dgcnn.py --seed 42 --out refed_results_seed42.json
python refed_train_dgcnn.py --seed 43 --out refed_results_seed43.json
python refed_train_dgcnn.py --seed 44 --out refed_results_seed44.json
python refed_aggregate.py                  # 跨 seed 彙總
```

> ⚠️ **`--out` 一定要給。** `refed_train_dgcnn.py` 的 `--out` 預設是 `refed_results.json`
> —— 不給的話三次會互相覆蓋同一個檔案；而 `refed_aggregate.py` 找的是
> `refed_results_seed*.json`（可用 `--glob` 改），會直接報
> `no result files matching` 結束。

REFED 每人只有一個 session，所以「跨 session」只能做成「跨受試者」。
用 subject-disjoint 4-fold × 3 seeds。

### 3.6 SEED 對照（截長度後的公平比較）

```bash
python seed_montage_classify_persubj_trunc.py            # DE_LDS
python seed_montage_classify_persubj_trunc.py --feat de  # 純 DE
```

會把每個 SEED trial 截到 REFED 的平均長度（約 102 秒，SEED 原生約 226 秒）再訓練，
這樣 REFED 落後 SEED 就不能用「SEED 每個 trial 資料比較多」解釋。

### 3.7 重現 REFED 論文的 SVM 基線

```bash
python svm_reproduce.py --exact
```

見陷阱 3 —— 這支腳本同時保留了「照論文文字」與「照論文程式碼」兩個版本的實作。

### 3.8 超參數

```bash
python tune_dgcnn.py
```

記錄 DGCNN 超參數是怎麼選出來的，不是每次重現都要跑。
**這支沒有任何命令列參數**（連 `--help` 都不會理你，直接開始訓練），
特徵檔路徑 `refed_features_all64.npz` 寫死在 `main()` 裡。會跑很久。

---

## 4. 腳本一覽

| 檔案 | 用途 | 吃 |
|---|---|---|
| `refed_extract_all.py` | 全 64ch 因果逐秒特徵 | REFED raw |
| `refed_extract_libeer.py` | LibEER/SEED 相同前處理 | REFED raw |
| `refed_extract_features.py` | 早期因果 DE+PSD | REFED raw |
| `add_lds.py` | 非因果 LDS 平滑 | `_libeer.npz` |
| `add_kalman.py` | 因果 Kalman 平滑 | `_libeer.npz` |
| `refed_montage_regress.py` | **9 配置 DGCNN 回歸（主結果）**；也是 `MONTAGES`/`FOLDS` 的共用模組 | `_all64.npz` |
| `refed_montage_classify.py` | 9 配置 DGCNN 三分類 | `_all64.npz` |
| `refed_clip_metrics.py` | 夾範圍後重算指標 | 上兩者的預測 |
| `svm_persubj_montage.py` | 每人 SVM 回歸＋分類基線 | `_libeer*.npz` |
| `refed_train_dgcnn.py` | 跨受試者，比較正規化方案 | `refed_features.npz` |
| `refed_aggregate.py` | 跨 seed 彙總 | 上者的 JSON |
| `seed_montage_classify_persubj_trunc.py` | SEED 截長度對照 | SEED raw |
| `svm_reproduce.py` | 重現論文 Table 6 | REFED raw |
| `tune_dgcnn.py` | 超參數探索 | `_all64.npz` |

---

## 5. 主要結果

**電極配置（valence CCC，pooled）**

| 配置 | 通道數 | CCC |
|---|---|---|
| whole | 62 | +0.205 |
| **vr_ring** | **20** | **+0.177**（全腦的 86%） |
| quick20 | 19 | +0.152 |
| glasses | 11 | +0.109 |
| headtop | 21 | +0.106 |

- **位置壓倒數量。** 20／19／21 通道的三組，電極數幾乎一樣，效果差到 1.7 倍。
  關鍵是**空間跨度** —— vr_ring 橫跨額到枕，headtop 全擠在頭頂。
- **綁帶往下壓到枕葉最好，再往下是淨負面。** 加 P7/P8、O1/OZ/O2 有效；
  但 M1/M2（乳突）、CB1/CB2（小腦）會變差 —— 那裡收到的頸部肌電多過腦訊號。
- **即時性在這個任務上不用付代價。** 因果特徵 +0.205 略勝 LibEER 非因果 +0.189。
  這跟 SEED 上「DE_LDS 值 10 分」的經驗相反，因為 SEED 標籤是靜態的、REFED 是動態的。
- **DGCNN ≈ SVM。** 每人每折只有約 8 段訓練影片，模型複雜度換不到東西，
  SVM 是合理的輕量替代。
- **正規化統計量不要跨 session 共用。** 用各人自己的 resting baseline 正規化，
  cross-subject macro recall 0.373 → 0.410（配對檢定 p=0.021）。幫助有限但真實。

**尚未排除的 confound**：前額電極（FP/AF）的 EOG，以及枕葉電極的視覺反應。
vr_ring 的優勢有多少來自「看到畫面」而非情緒本身，目前分不出來 —— **發表前必須補對照**。

---

## 6. 已知陷阱（動手前先讀）

1. **`--modes pooled` 會洩漏。** `refed_montage_regress.py` 的 `run_pooled()` 訓練時
   包含目標受試者自己的其他影片，回答的是「更多資料對這個人的留出影片有沒有幫助」，
   不是「能不能推廣到新的人」。要用就明確標成 within-subject，不要當成 cross-subject。

2. **REFED 上 accuracy 完全沒有鑑別力。** 多數類（positive）佔 45.7%，
   global normalization 的 accuracy 0.445 其實就是在猜多數類（negative recall 只有 0.20）。
   **一律看 macro recall。**

3. **論文的文字和它自己釋出的程式碼互相矛盾。** 論文 §4.2 與 Appendix E Eq.1 寫的是
   三分類門檻 `[0.4, 0.6]` 加封閉式高斯熵 DE 公式 —— 照這個做完全湊不出 Table 6 的數字
   （valence 0.47 vs 論文 0.524）。實際釋出的程式碼
   （[REFED-codes](https://github.com/REFED-dataset/REFED-codes)，`DE_PSD.py` +
   `utils_data.py` 的 `label_to_3c()`）用的是 STFT+Hanning window 的 PSD-based DE，
   門檻是 **`[0.3, 0.7]`**。照程式碼做就幾乎完全重現（valence 0.523 vs 0.524）。
   消融顯示**差距幾乎全來自門檻寬度**，DE 公式的選擇影響很小。
   → **要對論文數字時，先看它的程式碼，不要只信正文。**

4. **前處理不能無腦照搬。** 曾經以為 REFED 缺全域帶通是主因，照 LibEER 的 SEED 流程
   補上之後結果反而崩掉（CCC 0.205 → 0.049）。控制實驗確認關鍵是**時間平滑**：
   把 Kalman 加回去就恢復到 +0.189。原因是 REFED 標籤變化慢，未平滑的逐秒 DE
   時間尺度對不上；LibEER 的 plain DE 是為 SEED「每 trial 固定標籤、逐秒獨立分類」設計的。
   **同一套前處理換個任務就不是最佳解。**

5. **SVM 的 `max_iter` 不設上限會跑到天荒地老。** 純 DE（沒平滑、比較雜）會讓
   `svm_persubj_montage.py` 的某些 (kernel, C) 組合不收斂 —— 曾經單一折燒掉 9 小時以上
   CPU 時間。現在固定 `max_iter=20000`。若訓練停住又沒新 log，先確認 CPU 時間有沒有在累加，
   不要直接假設它只是慢。

6. **cross-subject 幾乎立刻過擬合。** best epoch 中位數只有 3–7（總共 60 epoch）。
   看到訓練曲線很早就到頂是正常的，不是設定錯誤。

---

## 7. repo 裡沒有什麼

本目錄只保留能重現模型結果的程式碼。以下**刻意不放**：

- **結果檔**（`*.json`）、**圖**（`*.png`）、**儀表板與報告**（`*.html`）
  —— 都是上面指令的產物，跑一次就有。
- **特徵快取與預測**（`*.npz`，約 250MB）、**執行紀錄**（`*.log`）。
- **探索性與作圖腳本** —— t-SNE / UMAP / silhouette / saliency 視覺化、
  各種 `fig_*` `dash_*` `build_*` 產圖與儀表板產生器、以及開發過程的診斷腳本。
  它們對重現結果沒有必要，留在原作者機器上。

需要那些的話直接找 Shine。
