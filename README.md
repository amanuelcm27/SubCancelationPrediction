# 📊 YouTube Trending Video Views Prediction (US + Canada)

## 🎯 Goal

The goal of this project is to predict the **view count** of trending YouTube videos using available metadata such as title, tags, likes, dislikes, publish time, and more. The dataset consists of YouTube trending data, starting with the **US dataset** for training and later evaluating generalization on the **Canada dataset**.

---

## 📁 Dataset

The dataset is publicly available via Kaggle:
- [Trending YouTube Video Statistics](https://www.kaggle.com/datasets/datasnaek/youtube-new)

We used:
- `USvideos.csv` for training
- `CAvideos.csv` for unseen test data

---

## 🔧 Feature Engineering & Preprocessing

### 🧹 Cleaned & Transformed Features:
- **Datetime columns**:
  - `publish_time` ➝ extracted:
    - `at_what_hour` (publish hour)
    - `dayofweek` (day of week)
  - `trending_date` ➝ converted to datetime
  - Created `how_long_till_trending` = time between publish and trend (in seconds)

- **Boolean columns**:
  - Converted `comments_disabled`, `ratings_disabled` to integers (`0` or `1`)

- **Dropped columns**:
  - Removed `category_id`, `thumbnail_link`, `video_error_or_removed`, and `description` to avoid noise
  - Dropped `publish_time`, `trending_date` after extracting useful info

---

## 🤖 Embedding Text Features (NLP)

I used **Sentence-BERT (SBERT)** via the `sentence-transformers` library to convert textual columns into embeddings.

- Model: `all-MiniLM-L6-v2`
- GPU Accelerated via `.to('cuda')` for fast encoding
- Embedded features:
  - `title`
  - `channel_title`
  - `tags`

Each embedding has a dimensionality of **384**, producing rich semantic vectors for each input string.

---

## 🧪 Final Feature Vector

After processing:
- `title_embedding` → (n, 384)
- `channel_title_embedding` → (n, 384)
- `tags_embedding` → (n, 384)
- Structured numeric features:
  - `likes`, `dislikes`, `comment_count`, `at_what_hour`, `dayofweek`, `how_long_till_trending`
- Final shape of `X`: (n, **1159** features)
Embedding the text to vector took time: **~20 Minutes**
Note: i **do not scale the target `views`** anymore, as it's unnecessary for tree-based models like XGBoost.

---

## 🏗️ Model Training

### Model: `XGBRegressor` from `xgboost`
- `n_estimators=100`
- `tree_method='hist'`
- `device='cuda'` for GPU acceleration (using RTX 3050 Ti)

Training time: **~4 seconds**

---

## 📈 Results

### ✅ On US dataset (train/test split):

- **R² Score**: `0.987`
- **MSE**: Very low (scaled value not shown here due to raw units)

### ✅ With 5-Fold Cross-Validation:

- **Avg R²**: `0.9908`
- **Avg MSE**: ~`9.85e-06` (when scaled)

### ✅ On Unseen Canadian Dataset (CA):

- **R² Score**: `0.8958` ✅
- **MSE**: `~1.2 trillion` (in raw view count scale)

> Performance on CA improved **significantly** after i stopped normalizing the target (`views`) — proving that tree models perform better with raw targets.

---

## 🧠 Insights & Learnings

- **Tree models don't need target scaling** → removing `MinMaxScaler` on `views` made predictions more accurate and interpretable
- **Embeddings improve performance** dramatically → capturing semantic meaning of titles/tags/channels gives the model deep insight
- **Cross-country generalization works** → The US-trained model still performs strongly on the Canadian dataset (both English-speaking)
- **GPU acceleration matters** → Sentence-BERT and XGBoost run **10x faster** on GPU
- **Data leakage can hurt** → Accidentally including `views` in features inflates performance. I fixed this and retrained properly.

---

## 📊 Visualization

I used scatter plots to visualize:
- Actual vs Predicted view counts
- Distributions of key features
- Cross-country feature comparison


---


---

## 📚 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `sentence-transformers`
- `xgboost`
- `scikit-learn`
- `torch` (for GPU usage)

---

## Credits

- Dataset: [Kaggle YouTube Trending Data](https://www.kaggle.com/datasets/datasnaek/youtube-new)
- Sentence Embeddings: [Sentence-Transformers](https://www.sbert.net/)
- Model: [XGBoost](https://xgboost.ai)

---


