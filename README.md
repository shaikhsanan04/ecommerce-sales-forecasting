# E-Commerce Sales Forecasting — Streamlit App

A full-featured, dark-themed interactive Streamlit application for the Olist Brazilian E-Commerce Sales Forecasting ML project.

## 🚀 Live Demo
[Deploy on Streamlit Cloud →](#deployment)

## 📁 Repository Structure

```
ecommerce-sales-forecasting/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Streamlit theme config
├── datasets/
│   ├── olist_orders_dataset.csv
│   └── olist_order_items_dataset.csv
└── ecommerce_sales_forecasting_ml_project.ipynb
```

## 📦 App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Hero section, KPIs, full revenue timeline |
| 🔍 EDA | Data preview, distributions, temporal patterns |
| ⚙️ Feature Engineering | Pipeline steps, feature table, correlation |
| 🤖 Model Training | LR vs RF, code, coefficients, importances |
| 📊 Results & Evaluation | RMSE comparison, actual vs predicted, residuals |
| 💡 Business Insights | Actionable findings and conclusion |

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/shaikhsanan04/ecommerce-sales-forecasting
cd ecommerce-sales-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## ☁️ Deployment (Streamlit Cloud)

1. Push `app.py`, `requirements.txt`, and `.streamlit/config.toml` to your GitHub repo
2. Go to **https://share.streamlit.io**
3. Click **"New app"**
4. Select:
   - Repository: `shaikhsanan04/ecommerce-sales-forecasting`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy** — done! 🎉

> **Note:** Make sure the `datasets/` folder with both CSV files is committed to the repo.

## 📊 Dataset

**Olist Brazilian E-Commerce Public Dataset** (Kaggle)
- `olist_orders_dataset.csv` — 99,441 order records
- `olist_order_items_dataset.csv` — 112,650 item records

## 🧠 Models

| Model | RMSE |
|-------|------|
| Linear Regression | ~13,751 BRL |
| Random Forest | ~14,688 BRL |

Linear Regression wins — time-based features have a mostly linear relationship with daily revenue.

## 👤 Author

**Sanan Shaikh** — [github.com/shaikhsanan04](https://github.com/shaikhsanan04)
