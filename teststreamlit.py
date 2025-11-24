# app_streamlit.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# ----------------------
# CONFIG
# ----------------------
# Paths to models/data
MODEL_PATH = Path("rf_pipeline.pkl")
ISO_PATH = Path("isolation_forest.pkl")
SAMPLE_PATH = Path("sample_data.csv")
FI_CSV = Path("feature_importances.csv")

# logo path from your uploaded assets (kept from conversation)
LOGO_PATH = "/mnt/data/cf757764-11bb-473e-a093-e6e70fa0bf21.png"

PENDING_PATH = Path("pending_listings.csv")
LOG_PATH = Path("prediction_logs.csv")

ADMIN_PASSWORD = "123@"  

CURRENT_YEAR = datetime.now().year

# Streamlit page config
st.set_page_config(page_title="Dự đoán giá - Xe máy cũ", layout="wide", initial_sidebar_state="collapsed")

# ----------------------
# CUSTOM CSS 
# ----------------------
st.markdown("""
<style>

.navbar {
    display: flex;
    justify-content: center;
    gap: 18px;
    padding: 10px 0 25px 0;
}

.nav-item {
    background: #e8f0fc;
    color: #003366 !important;
    padding: 12px 24px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 16px;
    min-width: 160px;
    text-align: center;
    border: 1px solid #bcd2f0;
    transition: 0.25s;
}

.nav-item:hover {
    background: #cfe2ff;
    border-color: #6da2f7;
    cursor: pointer;
}

.header-box {
    padding: 35px;
    background: linear-gradient(to right, #eef5ff, #ffffff);
    border-radius: 18px;
    margin-top: 5px;
}

.feature-card {
    background: #ffffff;
    border: 1px solid #d7e3f5;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    transition: 0.25s;
    height: 170px;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    cursor: pointer;
}

</style>
""", unsafe_allow_html=True)


# top navigation helper
def top_nav(selected_page: str = None):
    # pages and labels
    st.image("Logoct.jpg")
    pages = [
        ("home", "Trang chủ"),
        ("problem", "Bài toán nghiệp vụ"),
        ("predict", "Dự đoán giá"),
        ("anom", "Kiểm tra bất thường"),
        ("admin", "Chế độ quản trị viên"),
        ("logs", "Nhật ký hệ thống"),
        ("report", "Đánh giá & Báo cáo"),
        ("team", "Thông tin nhóm thực hiện"),
    ]
    # render brand left
    brand_html = ""
    if Path(LOGO_PATH).exists():
        brand_html = f'<div class="brand"><img src="{LOGO_PATH}"/> <div style="font-weight:700;color:#0b57a4">XeMáy - Dự đoán giá</div></div>'
    st.markdown(f'<div class="top-nav">{brand_html}', unsafe_allow_html=True)
    # render items centered
    nav_html = ""
    for key, label in pages:
        style = "background: rgba(11,87,164,0.08);" if key == selected_page else ""
        nav_html += f'<div class="nav-item" onclick="window.streamlit.setComponentValue(\'{key}\')" style="{style}">{label}</div>'
    st.markdown(nav_html + "</div>", unsafe_allow_html=True)

# Provide a tiny JS bridge using components to catch clicks (fallback: use st.button)
# We'll implement navigation via session state buttons instead of JS for reliability.

# ----------------------
# Session state & navigation
# ----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False
if "admin_user" not in st.session_state:
    st.session_state.admin_user = None

# Simple top nav using columns and buttons 
def render_top_nav_buttons():
    cols = st.columns([1,1,1,1,1,1,1,1])
    labels = [("home","Trang chủ"),("problem","Bài toán nghiệp vụ"),("predict","Dự đoán giá"),
              ("anom","Kiểm tra bất thường"),("admin","Quản trị viên"),("logs","Nhật ký"),
              ("report","Báo cáo"),("team","Nhóm thực hiện")]
    for col, (key, lab) in zip(cols, labels):
        if col.button(lab):
            # admin button routes to auth if not logged in
            if key == "admin" and not st.session_state.admin_auth:
                st.session_state.page = "admin_login"
            else:
                st.session_state.page = key

# Render top hero/header
with st.container():
    render_top_nav_buttons()

# ----------------------
# SAFE model/data loader
# ----------------------
@st.cache_resource
def load_models():
    model = None
    iso = None
    sample = pd.DataFrame()
    errors = []
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        else:
            errors.append(str(MODEL_PATH))
    except Exception as e:
        errors.append(f"model:{e}")
    try:
        if ISO_PATH.exists():
            iso = joblib.load(ISO_PATH)
        else:
            errors.append(str(ISO_PATH))
    except Exception as e:
        errors.append(f"iso:{e}")
    try:
        if SAMPLE_PATH.exists():
            sample = pd.read_csv(SAMPLE_PATH)
            # basic normalization
            sample.columns = [c.strip() for c in sample.columns]
            for col in ["Gia_trieu","Giá","Khoảng giá min","Khoảng giá max"]:
                if col in sample.columns:
                    sample[col] = pd.to_numeric(sample[col], errors="coerce")
        else:
            errors.append(str(SAMPLE_PATH))
    except Exception as e:
        errors.append(f"sample:{e}")
    return model, iso, sample, errors

model, iso, sample_df, loader_errors = load_models()

# ----------------------
# Utility helpers
# ----------------------
def human_trieu(v):
    try:
        return f"{float(v):,.2f} Triệu"
    except:
        return v

def save_log(record: dict):
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame()
    df = pd.concat([pd.DataFrame([record]), df], ignore_index=True, sort=False)
    df.to_csv(LOG_PATH, index=False)

def add_pending(entry: dict):
    if PENDING_PATH.exists():
        df = pd.read_csv(PENDING_PATH)
    else:
        df = pd.DataFrame()
    entry["id"] = int(datetime.utcnow().timestamp() * 1000)
    df = pd.concat([pd.DataFrame([entry]), df], ignore_index=True, sort=False)
    df.to_csv(PENDING_PATH, index=False)
    return entry["id"]

# ----------------------
# PAGES
# ----------------------
def page_home():
    st.image("chotot.jpg")
    st.markdown("## <span style='color:#003366; font-weight:700;'>Ứng dụng dự đoán giá xe máy cũ</span>", unsafe_allow_html=True)

    st.markdown("""
    ### 📌 Giới thiệu  
    Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá bán…)  
    được thu thập từ **Chợ Tốt** (trước ngày **01/07/2025**).  
    Ứng dụng hỗ trợ:
    - Dự đoán giá xe cũ bằng Random Forest  
    - Phát hiện bất thường bằng Isolation Forest  
    - Dashboard thị trường xe máy Việt Nam
    """)

    st.markdown("## 📊 Thống kê mô tả nhanh")

    df = sample_df.copy()

    # ---- 1. Tuổi xe ----
    df["Tuổi xe"] = CURRENT_YEAR - df["Năm đăng ký"]
    fig1, ax1 = plt.subplots(figsize=(5,3))
    sns.histplot(df["Tuổi xe"], bins=20, kde=True, color="#0b57a4", ax=ax1)
    ax1.set_title("Phân bố tuổi xe")
    st.pyplot(fig1)

    # ---- 2. Top thương hiệu ----
    top_brands = df["Thương hiệu"].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(5,3))
    sns.barplot(x=top_brands.values, y=top_brands.index, palette="Blues_r", ax=ax2)
    ax2.set_title("Top 10 thương hiệu phổ biến")
    st.pyplot(fig2)

    # ---- 3. Khoảng giá ----
    price_col = "Gia_trieu" if "Gia_trieu" in df.columns else "Giá"
    fig3, ax3 = plt.subplots(figsize=(5,3))
    sns.histplot(df[price_col], bins=40, kde=True, color="#0b57a4", ax=ax3)
    ax3.set_title("Phân bố giá thị trường (Triệu)")
    st.pyplot(fig3)

    # ---- 4. Số km đã đi ----
    fig4, ax4 = plt.subplots(figsize=(5,3))
    sns.histplot(df["Số Km đã đi"], bins=40, kde=False, color="#0b57a4", ax=ax4)
    ax4.set_title("Phân bố số Km đã đi")
    st.pyplot(fig4)



def page_problem():
    st.image("xe_may.jpg")
    st.title("Bài toán nghiệp vụ")
    st.markdown("""
- **Mục tiêu:** Dự đoán giá bán hợp lý cho xe máy cũ và phát hiện tin đăng có giá bất thường.
- **Input:** Thương hiệu, Dòng xe, Năm đăng ký, Số Km, Loại xe, Dung tích, Xuất xứ, (Giá thực - tùy chọn).
- **Output:** Giá dự đoán (Triệu VNĐ) + Giải thích dạng tư vấn + Gợi ý hành động.
- **Phương pháp:** RandomForest cho regression; IsolationForest + thống kê cho anomaly detection.
    """)


def page_predict():
    st.image("xe_may_cu.jpg")
    st.title("Dự đoán giá")

    st.markdown("## 🔹 Chọn chế độ dự đoán")
    mode = st.radio(
        "Chọn phương thức dự đoán:",
        ["Nhập thủ công 1 xe", "Tải lên file CSV/XLSX (dự đoán hàng loạt)"],
        horizontal=True
    )

    # ==============================================================
    # 1) ---- MODE 1: MANUAL PREDICT ----
    # ==============================================================
    if mode == "Nhập thủ công 1 xe":
        with st.form("predict_form"):
            col1, col2 = st.columns([2,1])
            with col1:
                title = st.text_input("Tiêu đề tin đăng", value="Bán SH Mode 125 chính chủ")
                desc = st.text_area("Mô tả chi tiết", value="Xe đẹp, bao test, biển số TP.")
                brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique()))
                model_name = st.text_input("Dòng xe", value="")
                loai = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique()))
            with col2:
                dungtich = st.text_input("Dung tích", value="125")
                xuatxu = st.text_input("Xuất xứ", value="unknown")
                age = st.slider("Tuổi xe (năm)", 0, 50, 3)
                year_reg = CURRENT_YEAR - age
                km = st.number_input("Số Km đã đi", 0, 500000, value=20000, step=1000)
                price_input = st.number_input("Giá thực (Triệu, tùy chọn)", 0.0, value=0.0)
                min_p = st.number_input("Khoảng_giá_min (Triệu)", 0.0, value=0.0)
                max_p = st.number_input("Khoảng_giá_max (Triệu)", 0.0, value=0.0)

            save_flag = st.checkbox("Lưu để admin duyệt")
            submitted = st.form_submit_button("Dự đoán & Kiểm tra")

        if submitted:
            input_df = pd.DataFrame([{
                "Thương hiệu": brand,
                "Dòng xe": model_name or "unknown",
                "Năm đăng ký": int(year_reg),
                "Số Km đã đi": int(km),
                "Tình trạng": "Đã sử dụng",
                "Loại xe": loai,
                "Dung tích xe": dungtich,
                "Xuất xứ": xuatxu
            }])

            # ---- MODEL PREDICT ----
            if model is None:
                st.warning("Model chưa có — dùng giá trung vị mẫu.")
                pred = float(sample_df['Gia_trieu'].median())
            else:
                try:
                    pred = float(model.predict(input_df)[0])
                except Exception as e:
                    st.error("Lỗi predict: " + str(e))
                    pred = 0.0

            # ---- Anomaly reasoning ----
            brand_median = None
            if 'Thương hiệu' in sample_df.columns:
                dfb = sample_df[sample_df['Thương hiệu'] == brand]
                if not dfb.empty:
                    brand_median = float(dfb['Gia_trieu'].median())

            if price_input > 0:
                resid = price_input - pred
                if abs(resid) / (pred + 1e-6) < 0.15:
                    verdict = "Bình thường"
                    explanation = "Giá hợp lý, trong vùng an toàn."
                elif resid < 0:
                    verdict = "Giá thấp bất thường"
                    explanation = "Thấp hơn nhiều so với dự đoán — kiểm tra giấy tờ / tình trạng."
                else:
                    verdict = "Giá cao bất thường"
                    explanation = "Cao hơn thị trường — cân nhắc kiểm tra kỹ."
            else:
                verdict = "Không có giá thực"
                explanation = "Hệ thống chỉ dự đoán, không thể so sánh."

            # ---- OUTPUT ----
            st.markdown("### ✅ Kết quả dự đoán")
            st.write(f"**Giá dự đoán:** {human_trieu(pred)}")
            st.write(f"**Kết luận:** {verdict}")
            st.write(f"**Giải thích:** {explanation}")

            if brand_median:
                st.write(f"- Trung bình giá thương hiệu: {human_trieu(brand_median)}")

            # ---- SAVE ADMIN ----
            if save_flag:
                entry = {
                    "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                    "Tiêu_đề": title,
                    "Mô_tả_chi_tiết": desc,
                    "Thương hiệu": brand,
                    "Dòng xe": model_name,
                    "Năm đăng ký": year_reg,
                    "Số Km đã đi": km,
                    "Loại xe": loai,
                    "Dung tích xe": dungtich,
                    "Xuất xứ": xuatxu,
                    "Giá_thực": price_input,
                    "Giá_dự_đoán": pred,
                    "verdict": verdict
                }
                pid = add_pending(entry)
                st.success(f"Đã lưu submission (id={pid}) để admin duyệt.")

            save_log({
                "timestamp": datetime.now().isoformat(" ", "seconds"),
                "mode": "single",
                "pred": pred,
                "price_input": price_input,
                "verdict": verdict
            })

    # ==============================================================
    # 2) ---- MODE 2: BULK CSV/XLSX ----
    # ==============================================================
    else:
        st.markdown("### 📤 Tải lên file CSV hoặc XLSX để dự đoán hàng loạt")

        uploaded = st.file_uploader("Chọn file:", type=["csv", "xlsx"])

        if uploaded:
            # ---- READ INPUT FILE ----
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
            except Exception as e:
                st.error("Không đọc được file: " + str(e))
                return

            st.success(f"Đã tải file: {uploaded.name}")
            st.write("**Preview dữ liệu:**")
            st.dataframe(df.head(20))

            # ---- REQUIRED FIELDS ----
            required_cols = [
                "Thương hiệu","Dòng xe","Năm đăng ký","Số Km đã đi",
                "Loại xe","Dung tích xe","Xuất xứ"
            ]

            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error("Thiếu cột bắt buộc: " + ", ".join(missing))
                st.info("Bạn cần chuẩn hoá file trước khi dự đoán.")
                return

            if st.button("🚀 Chạy dự đoán cho toàn bộ file"):
                try:
                    if model is None:
                        df["Giá_dự_đoán"] = sample_df["Gia_trieu"].median()
                    else:
                        df["Giá_dự_đoán"] = model.predict(df[required_cols])

                    st.success("Hoàn tất dự đoán!")

                    st.write("### 🔎 Kết quả (20 dòng đầu):")
                    st.dataframe(df.head(20))

                    # --- ALLOW DOWNLOAD ---
                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Tải về file kết quả (CSV)",
                        csv_out,
                        file_name="du_doan_gia_xe.csv",
                        mime="text/csv"
                    )

                    save_log({
                        "timestamp": datetime.now().isoformat(" ", "seconds"),
                        "mode": "bulk",
                        "rows": len(df)
                    })

                except Exception as e:
                    st.error("Lỗi dự đoán hàng loạt: " + str(e))

def page_anom():
    st.title("Kiểm tra bất thường")

    with st.form("anom"):
        brand = st.text_input("Thương hiệu").strip()
        model_name = st.text_input("Dòng xe").strip()
        age = st.slider("Tuổi xe (năm)", 0, 50, 3)
        year_reg = CURRENT_YEAR - age
        km = st.number_input("Số Km đã đi", 0, 500000, 20000)
        xuatxu = st.text_input("Xuất xứ", value="unknown")
        gia = st.number_input("Giá thực (Triệu)", 0.0)
        submitted = st.form_submit_button("Check")

    if submitted:
        # VALIDATION ------------------------------------
        valid_brands = list(sample_df["Thương hiệu"].unique())

        if brand not in valid_brands:
            st.error("❌ Thương hiệu không tồn tại trong hệ thống. Vui lòng nhập lại.")
            st.info("Gợi ý: " + ", ".join(valid_brands[:20]) + " ...")
            return

        df_brand = sample_df[sample_df["Thương hiệu"] == brand]
        valid_models = list(df_brand["Dòng xe"].unique())

        if model_name not in valid_models:
            st.error("❌ Dòng xe không tồn tại. Vui lòng nhập lại đúng theo danh sách.")
            st.info("Gợi ý: " + ", ".join(valid_models[:20]) + " ...")
            return

        # ANOMALY ---------------------------------------
        dfb = df_brand
        p10 = dfb["Gia_trieu"].quantile(0.10)
        p90 = dfb["Gia_trieu"].quantile(0.90)

        if gia < p10:
            verdict = "Giá thấp bất thường"
        elif gia > p90:
            verdict = "Giá cao bất thường"
        else:
            verdict = "Bình thường"

        st.success(f"Kết luận: {verdict}")


def page_admin_login():
    st.title("Đăng nhập quản trị")
    pwd = st.text_input("Vui lòng nhập mật khẩu:", type="password")
    if st.button("Đăng nhập"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.admin_auth = True
            st.session_state.page = "admin"
            st.experimental_rerun()
        else:
            st.error("Sai mật khẩu. Vui lòng thử lại.")

def page_admin():
    if not st.session_state.admin_auth:
        st.warning("Bạn chưa đăng nhập admin.")
        st.session_state.page = "admin_login"
        return
    st.title("Chế độ quản trị viên")
    st.markdown("Chọn tab quản trị")
    tab = st.selectbox("Chức năng", ["Submissions", "Nhật ký hệ thống", "Đăng xuất"])
    if tab == "Submissions":
        if PENDING_PATH.exists():
            df = pd.read_csv(PENDING_PATH)
        else:
            df = pd.DataFrame()
        st.write("Submissions:", len(df))
        st.dataframe(df)
        if not df.empty:
            ids = st.multiselect("Chọn id để thao tác", df["id"].tolist())
            if st.button("Approve selected"):
                df.loc[df['id'].isin(ids), "status"] = "approved"
                df.to_csv(PENDING_PATH, index=False)
                st.success("Đã approve")
            if st.button("Reject selected"):
                df.loc[df['id'].isin(ids), "status"] = "rejected"
                df.to_csv(PENDING_PATH, index=False)
                st.warning("Đã reject")
    elif tab == "Nhật ký hệ thống":
        if LOG_PATH.exists():
            logs = pd.read_csv(LOG_PATH)
            st.write("Total logs:", len(logs))
            st.dataframe(logs.sort_values("timestamp", ascending=False).head(500))
            st.download_button("Export logs CSV", data=logs.to_csv(index=False).encode('utf-8'), file_name="logs.csv", mime="text/csv")
        else:
            st.info("Chưa có logs.")
        page_problem()
    elif tab == "Đăng xuất":
        st.session_state.admin_auth = False
        st.session_state.page = "home"
        st.experimental_rerun()

def page_logs():
    st.title("Nhật ký hệ thống")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        st.dataframe(logs.sort_values("timestamp", ascending=False).head(500))
    else:
        st.info("Chưa có logs.")

# advanced report 
def page_report():
    st.title("Đánh giá & Báo cáo kết quả")

    df = sample_df.copy()
    price_col = "Gia_trieu" if "Gia_trieu" in df.columns else "Giá"

    st.markdown("### 1️⃣ Phân bố giá tổng thể")
    fig, ax = plt.subplots(figsize=(8,3))
    sns.histplot(df[price_col], bins=40, kde=True, color="#0b57a4", ax=ax)
    st.pyplot(fig)
    st.info("Nhận xét: Giá xe tập trung chủ yếu trong khoảng 10–40 triệu. Một số dòng SH/PKL tạo đỉnh ở vùng giá cao.")

    st.markdown("### 2️⃣ Phân bố giá theo thương hiệu")
    top_brands = df["Thương hiệu"].value_counts().head(8).index
    subset = df[df["Thương hiệu"].isin(top_brands)]
    fig2, ax2 = plt.subplots(figsize=(10,4))
    sns.violinplot(
        x=price_col, y="Thương hiệu",
        data=subset,
        palette=sns.light_palette("#0b57a4", n_colors=len(top_brands)),
        ax=ax2
    )
    st.pyplot(fig2)
    st.info("Nhận xét: Honda và Yamaha có phân bố giá rộng; VinFast giá thấp và ổn định hơn.")

    st.markdown("### 3️⃣ Quan hệ Km – Giá bán")
    x = df["Số Km đã đi"]
    y = df[price_col]
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.scatter(x, y, s=12, alpha=0.3)
    m, b = np.polyfit(x.dropna(), y.dropna(), 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax3.plot(xs, m*xs + b, color="#0b57a4")
    st.pyplot(fig3)
    st.info("Nhận xét: Xe chạy nhiều Km giảm giá rõ rệt; sau 50.000 Km tốc độ giảm mạnh hơn.")

    st.markdown("### 4️⃣ Độ quan trọng các đặc trưng")
    if FI_CSV.exists():
        fi = pd.read_csv(FI_CSV)
        top = fi.head(15)
        fig4, ax4 = plt.subplots(figsize=(8,4))
        ax4.barh(top["feature"][::-1], top["importance"][::-1], color="#0b57a4")
        st.pyplot(fig4)
        st.info("Nhận xét: Km, Năm đăng ký và Thương hiệu là 3 yếu tố quyết định giá mạnh nhất.")

    st.markdown("### 5️⃣ Heatmap tương quan")
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig5, ax5 = plt.subplots(figsize=(7,6))
    sns.heatmap(corr, annot=False, cmap="Blues", ax=ax5)
    st.pyplot(fig5)
    st.info("Nhận xét: Km và giá có tương quan âm rõ rệt; năm đăng ký và giá tương quan dương mạnh.")


def page_team():
    st.title("Thông tin nhóm thực hiện")
    st.markdown("- THÀNH VIÊN 1:")
    st.markdown(" Họ tên: Nguyen Thai Binh")
    st.markdown(" Email: thaibinh782k1@gmail.com")
    st.markdown("- THÀNH VIÊN 2:")
    st.markdown(" Họ tên: Nguyen Duy Thanh")
    st.markdown(" Email: duythanh200620@gmail.com")
    st.markdown("- Repo: https://github.com/ThaiBinh78/ML07_Project")
    st.markdown("- Ngày báo cáo: 22/11/2025")
# ----------------------
# Router: display page based on session_state.page
# ----------------------
pages_map = {
    "home": page_home,
    "problem": page_problem,
    "predict": page_predict,
    "anom": page_anom,
    "admin_login": page_admin_login,
    "admin": page_admin,
    "logs": page_logs,
    "report": page_report,
    "team": page_team
}

# initial render
selected = st.session_state.page
if selected in pages_map:
    try:
        pages_map[selected]()
    except Exception as e:
        st.error("Lỗi khi render trang: " + str(e))
        st.write(traceback.format_exc())
else:
    page_home()



