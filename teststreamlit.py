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
# Paths to models/data (put these files in same folder)
MODEL_PATH = Path("rf_pipeline.pkl")
ISO_PATH = Path("isolation_forest.pkl")
SAMPLE_PATH = Path("sample_data.csv")
FI_CSV = Path("feature_importances.csv")

# logo path from your uploaded assets (kept from conversation)
LOGO_PATH = "/mnt/data/cf757764-11bb-473e-a093-e6e70fa0bf21.png"

PENDING_PATH = Path("pending_listings.csv")
LOG_PATH = Path("prediction_logs.csv")

ADMIN_PASSWORD = "123@"  # per your request

CURRENT_YEAR = datetime.now().year

# Streamlit page config
st.set_page_config(page_title="Dự đoán giá - Xe máy cũ", layout="wide", initial_sidebar_state="collapsed")

# =============================================================
# 1. CSS giao diện xanh – giống IFB2025
# =============================================================
st.markdown("""
<style>

.navbar {
    display: flex;
    justify-content: center;
    gap: 25px;
    padding: 15px 0px;
}

.navbtn {
    background: #0d1b2a;
    color: white;
    padding: 12px 18px;
    border-radius: 10px;
    font-size: 17px;
    border: 2px solid #0d1b2a;
    transition: 0.2s;
}

.navbtn:hover {
    border-color: #1b6ca8;
    background: #1b263b;
    cursor: pointer;
}

.headerbox {
    padding: 25px;
    background: linear-gradient(to right, #f5f9ff, #ffffff);
    border-radius: 12px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# 2. SESSION STATE điều hướng
# =============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"


# =============================================================
# 3. NAVBAR – Có key cho mọi button (KHÔNG BAO GIỜ LỖI)
# =============================================================
col = st.container()
with col:
    st.markdown('<div class="navbar">', unsafe_allow_html=True)

    if st.button("Trang chủ", key="nav_home"):
        st.session_state.page = "home"

    if st.button("Bài toán nghiệp vụ", key="nav_btv"):
        st.session_state.page = "business"

    if st.button("Dự đoán giá", key="nav_predict"):
        st.session_state.page = "predict"

    if st.button("Kiểm tra bất thường", key="nav_anom"):
        st.session_state.page = "anomaly"

    if st.button("Quản trị viên", key="nav_admin"):
        st.session_state.page = "admin"

    if st.button("Nhật ký", key="nav_log"):
        st.session_state.page = "log"

    if st.button("Đánh giá & Báo cáo", key="nav_report"):
        st.session_state.page = "report"

    if st.button("Nhóm", key="nav_team"):
        st.session_state.page = "team"

    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================
# 4. Render Page
# =============================================================
def page_home():
    st.markdown('<div class="headerbox">', unsafe_allow_html=True)
    st.markdown("## **Chào mừng — Ứng dụng dự đoán giá xe máy cũ**")
    st.write("Chọn một mục trên thanh menu để bắt đầu.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_business():
    st.write("### Bài toán nghiệp vụ")

def page_predict():
    st.write("### Trang dự đoán giá")

def page_anomaly():
    st.write("### Kiểm tra bất thường")

def page_admin():
    st.write("### Chế độ quản trị viên")

def page_log():
    st.write("### Nhật ký hệ thống")

def page_report():
    st.write("### Đánh giá & Báo cáo")

def page_team():
    st.write("### Nhóm thực hiện")


# Router
pages = {
    "home": page_home,
    "business": page_business,
    "predict": page_predict,
    "anomaly": page_anom,
    "admin": page_admin,
    "log": page_log,
    "report": page_report,
    "team": page_team,
}

pages[st.session_state.page]()

# ----------------------
# Session state & navigation
# ----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False
if "admin_user" not in st.session_state:
    st.session_state.admin_user = None

# Simple top nav using columns and buttons (reliable in Streamlit)
def render_top_nav_buttons():
    cols = st.columns([1,1,1,1,1,1,1,1])
    labels = [("home","Trang chủ"),("problem","Bài toán nghiệp vụ"),("predict","Dự đoán giá"),
              ("anom","Kiểm tra bất thường"),("admin","Quản trị viên"),("logs","Nhật ký"),
              ("report","Đánh giá & Báo cáo"),("team","Nhóm")]
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
    st.markdown("<div class='hero'><h2 style='color:#0b57a4'>Chào mừng — Ứng dụng dự đoán giá xe máy cũ</h2><p>Chọn một mục trên thanh menu để bắt đầu.</p></div>", unsafe_allow_html=True)
    if loader_errors:
        st.warning("Một số file model/data chưa có hoặc chưa load được. Kiểm tra: " + ", ".join(loader_errors))
    st.markdown("### Lựa chọn nhanh")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Dự đoán nhanh"):
            st.session_state.page = "predict"
    with c2:
        if st.button("Kiểm tra bất thường"):
            st.session_state.page = "anom"
    with c3:
        if st.button("Đánh giá & Báo cáo"):
            st.session_state.page = "report"

def page_problem():
    st.title("Bài toán nghiệp vụ")
    st.markdown("""
- **Mục tiêu:** Dự đoán giá bán hợp lý cho xe máy cũ và phát hiện tin đăng có giá bất thường.
- **Input:** Thương hiệu, Dòng xe, Năm đăng ký, Số Km, Loại xe, Dung tích, Xuất xứ, (Giá thực - tùy chọn).
- **Output:** Giá dự đoán (Triệu VNĐ) + Giải thích dạng tư vấn + Gợi ý hành động.
- **Phương pháp:** RandomForest cho regression; IsolationForest + thống kê cho anomaly detection.
    """)

def page_predict():
    st.title("Dự đoán giá — Nhập tay (User)")
    with st.form("predict_form"):
        col1, col2 = st.columns([2,1])
        with col1:
            title = st.text_input("Tiêu đề tin đăng", value="Bán SH Mode 125 chính chủ")
            desc = st.text_area("Mô tả chi tiết", value="Xe đẹp, bao test, biển số TP, giá có thương lượng.")
            brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique()) if 'Thương hiệu' in sample_df.columns else ["unknown"])
            model_name = st.text_input("Dòng xe", value="")
            loai = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique()) if 'Loại xe' in sample_df.columns else ["unknown"])
        with col2:
            dungtich = st.text_input("Dung tích", value="125")
            xuatxu = st.text_input("Xuất xứ", value="unknown")
            age = st.slider("Tuổi xe (năm)", 0, 50, 3)
            year_reg = CURRENT_YEAR - age
            km = st.number_input("Số Km đã đi", 0, 500000, value=20000, step=1000)
            price_input = st.number_input("Giá thực (Triệu, tùy chọn)", 0.0, value=0.0, step=0.1, format="%.2f")
            min_p = st.number_input("Khoảng_giá_min (Triệu)", 0.0, value=0.0, step=0.1, format="%.2f")
            max_p = st.number_input("Khoảng_giá_max (Triệu)", 0.0, value=0.0, step=0.1, format="%.2f")
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
        # try predict
        if model is None:
            st.warning("Model chưa có — ứng dụng hoạt động ở chế độ demo (trả về giá trung bình).")
            demo_pred = sample_df['Gia_trieu'].median() if 'Gia_trieu' in sample_df.columns else 0.0
            pred = float(demo_pred)
        else:
            try:
                pred = float(model.predict(input_df)[0])
            except Exception as e:
                st.error("Lỗi predict: " + str(e))
                pred = 0.0

        # compute simplistic anomaly reasoning for user-friendly explanation (no numeric score shown)
        # We'll derive residual vs brand median and produce human guidance.
        brand_median = None
        if 'Thương hiệu' in sample_df.columns and 'Gia_trieu' in sample_df.columns:
            dfb = sample_df[sample_df['Thương hiệu'] == brand]
            if len(dfb) > 0:
                brand_median = float(dfb['Gia_trieu'].median())

        # verdict & explanation
        if price_input > 0:
            resid = price_input - pred
            if abs(resid) / (pred+1e-6) < 0.15:
                verdict = "Bình thường"
                explanation = "Giá bạn nhập nằm trong vùng an toàn cho dòng xe này. Có thể đăng bán hoặc thương lượng."
            elif resid < 0:
                verdict = "Giá thấp bất thường"
                explanation = ("Giá này thấp hơn nhiều so với dự đoán. Nếu bạn là người bán, kiểm tra: "
                               "đơn vị nhập, biển số tỉnh, tình trạng sửa chữa/đã thay máy. Nếu mua: đề phòng lừa đảo.")
            else:
                verdict = "Giá cao bất thường"
                explanation = ("Giá cao hơn nhiều so với dự đoán. Kiểm tra tính xác thực, giấy tờ, hình ảnh thực tế.")
        else:
            verdict = "Không có giá thực để so sánh"
            explanation = "Bạn chưa nhập giá thực — hệ thống chỉ đưa ra giá dự đoán để tham khảo."

        # show results (user-friendly)
        st.markdown("### Kết quả")
        st.write(f"**Giá dự đoán:** {human_trieu(pred)}")
        st.write(f"**Kết luận:** {verdict}")
        st.write("**Giải thích:**")
        st.write(explanation)
        # more detailed reasons
        reasons = []
        if brand_median is not None:
            reasons.append(f"- Trung vị giá thương hiệu ({brand}) ≈ {human_trieu(brand_median)}")
        if price_input>0:
            reasons.append(f"- Chênh lệch so với giá bạn nhập: {human_trieu(price_input - pred)}")
        if not reasons:
            reasons.append("- Không đủ dữ liệu mẫu để phân tích thêm.")
        for r in reasons:
            st.write(r)

        # Save for admin if requested
        if save_flag:
            entry = {
                "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                "Tiêu_đề": title,
                "Mô_tả_chi_tiết": desc,
                "Địa_chỉ": "",
                "Thương hiệu": brand,
                "Dòng xe": model_name,
                "Năm đăng ký": year_reg,
                "Số Km đã đi": km,
                "Loại xe": loai,
                "Dung tích xe": dungtich,
                "Xuất xứ": xuatxu,
                "Giá_thực": (price_input if price_input>0 else np.nan),
                "Giá_dự_đoán": float(pred),
                "verdict": verdict,
                "notes": ""
            }
            pid = add_pending(entry)
            st.success(f"Đã lưu submission (id={pid}) để admin duyệt.")

        # Log
        save_log({
            "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
            "mode": "single",
            "pred": float(pred),
            "price_input": float(price_input) if price_input>0 else np.nan,
            "verdict": verdict
        })

def page_anom():
    st.title("Kiểm tra bất thường (nhanh)")
    with st.form("anom"):
        brand = st.text_input("Thương hiệu", value="unknown")
        model_name = st.text_input("Dòng xe", value="unknown")
        age = st.slider("Tuổi xe (năm)", 0, 50, 3)
        year_reg = CURRENT_YEAR - age
        km = st.number_input("Số Km đã đi", 0, 500000, value=20000, step=1000)
        xuatxu = st.text_input("Xuất xứ", value="unknown")
        gia = st.number_input("Giá thực (Triệu)", 0.0, value=0.0, step=0.1, format="%.2f")
        submitted = st.form_submit_button("Check")
    if submitted:
        # simple anomaly check using brand quantiles available in sample_df
        verdict = "Bình thường"
        explanation = "Giá nằm trong vùng an toàn."
        if 'Gia_trieu' in sample_df.columns and 'Thương hiệu' in sample_df.columns:
            dfb = sample_df[sample_df['Thương hiệu'] == brand]
            if len(dfb) >= 10:
                p10 = dfb['Gia_trieu'].quantile(0.10)
                p90 = dfb['Gia_trieu'].quantile(0.90)
                if not np.isnan(gia) and gia > 0:
                    if gia < p10:
                        verdict = "Giá thấp bất thường"
                        explanation = "Giá thấp hơn 10% mẫu; kiểm tra kỹ giấy tờ và tình trạng."
                    elif gia > p90:
                        verdict = "Giá cao bất thường"
                        explanation = "Giá cao hơn 90% mẫu; kiểm tra tính xác thực."
        st.write("Kết luận:", verdict)
        st.write("Giải thích:", explanation)
        save_log({
            "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
            "mode": "anom_quick",
            "pred": None,
            "price_input": float(gia) if gia>0 else np.nan,
            "verdict": verdict
        })

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
    tab = st.selectbox("Chức năng", ["Submissions", "Nhật ký hệ thống", "Đánh giá & Báo cáo", "Bài toán nghiệp vụ", "Đăng xuất"])
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
    elif tab == "Đánh giá & Báo cáo":
        st.write("Mời chuyển sang tab Đánh giá & Báo cáo (giống user) — admin có thể xem thêm chi tiết kỹ thuật ở đó.")
        st.session_state.page = "report"
    elif tab == "Bài toán nghiệp vụ":
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
        st.download_button("Export logs CSV", data=logs.to_csv(index=False).encode('utf-8'), file_name="logs.csv", mime="text/csv")
    else:
        st.info("Chưa có logs.")

# advanced report (6 plots with blue palette)
def page_report():
    st.title("Đánh giá & Báo cáo kết quả")
    price_col = 'Gia_trieu' if 'Gia_trieu' in sample_df.columns else ('Giá' if 'Giá' in sample_df.columns else None)
    if price_col is None:
        st.error("Sample data thiếu cột giá.")
        return
    df = sample_df.dropna(subset=[price_col]).copy()
    # 1 Histogram
    st.markdown("### 1. Phân bố giá tổng thể")
    fig, ax = plt.subplots(figsize=(8,3))
    sns.histplot(df[price_col], bins=40, kde=True, color="#0b57a4", ax=ax)
    ax.set_xlabel("Giá (Triệu)")
    st.pyplot(fig)
    # 2 Box/Violin by brand (top 8)
    st.markdown("### 2. Phân bố giá theo thương hiệu (violin/box)")
    top_brands = df['Thương hiệu'].value_counts().head(8).index.tolist() if 'Thương hiệu' in df.columns else []
    if top_brands:
        fig2, ax2 = plt.subplots(figsize=(10,4))
        subset = df[df['Thương hiệu'].isin(top_brands)]
        sns.violinplot(x='Gia_trieu', y='Thương hiệu', data=subset, order=top_brands, palette=sns.light_palette("#0b57a4", n_colors=len(top_brands)), ax=ax2)
        ax2.set_xlabel("Giá (Triệu)")
        st.pyplot(fig2)
    else:
        st.info("Không đủ dữ liệu để vẽ theo thương hiệu.")
    # 3 Scatter Km vs Price + trendline
    st.markdown("### 3. Số Km vs Giá (scatter + trendline)")
    if 'Số Km đã đi' in df.columns:
        x = pd.to_numeric(df['Số Km đã đi'], errors='coerce')
        y = pd.to_numeric(df[price_col], errors='coerce')
        mask = (~x.isna()) & (~y.isna())
        if mask.sum() > 10:
            fig3, ax3 = plt.subplots(figsize=(8,4))
            ax3.scatter(x[mask], y[mask], s=10, alpha=0.4)
            m, b = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax3.plot(xs, m*xs + b, color="#0366b3", linewidth=2)
            ax3.set_xlabel("Số Km đã đi")
            ax3.set_ylabel("Giá (Triệu)")
            st.pyplot(fig3)
        else:
            st.info("Không đủ dữ liệu Km.")
    else:
        st.info("Không có cột 'Số Km đã đi' trong mẫu.")
    # 4 Feature importances (group-level)
    st.markdown("### 4. Độ quan trọng các đặc trưng")
    if FI_CSV.exists():
        fi = pd.read_csv(FI_CSV)
        top = fi.head(20)
        fig4, ax4 = plt.subplots(figsize=(8,4))
        ax4.barh(top['feature'][::-1], top['importance'][::-1], color=sns.light_palette("#0b57a4", n_colors=len(top))[::-1])
        st.pyplot(fig4)
    else:
        st.info("feature_importances.csv không tìm thấy.")
    # 5 Heatmap numeric corr
    st.markdown("### 5. Heatmap tương quan numeric")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig5, ax5 = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax5)
        st.pyplot(fig5)
    else:
        st.info("Không đủ biến numeric.")
    # 6 Anomaly score distribution (from logs)
    st.markdown("### 6. Phân bố Anomaly Score (internal)")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        if 'anomaly_score' in logs.columns:
            fig6, ax6 = plt.subplots(figsize=(8,3))
            sns.histplot(logs['anomaly_score'].dropna(), bins=30, color="#0b57a4", ax=ax6)
            ax6.set_xlabel("Anomaly Score (internal)")
            st.pyplot(fig6)
        else:
            st.info("Chưa có trường anomaly_score trong logs.")
    else:
        st.info("Chưa có logs để vẽ.")

def page_team():
    st.title("Thông tin nhóm thực hiện")
    st.markdown("- Họ tên: Nguyen Thai Binh")
    st.markdown("- Email: thaibinh782k1@gmail.com")
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

