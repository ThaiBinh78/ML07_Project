# app_motor_price.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import requests
from PIL import Image

# ----------------------
# CONFIG
# ----------------------
MODEL_PATH = "rf_pipeline.pkl"
ISO_PATH = "isolation_forest.pkl"
SAMPLE_PATH = "sample_data.csv"
FI_CSV = "feature_importances.csv"

BASE_DIR = Path(".")
PENDING_PATH = BASE_DIR / "pending_listings.csv"
LOG_PATH = BASE_DIR / "prediction_logs.csv"

CURRENT_YEAR = datetime.now().year

st.set_page_config(
    page_title="MotorPrice Pro - Dự đoán giá xe máy cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# CUSTOM CSS - FIXED FOR LIGHT/DARK MODE
# ----------------------
st.markdown("""
<style>
    /* Main background - compatible with both modes */
    .main-container {
        background: transparent;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: none;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white;
    }
    
    .main-header p {
        font-size: 1.4rem;
        opacity: 0.95;
        margin: 15px 0 0 0;
        font-weight: 300;
        color: white;
    }
    
    /* Card styling - compatible with both modes */
    .feature-card {
        background: var(--background-color, white);
        padding: 30px 25px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid var(--border-color, #e0e6ed);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .feature-card h3 {
        color: var(--text-color, #2c3e50);
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    .feature-card p {
        color: var(--secondary-text-color, #7f8c8d);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Text colors that work in both modes */
    .text-primary {
        color: var(--text-color, #2c3e50) !important;
    }
    
    .text-secondary {
        color: var(--secondary-text-color, #7f8c8d) !important;
    }
    
    .text-white {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    /* Metric cards */
    .stMetric {
        background: var(--background-color, white);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color, #e0e6ed);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        background: var(--background-color, white);
    }
    
    /* Form styling */
    .custom-form {
        background: var(--background-color, white);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color, #e0e6ed);
    }
    
    /* Sidebar menu items */
    .sidebar-menu-item {
        padding: 15px 20px;
        margin: 8px 0;
        border-radius: 12px;
        background: rgba(255,255,255,0.1);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        border: none;
        width: 100%;
        text-align: left;
    }
    
    .sidebar-menu-item:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(5px);
        color: white;
    }
    
    .sidebar-menu-item.active {
        background: rgba(255,255,255,0.25);
        border-left: 4px solid #e74c3c;
        color: white;
    }
    
    /* Price display card */
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    
    .price-card.normal {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    }
    
    .price-card.warning {
        background: linear-gradient(135deg, #f46b45 0%, #eea849 100%);
    }
    
    .price-card.danger {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    .price-card h2 {
        font-size: 1.8rem;
        margin: 0 0 15px 0;
        font-weight: 600;
        color: white;
    }
    
    .price-card h1 {
        font-size: 2.8rem;
        margin: 10px 0;
        font-weight: 800;
        color: white;
    }
    
    .price-card p {
        font-size: 1.2rem;
        margin: 0;
        opacity: 0.95;
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: var(--background-color, white);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color, #e0e6ed);
        margin: 15px 0;
    }
    
    .info-box h3, .info-box h4 {
        color: var(--text-color, #2c3e50);
        margin-top: 0;
    }
    
    .info-box p {
        color: var(--secondary-text-color, #7f8c8d);
    }
    
    /* Custom container */
    .custom-container {
        background: var(--background-color, white);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color, #e0e6ed);
        margin: 15px 0;
    }
    
    /* Fix Streamlit default text colors */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-color, #2c3e50) !important;
    }
    
    .stMarkdown p {
        color: var(--text-color, #2c3e50) !important;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        color: var(--text-color, #2c3e50) !important;
        background-color: var(--background-color, white) !important;
    }
    
    .stSelectbox>div>div>select {
        color: var(--text-color, #2c3e50) !important;
        background-color: var(--background-color, white) !important;
    }
    
    /* Table styling */
    .custom-table {
        background: var(--background-color, white);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* Section headers */
    .section-header {
        color: var(--text-color, #2c3e50);
        text-align: center;
        margin-bottom: 30px;
    }
    
    .section-subheader {
        color: var(--secondary-text-color, #7f8c8d);
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Helpers
# ----------------------
@st.cache_resource
def load_models_and_sample(rf_path, iso_path, sample_path):
    try:
        model = joblib.load(rf_path)
        iso = joblib.load(iso_path)
        sample = pd.read_csv(sample_path)
        # sanitize sample numeric columns
        for col in ["Gia_trieu", "Giá", "Khoảng giá min", "Khoảng giá max"]:
            if col in sample.columns:
                sample[col] = pd.to_numeric(sample[col], errors="coerce")
        return model, iso, sample
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, None, pd.DataFrame()

def ensure_cols_for_upload(df):
    required = [
        "Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe",
        "Năm_đăng_ký","Số_Km_đã_đi","Giá","Khoảng_giá_min","Khoảng_giá_max",
        "Tiêu_đề","Mô_tả_chi_tiết","Địa_chỉ"
    ]
    missing = [c for c in required if c not in df.columns]
    return missing

def add_pending(entry: dict):
    if PENDING_PATH.exists():
        df = pd.read_csv(PENDING_PATH)
    else:
        df = pd.DataFrame()
    entry_id = int(datetime.utcnow().timestamp() * 1000)
    entry["id"] = entry_id
    df = pd.concat([pd.DataFrame([entry]), df], ignore_index=True, sort=False)
    df.to_csv(PENDING_PATH, index=False)
    return entry_id

def log_prediction(record: dict):
    if Path(LOG_PATH).exists():
        logs = pd.read_csv(LOG_PATH)
    else:
        logs = pd.DataFrame()
    logs = pd.concat([pd.DataFrame([record]), logs], ignore_index=True, sort=False)
    logs.to_csv(LOG_PATH, index=False)

def human_currency(x):
    try:
        return f"{int(round(float(x))):,} Triệu"
    except:
        return x

def compute_anomaly_score(sample_df, brand, actual_price, pred, iso, X_trans_for_iso):
    """
    Compute 4 components:
    1) residual z (brand IQR or global std fallback)
    2) min/max violation
    3) outside [P10,P90]
    4) isolation forest raw score -> normalized
    Return final_score (0-100) and dict of details.
    """
    resid = (actual_price - pred) if not pd.isna(actual_price) else (0 - pred)
    sample_brand = sample_df[sample_df['Thương hiệu'] == brand] if 'Thương hiệu' in sample_df.columns else pd.DataFrame()
    # resid_z
    if len(sample_brand) >= 10 and 'Gia_trieu' in sample_brand.columns:
        iqr = (sample_brand['Gia_trieu'].quantile(0.75) - sample_brand['Gia_trieu'].quantile(0.25)) or 1.0
        resid_z = abs(resid) / iqr
    else:
        resid_z = abs(resid) / max(1.0, sample_df['Gia_trieu'].std() if 'Gia_trieu' in sample_df.columns else 1.0)
    # min/max
    min_price = sample_brand['Khoảng giá min'].min() if ('Khoảng giá min' in sample_brand.columns and len(sample_brand)>0) else np.nan
    max_price = sample_brand['Khoảng giá max'].max() if ('Khoảng giá max' in sample_brand.columns and len(sample_brand)>0) else np.nan
    violate_minmax = int((not pd.isna(min_price) and actual_price < min_price) or (not pd.isna(max_price) and actual_price > max_price))
    # p10/p90
    p10 = sample_brand['Gia_trieu'].quantile(0.10) if (len(sample_brand)>0 and 'Gia_trieu' in sample_brand.columns) else np.nan
    p90 = sample_brand['Gia_trieu'].quantile(0.90) if (len(sample_brand)>0 and 'Gia_trieu' in sample_brand.columns) else np.nan
    outside_p10p90 = int((not pd.isna(p10) and actual_price < p10) or (not pd.isna(p90) and actual_price > p90))
    # isolation: X_trans_for_iso must include residual appended (1xN)
    iso_vec = X_trans_for_iso
    # ensure dense
    if hasattr(iso_vec, "toarray"):
        iso_vec = iso_vec.toarray()
    iso_vec = np.asarray(iso_vec)
    # predict iso raw score
    try:
        iso_score_raw = - iso.decision_function(iso_vec.reshape(1, -1))[0]
        iso_flag = int(iso.predict(iso_vec.reshape(1, -1))[0] == -1)
    except Exception:
        # fallback to 0
        iso_score_raw = 0.0
        iso_flag = 0
    # combine weights
    w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
    score1 = min(1.0, resid_z / 3.0) * 100.0
    score2 = violate_minmax * 100.0
    score3 = outside_p10p90 * 100.0
    score4 = min(1.0, iso_score_raw / 0.5) * 100.0
    final_score = w1*score1 + w2*score2 + w3*score3 + w4*score4
    return final_score, {
        "resid": float(resid),
        "resid_z": float(resid_z),
        "violate_minmax": int(violate_minmax),
        "outside_p10p90": int(outside_p10p90),
        "iso_flag": int(iso_flag),
        "iso_score_raw": float(iso_score_raw),
        "score_components": {"score1": score1, "score2": score2, "score3": score3, "score4": score4}
    }

# ----------------------
# Load models & sample
# ----------------------
try:
    model, iso, sample_df = load_models_and_sample(MODEL_PATH, ISO_PATH, SAMPLE_PATH)
except Exception as e:
    st.error("Không thể load model/sample. Kiểm tra đường dẫn:")
    st.write(str(e))
    st.stop()

# ----------------------
# SIDEBAR - Professional Navigation
# ----------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: white; font-size: 1.8rem; margin-bottom: 0;">🏍️ MotorPrice Pro</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">AI-Powered Motorcycle Valuation</p>
    </div>
    <hr style="border-color: rgba(255,255,255,0.2);">
    """, unsafe_allow_html=True)
    
    # Navigation menu
    menu_options = {
        "🏠 Trang Chủ": "home",
        "📊 Dự Đoán Giá": "prediction",
        "🔍 Kiểm Tra Bất Thường": "anomaly", 
        "📈 Báo Cáo & Thống Kê": "reports",
        "🛠️ Quản Trị Viên": "admin",
        "📋 Nhật Ký Hệ Thống": "logs",
        "👨‍💻 Nhóm Thực Hiện": "team"
    }
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    
    for menu_name, page_key in menu_options.items():
        if st.button(menu_name, key=page_key, use_container_width=True):
            st.session_state.current_page = page_key

# ----------------------
# HEADER - Professional Header
# ----------------------
st.markdown("""
<div class="main-header">
    <h1 class="text-white">🏍️ MotorPrice Pro</h1>
    <p class="text-white">Hệ Thống Dự Đoán Giá Xe Máy Cũ Thông Minh Sử dụng AI</p>
</div>
""", unsafe_allow_html=True)

# ----------------------
# PAGE: HOME
# ----------------------
if st.session_state.current_page == "home":
    st.markdown("""
    <div class="section-header">
        <h2>Chào mừng đến với MotorPrice Pro</h2>
        <p class="section-subheader">Công nghệ AI tiên tiến giúp bạn dự đoán giá xe máy cũ chính xác và phát hiện các giao dịch bất thường</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 class="text-primary">📊 Dự Đoán Giá Thông Minh</h3>
            <p class="text-secondary">Sử dụng machine learning và AI để dự đoán giá xe chính xác dựa trên đặc điểm và tình trạng xe</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 class="text-primary">🔍 Phát Hiện Bất Thường</h3>
            <p class="text-secondary">Hệ thống cảnh báo thông minh giúp phát hiện giá bất thường và nghi ngờ gian lận</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3 class="text-primary">📈 Phân Tích Thị Trường</h3>
            <p class="text-secondary">Theo dõi xu hướng giá và phân tích thị trường xe máy cũ toàn diện</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h2 class="text-primary">Thống Kê Hệ Thống</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Dữ Liệu Huấn Luyện", f"{len(sample_df):,}", "mẫu")
    
    with col2:
        try:
            n_trees = model.named_steps['rf'].n_estimators if model else "N/A"
            st.metric("🌳 Số Cây Random Forest", str(n_trees))
        except:
            st.metric("🌳 Số Cây Random Forest", "N/A")
    
    with col3:
        if PENDING_PATH.exists():
            pending_df = pd.read_csv(PENDING_PATH)
            pending_count = len(pending_df)
        else:
            pending_count = 0
        st.metric("⏳ Đang Chờ Duyệt", f"{pending_count}", "submission")
    
    with col4:
        if LOG_PATH.exists():
            logs_df = pd.read_csv(LOG_PATH)
            log_count = len(logs_df)
        else:
            log_count = 0
        st.metric("📝 Lượt Dự Đoán", f"{log_count:,}", "lượt")

# ----------------------
# PAGE: PREDICTION
# ----------------------
elif st.session_state.current_page == "prediction":
    st.markdown("""
    <div class="section-header">
        <h2 class="text-primary">📊 Dự Đoán Giá Xe</h2>
        <p class="section-subheader">Chọn phương thức nhập liệu phù hợp với nhu cầu của bạn</p>
    </div>
    """, unsafe_allow_html=True)
    
    mode = st.radio(
        "**Chọn chế độ dự đoán:**",
        ["Nhập thông tin thủ công", "Upload file CSV/XLSX (dự đoán hàng loạt)"],
        horizontal=True
    )
    
    if mode == "Nhập thông tin thủ công":
        with st.form("predict_form", clear_on_submit=False):
            st.markdown("""
            <div class="custom-form">
            """, unsafe_allow_html=True)
            
            st.markdown('<h3 class="text-primary">🚗 Thông Tin Xe</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<p class="text-secondary">📝 Thông tin cơ bản</p>', unsafe_allow_html=True)
                title = st.text_input("Tiêu đề tin đăng", value="Bán SH Mode 125 chính chủ")
                description = st.text_area("Mô tả chi tiết", value="Xe đẹp, bao test, biển số TP, giá có thương lượng.")
                brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique().tolist()))
                model_name = st.text_input("Dòng xe", placeholder="Ví dụ: SH 150i, Vision, etc.")
                loai = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique().tolist()))
            
            with col2:
                st.markdown('<p class="text-secondary">🔧 Thông số kỹ thuật</p>', unsafe_allow_html=True)
                dungtich = st.text_input("Dung tích xe", value="125", placeholder="Ví dụ: 125, 150, etc.")
                age = st.slider("Tuổi xe (năm)", 0, 50, 3)
                year_reg = int(CURRENT_YEAR - age)
                st.info(f"**Năm đăng ký:** {year_reg}")
                km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=20000, step=1000)
                price_input = st.number_input("Giá thực (Triệu VNĐ) — tùy chọn", min_value=0.0, value=0.0, step=1.0)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                save_flag = st.checkbox("💾 Lưu để admin duyệt")
            with col2:
                submitted = st.form_submit_button("🚀 Dự đoán & Kiểm tra", use_container_width=True)
        
        if submitted:
            # Build input dataframe
            input_df = pd.DataFrame([{
                "Thương hiệu": brand,
                "Dòng xe": model_name if model_name.strip() != "" else "unknown",
                "Năm đăng ký": year_reg,
                "Số Km đã đi": km,
                "Tình trạng": "Đã sử dụng",
                "Loại xe": loai,
                "Dung tích xe": dungtich,
                "Xuất xứ": "unknown"
            }])
            
            # Predict
            if model is None:
                st.warning("Model chưa có — dùng giá trung vị mẫu.")
                pred = float(sample_df['Gia_trieu'].median())
            else:
                try:
                    pred = float(model.predict(input_df)[0])
                except Exception as e:
                    st.error("Lỗi predict: " + str(e))
                    pred = 0.0
            
            # Anomaly detection and verdict
            if price_input > 0:
                resid = price_input - pred
                if abs(resid) / (pred + 1e-6) < 0.15:
                    verdict = "Bình thường"
                    explanation = "Giá hợp lý, trong vùng an toàn."
                    card_class = "normal"
                elif resid < 0:
                    verdict = "Giá thấp bất thường"
                    explanation = "Thấp hơn nhiều so với dự đoán — kiểm tra giấy tờ / tình trạng."
                    card_class = "danger"
                else:
                    verdict = "Giá cao bất thường"
                    explanation = "Cao hơn thị trường — cân nhắc kiểm tra kỹ."
                    card_class = "warning"
            else:
                verdict = "Không có giá thực để so sánh"
                explanation = "Hệ thống chỉ dự đoán, không thể so sánh với giá thực."
                card_class = ""
            
            # Display results in beautiful card
            pred_vnd = f"{pred * 1000000:,.0f}".replace(",", ".")
            
            st.markdown(f"""
            <div class="price-card {card_class}">
                <h2 class="text-white">Giá Ước Tính Thị Trường</h2>
                <h1 class="text-white">{pred_vnd} VND</h1>
                <p class="text-white">{verdict}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display input parameters
            st.markdown("""
            <div class="custom-container">
                <h4 class="text-primary">📋 Thông số đầu vào</h4>
            """, unsafe_allow_html=True)
            
            input_params = {
                "Thương hiệu": brand,
                "Dòng xe": model_name or "unknown",
                "Năm đăng ký": year_reg,
                "Số Km đã đi": f"{km:,}".replace(",", "."),
                "Tình trạng": "Đã sử dụng",
                "Loại xe": loai,
                "Dung tích": f"{dungtich} cc",
                "Xuất xứ": "Việt Nam"
            }
            
            params_df = pd.DataFrame(list(input_params.items()), columns=["Thuộc tính", "Giá trị"])
            st.table(params_df)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Explanation
            st.markdown(f"""
            <div class="info-box">
                <h4 class="text-primary">📝 Giải thích</h4>
                <p class="text-secondary">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save to admin if requested
            if save_flag:
                entry = {
                    "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                    "Tiêu_đề": title,
                    "Mô_tả_chi_tiết": description,
                    "Thương hiệu": brand,
                    "Dòng xe": model_name,
                    "Năm đăng ký": year_reg,
                    "Số Km đã đi": km,
                    "Loại xe": loai,
                    "Dung tích xe": dungtich,
                    "Giá_thực": price_input,
                    "Giá_dự_đoán": pred,
                    "verdict": verdict
                }
                pid = add_pending(entry)
                st.success(f"✅ Đã lưu submission (id={pid}) để admin duyệt.")
    
    else:  # Batch prediction mode
        st.markdown("""
        <div class="custom-form">
            <h3 class="text-primary">📁 Upload File Dự Đoán Hàng Loạt</h3>
            <p class="text-secondary">File cần có các cột: Thương_hiệu, Dòng_xe, Loại_xe, Dung_tích_xe, Năm_đăng_ký, Số_Km_đã_đi, Giá (tùy chọn)</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Chọn file CSV hoặc Excel", type=["csv", "xlsx"])
        
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                
                st.success(f"✅ Đã tải file: {uploaded.name}")
                
                # Display preview
                st.markdown("**👀 Preview dữ liệu:**")
                st.dataframe(df.head(10))
                
                # Check required columns
                required_cols = ["Thương_hiệu", "Dòng_xe", "Năm_đăng_ký", "Số_Km_đã_đi", "Loại_xe", "Dung_tích_xe"]
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    st.error(f"❌ Thiếu cột bắt buộc: {', '.join(missing)}")
                else:
                    if st.button("🚀 Chạy dự đoán cho toàn bộ file", use_container_width=True):
                        with st.spinner("Đang xử lý dự đoán..."):
                            # Perform batch prediction
                            if model is None:
                                df["Giá_dự_đoán"] = sample_df["Gia_trieu"].median()
                            else:
                                # Prepare input data
                                input_data = pd.DataFrame({
                                    "Thương hiệu": df["Thương_hiệu"],
                                    "Dòng xe": df["Dòng_xe"].fillna("unknown"),
                                    "Năm đăng ký": df["Năm_đăng_ký"],
                                    "Số Km đã đi": df["Số_Km_đã_đi"],
                                    "Tình trạng": "Đã sử dụng",
                                    "Loại xe": df["Loại_xe"],
                                    "Dung tích xe": df["Dung_tích_xe"].astype(str),
                                    "Xuất xứ": "unknown"
                                })
                                df["Giá_dự_đoán"] = model.predict(input_data)
                            
                            st.success("✅ Hoàn tất dự đoán!")
                            
                            # Display results
                            st.markdown("**📊 Kết quả dự đoán (10 dòng đầu):**")
                            st.dataframe(df.head(10))
                            
                            # Download button
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "💾 Tải về file kết quả (CSV)",
                                data=csv,
                                file_name="ket_qua_du_doan.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            
            except Exception as e:
                st.error(f"❌ Lỗi khi đọc file: {str(e)}")

# ----------------------
# PAGE: ANOMALY DETECTION
# ----------------------
elif st.session_state.current_page == "anomaly":
    st.markdown("""
    <div class="section-header">
        <h2 class="text-primary">🔍 Kiểm Tra Bất Thường</h2>
        <p class="section-subheader">Phát hiện giá xe bất thường so với thị trường</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("anomaly_form"):
        st.markdown("""
        <div class="custom-form">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="text-secondary">🚗 Thông tin xe</p>', unsafe_allow_html=True)
            brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique()))
            model_name = st.text_input("Dòng xe", placeholder="Nhập dòng xe cụ thể")
            age = st.slider("Tuổi xe (năm)", 0, 50, 3)
            year_reg = CURRENT_YEAR - age
            km = st.number_input("Số Km đã đi", 0, 500000, 20000)
        
        with col2:
            st.markdown('<p class="text-secondary">💰 Thông tin giá</p>', unsafe_allow_html=True)
            actual_price = st.number_input("Giá thực tế (Triệu VNĐ)", 0.0, 1000.0, 50.0, step=1.0)
            loai_xe = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique()))
            dung_tich = st.text_input("Dung tích xe", value="125")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        submitted = st.form_submit_button("🔍 Kiểm tra bất thường", use_container_width=True)
    
    if submitted:
        # Simple anomaly detection based on brand and model
        brand_data = sample_df[sample_df['Thương hiệu'] == brand]
        
        if not brand_data.empty:
            # Calculate percentiles
            p10 = brand_data['Gia_trieu'].quantile(0.10)
            p25 = brand_data
