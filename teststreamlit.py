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

# ===== Audio Player cố định góc phải trên =====
audio_url = "https://raw.githubusercontent.com/ThaiBinh78/ML07_Project/main/Chill_Guy.mp3"

st.markdown(f"""
<style>
#fixed-audio {{
    position: fixed;
    top: 60px;         
    right: 20px;       
    width: 280px;       
    z-index: 9999;
    background: rgba(255,255,255,0.95);
    padding: 8px 12px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    display: flex;
    align-items: center;
    border: 1px solid #e0e6ed;
}}
#fixed-audio audio {{
    width: 100%;
    height: 30px;      
}}
</style>

<div id="fixed-audio">
    <audio controls autoplay loop>
        <source src="{audio_url}" type="audio/mpeg">
        Trình duyệt không hỗ trợ audio.
    </audio>
</div>
""", unsafe_allow_html=True)

# ----------------------
# CUSTOM CSS
# ----------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
   
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #435F7A 0%, #3498db 100%);
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
    }
   
    .main-header p {
        font-size: 1.4rem;
        opacity: 0.95;
        margin: 15px 0 0 0;
        font-weight: 300;
    }
   
    /* Card styling */
    .feature-card {
        background: white;
        padding: 30px 25px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
   
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
   
    .feature-card h3 {
        color: #435F7A;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 15px;
    }
   
    .feature-card p {
        color: #7f8c8d;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Custom metric cards for statistics */
    .metric-card-custom {
        background: white;
        padding: 25px 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e0e6ed;
        text-align: center;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card-custom:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    /* Ensure metric containers have equal height */
    [data-testid="metric-container"] {
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Ensure columns have equal height */
    .css-1r6slb0 {
        display: flex;
        flex-direction: column;
    }

    /* Hide empty delta arrows */
    [data-testid="metric-container"] [data-testid="stMetricDelta"]:empty {
        display: none;
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
    }
   
    /* Metric cards */
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e0e6ed;
        color: #435F7A !important; /* Dark text for light background */
    }
   
    .stMetric > div > div > div > p { /* Label */
        color: #435F7A !important;
    }
   
    .stMetric > div > div > div > small { /* Delta if present */
        color: #435F7A !important;
    }
   
    /* Dataframe styling */
    .dataframe {
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
   
    /* Form styling */
    .stForm {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
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
    }
   
    .sidebar-menu-item.active {
        background: rgba(255,255,255,0.25);
        border-left: 4px solid #e74c3c;
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
    }
   
    .price-card h1 {
        font-size: 2.8rem;
        margin: 10px 0;
        font-weight: 800;
    }
   
    .price-card p {
        font-size: 1.2rem;
        margin: 0;
        opacity: 0.95;
    }

    /* Custom container for team page */
    .custom-container {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #1a1a1a 0%, #435F7A 100%);
        }

        .main-header {
            background: linear-gradient(135deg, #34495e 0%, #435F7A 100%);
            color: #ffffff;
        }

        .feature-card {
            background: #435F7A;
            color: #ffffff;
            border-left: 5px solid #3498db;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }

        .feature-card h3 {
            color: #ffffff;
        }

        .feature-card p {
            color: #bdc3c7;
        }

        /* Custom metric cards in dark mode */
        .metric-card-custom {
            background: #34495e;
            color: #ffffff;
            border: 1px solid #435F7A;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }

        .stMetric {
            background: #34495e;
            color: #ffffff !important;
            border: 1px solid #435F7A;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }

        .stMetric > div > div > div > p {
            color: #ffffff !important;
        }

        .stMetric > div > div > div > small {
            color: #ffffff !important;
        }

        .stForm {
            background: #34495e;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }

        /* Dataframe in dark mode */
        .dataframe {
            background: #435F7A;
            color: #ffffff;
        }

        /* Adjust other elements as needed */
        [data-testid="stMarkdownContainer"] h2, h3, h4 {
            color: #ffffff !important;
        }

        [data-testid="stMarkdownContainer"] p {
            color: #bdc3c7 !important;
        }

        .custom-container {
            background: #34495e !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }

        /* Override inline styles for dark mode */
        div[style*="background: #f8f9fa"] {
            background: #435F7A !important;
        }

        h3[style*="color: #435F7A"], h4[style*="color: #435F7A"] {
            color: #ffffff !important;
        }

        p[style*="color: #5a6c7d"] {
            color: #bdc3c7 !important;
        }

        div[style*="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)"] {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        }

        div[style*="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%)"] {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
        }

        div[style*="background: #667eea"] {
            background: #3498db !important;
        }

        /* Timeline text */
        div[style*="text-align: center; flex: 1;"] p {
            color: #ffffff !important;
        }

        /* Audio player in dark mode */
        #fixed-audio {
            background: rgba(52, 73, 94, 0.95);
            border: 1px solid #435F7A;
        }
        /* Custom styling for batch prediction section */
        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .download-sample-btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        }
        
        .download-sample-btn:hover {
            background: linear-gradient(135deg, #218838 0%, #1e9e8a 100%) !important;
        }
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
def add_pending(entry: dict):
    if PENDING_PATH.exists():
        df = pd.read_csv(PENDING_PATH)
    else:
        df = pd.DataFrame()
    entry_id = int(datetime.utcnow().timestamp() * 1000)
    entry["id"] = entry_id
    # Thêm timestamp nếu chưa có
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.now().isoformat(sep=' ', timespec='seconds')
    df = pd.concat([pd.DataFrame([entry]), df], ignore_index=True, sort=False)
    df.to_csv(PENDING_PATH, index=False)
    return entry_id
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
        <h1 style="color: white; font-size: 1.8rem; margin-bottom: 0;">MotorPrice Pro</h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Motorcycle Valuation</p>
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
    
    # Thêm CSS để căn trái menu
    st.markdown("""
    <style>
    /* Căn trái các nút trong sidebar */
    .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 20px !important;
    }
    
    /* Đảm bảo icon và text căn đều trái */
    .stButton > button > div {
        justify-content: flex-start !important;
    }
    
    /* Căn trái nội dung trong nút */
    .stButton > button > div > p {
        text-align: left !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
   
    for menu_name, page_key in menu_options.items():
        if st.button(menu_name, key=page_key, use_container_width=True):
            st.session_state.current_page = page_key
# ----------------------
# HEADER - Professional Header
# ----------------------
st.markdown("""
<div class="main-header">
    <h1>🏍️ MotorPrice Pro</h1>
    <p>Hệ Thống Dự Đoán Giá Xe Máy Cũ Thông Minh</p>
</div>
""", unsafe_allow_html=True)
# ----------------------
# PAGE: HOME
# ----------------------
if st.session_state.current_page == "home":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h2 style="color: #435F7A; font-size: 2.2rem; margin-bottom: 15px;">Chào mừng đến với MotorPrice Pro</h2>
        <p style="color: #7f8c8d; font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
            Công nghệ tiên tiến giúp bạn dự đoán giá xe máy cũ chính xác và phát hiện các giao dịch bất thường
        </p>
    </div>
    """, unsafe_allow_html=True)
   
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h3>📊 Dự Đoán Giá Thông Minh</h3>
            <p>Sử dụng machine learning để dự đoán giá xe chính xác dựa trên đặc điểm và tình trạng xe</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h3>🔍 Phát Hiện Bất Thường</h3>
            <p>Hệ thống cảnh báo thông minh giúp phát hiện giá bất thường và nghi ngờ gian lận</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <h3>📈 Phân Tích Thị Trường</h3>
            <p>Theo dõi xu hướng giá và phân tích thị trường xe máy cũ toàn diện</p>
        </div>
        """, unsafe_allow_html=True)
   
    # Statistics Section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 40px 0;">
        <h2 style="color: #435F7A; font-size: 2rem;">Thống Kê Hệ Thống</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Dữ Liệu Huấn Luyện", f"{len(sample_df):,}", "mẫu")
    
    with col2:
        try:
            n_trees = model.named_steps['rf'].n_estimators if model else "N/A"
            # Thêm delta trống để đồng bộ chiều cao
            st.metric("🌳 Số Cây Random Forest", f"{n_trees}", "cây")
        except:
            st.metric("🌳 Số Cây Random Forest", "N/A", "cây")
    
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
# ----------------------
# PAGE: PREDICTION
# ----------------------
elif st.session_state.current_page == "prediction":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #435F7A; font-size: 2.2rem;">📊 Dự Đoán Giá Xe</h2>
        <p style="color: #7f8c8d; font-size: 1.1rem;">Chọn chế độ phù hợp với nhu cầu của bạn</p>
    </div>
    """, unsafe_allow_html=True)
   
    # Thay đổi radio button thành chọn người mua/người bán
    user_type = st.radio(
        "**Bạn là:**",
        ["👤 Người mua xe", "🏪 Người bán xe", "📁 Dự đoán hàng loạt (CSV/XLSX)"],
        horizontal=True
    )
   
    if user_type == "👤 Người mua xe":
        with st.form("buyer_form", clear_on_submit=False):
            st.markdown("""
            <div style="text-align: center;">
            """, unsafe_allow_html=True)
    
            st.subheader("🛒 Thông Tin Xe Bạn Muốn Mua")
            st.markdown("</div>", unsafe_allow_html=True)
           
            col1, col2 = st.columns(2)
           
            with col1:
                st.markdown("**📝 Thông tin xe**")
                brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique().tolist()))
                model_name = st.text_input("Dòng xe", placeholder="Ví dụ: SH 150i, Vision, etc.")
                loai = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique().tolist()))
                dungtich = st.text_input("Dung tích xe", value="125", placeholder="Ví dụ: 125, 150, etc.")
           
            with col2:
                st.markdown("**💰 Thông tin ngân sách**")
                budget = st.number_input("Ngân sách của bạn (Triệu VNĐ)", min_value=0.0, value=50.0, step=1.0)
                st.markdown("---")
                st.markdown("**ℹ️ Thông tin bổ sung**")
                age = st.slider("Tuổi xe mong muốn (năm)", 0, 10, 3)
                max_km = st.number_input("Số Km tối đa chấp nhận", min_value=0, max_value=100000, value=30000, step=1000)
           
            submitted = st.form_submit_button("🔍 Kiểm tra ngân sách", use_container_width=True)
       
        if submitted:
            # Kiểm tra xem dòng xe có trong dữ liệu không
            brand_models = sample_df[sample_df['Thương hiệu'] == brand]
            available_models = brand_models['Dòng xe'].dropna().unique().tolist()
           
            if model_name and model_name not in available_models:
                st.error(f"❌ Hiện chúng tôi chưa có dữ liệu về dòng xe **{model_name}** của **{brand}**")
                st.info("""
                **Đề xuất:** 
                - Bạn có thể gửi yêu cầu cập nhật thông tin cho dòng xe này
                - Admin sẽ xem xét và bổ sung dữ liệu trong thời gian sớm nhất
                """)
               
                if st.button("📩 Gửi yêu cầu cập nhật dữ liệu"):
                    suggestion_entry = {
                        "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                        "type": "data_request",
                        "brand": brand,
                        "model": model_name,
                        "loai_xe": loai,
                        "dung_tich": dungtich,
                        "status": "pending"
                    }
                    pid = add_pending(suggestion_entry)
                    st.success(f"✅ Đã gửi yêu cầu cập nhật (id={pid}). Chúng tôi sẽ thông báo khi có dữ liệu mới!")
            else:
                # Dự đoán giá
                input_df = pd.DataFrame([{
                    "Thương hiệu": brand,
                    "Dòng xe": model_name if model_name.strip() != "" else "unknown",
                    "Năm đăng ký": CURRENT_YEAR - age,
                    "Số Km đã đi": max_km,
                    "Tình trạng": "Đã sử dụng",
                    "Loại xe": loai,
                    "Dung tích xe": dungtich,
                    "Xuất xứ": "unknown"
                }])
               
                if model is None:
                    pred = float(sample_df['Gia_trieu'].median())
                else:
                    try:
                        pred = float(model.predict(input_df)[0])
                    except Exception as e:
                        st.error("Lỗi predict: " + str(e))
                        pred = 0.0
               
                # Tính toán giá min, max cho thương hiệu
                brand_data = sample_df[sample_df['Thương hiệu'] == brand]
                if not brand_data.empty:
                    min_price = brand_data['Gia_trieu'].min()
                    max_price = brand_data['Gia_trieu'].max()
                    median_price = brand_data['Gia_trieu'].median()
                   
                    # So sánh ngân sách với giá thị trường
                    budget_diff_min = min_price - budget
                    budget_diff_max = max_price - budget
                    budget_diff_median = median_price - budget
                   
                    st.markdown(f"""
                    <div class="price-card {'normal' if budget >= median_price else 'warning' if budget >= min_price else 'danger'}">
                        <h2>💰 Phân Tích Ngân Sách</h2>
                        <h1>{budget:.1f} Triệu VNĐ</h1>
                        <p>{'✅ Ngân sách phù hợp' if budget >= median_price else '⚠️ Cần xem xét' if budget >= min_price else '❌ Ngân sách chưa đủ'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                   
                    # Hiển thị so sánh chi tiết
                    col1, col2, col3 = st.columns(3)
                   
                    with col1:
                        st.metric("💸 Giá thấp nhất", f"{min_price:.1f} Triệu", 
                                 f"{budget_diff_min:+.1f} Triệu" if budget_diff_min != 0 else "Đủ")
                        st.caption("Xe cũ, tình trạng thấp")
                       
                    with col2:
                        st.metric("📊 Giá trung bình", f"{median_price:.1f} Triệu", 
                                 f"{budget_diff_median:+.1f} Triệu" if budget_diff_median != 0 else "Đủ")
                        st.caption("Xe trung bình, phổ biến")
                       
                    with col3:
                        st.metric("🚀 Giá cao nhất", f"{max_price:.1f} Triệu", 
                                 f"{budget_diff_max:+.1f} Triệu" if budget_diff_max != 0 else "Đủ")
                        st.caption("Xe mới, tình trạng tốt")
                   
                    # Khuyến nghị
                    st.markdown("---")
                    st.subheader("💡 Khuyến Nghị")
                   
                    if budget >= max_price:
                        st.success("""
                        **🎉 Ngân sách của bạn rất tốt!**
                        - Bạn có thể mua được xe trong tình trạng tốt nhất
                        - Nên tìm xe ít km, đời mới
                        - Có thể thương lượng giá tốt
                        """)
                    elif budget >= median_price:
                        st.info("""
                        **👍 Ngân sách phù hợp!**
                        - Bạn có thể mua được xe tình trạng trung bình đến tốt
                        - Tập trung vào xe 3-5 năm tuổi
                        - Kiểm tra kỹ lịch sử bảo dưỡng
                        """)
                    elif budget >= min_price:
                        st.warning("""
                        **⚠️ Ngân sách hơi thấp!**
                        - Chỉ có thể mua xe cũ, tình trạng thấp
                        - Cần kiểm tra kỹ giấy tờ và tình trạng xe
                        - Xem xét tăng ngân sách hoặc chọn dòng xe khác
                        """)
                    else:
                        st.error(f"""
                        **❌ Ngân sách chưa đủ!**
                        - Thiếu **{abs(budget_diff_min):.1f} Triệu** để mua xe rẻ nhất
                        - Thiếu **{abs(budget_diff_median):.1f} Triệu** để mua xe trung bình  
                        - Đề xuất: Tăng ngân sách hoặc chọn thương hiệu/dòng xe khác
                        """)

    elif user_type == "🏪 Người bán xe":
        with st.form("seller_form", clear_on_submit=False):
            st.markdown("""
            <div style="text-align: center;">
            """, unsafe_allow_html=True)
    
            st.subheader("🏍️ Thông Tin Xe Của Bạn")
            st.markdown("</div>", unsafe_allow_html=True)
           
            col1, col2 = st.columns([2, 1])
           
            with col1:
                st.markdown("**📝 Thông tin cơ bản**")
                title = st.text_input("Tiêu đề tin đăng", value="Bán SH Mode 125 chính chủ")
                description = st.text_area("Mô tả chi tiết", value="Xe đẹp, bao test, biển số TP, giá có thương lượng.")
                brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique().tolist()))
                model_name = st.text_input("Dòng xe", placeholder="Ví dụ: SH 150i, Vision, etc.")
                loai = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique().tolist()))
           
            with col2:
                st.markdown("**🔧 Thông số kỹ thuật**")
                dungtich = st.text_input("Dung tích xe", value="125", placeholder="Ví dụ: 125, 150, etc.")
                age = st.slider("Tuổi xe (năm)", 0, 50, 3)
                year_reg = int(CURRENT_YEAR - age)
                km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=20000, step=1000)
                selling_price = st.number_input("Giá muốn bán (Triệu VNĐ)", min_value=0.0, value=50.0, step=1.0)
           
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
           
            # So sánh giá muốn bán với giá dự đoán
            if selling_price > 0:
                resid = selling_price - pred
                if abs(resid) / (pred + 1e-6) < 0.15:
                    verdict = "Giá hợp lý"
                    explanation = "Giá của bạn phù hợp với thị trường, dễ bán."
                    card_class = "normal"
                elif resid < 0:
                    verdict = "Giá thấp"
                    explanation = "Giá của bạn thấp hơn thị trường — có thể bán nhanh nhưng lợi nhuận thấp."
                    card_class = "warning"
                else:
                    verdict = "Giá cao"
                    explanation = "Giá của bạn cao hơn thị trường — có thể khó bán, cần thương lượng."
                    card_class = "danger"
            else:
                verdict = "Chưa nhập giá bán"
                explanation = "Hệ thống chỉ dự đoán giá thị trường."
                card_class = ""
           
            # Display results
            pred_vnd = f"{pred * 1000000:,.0f}".replace(",", ".")
            selling_price_vnd = f"{selling_price * 1000000:,.0f}".replace(",", ".")
           
            st.markdown(f"""
            <div class="price-card {card_class}">
                <h2>💰 So Sánh Giá</h2>
                <div style="display: flex; justify-content: space-around; align-items: center;">
                    <div>
                        <p style="margin: 0; font-size: 1rem;">Giá bạn muốn bán</p>
                        <h3 style="margin: 10px 0; color: {'#ff6b6b' if selling_price > pred else '#51cf66'}">{selling_price_vnd} VND</h3>
                    </div>
                    <div style="font-size: 2rem;">🔄</div>
                    <div>
                        <p style="margin: 0; font-size: 1rem;">Giá thị trường</p>
                        <h3 style="margin: 10px 0;">{pred_vnd} VND</h3>
                    </div>
                </div>
                <p>{verdict}</p>
            </div>
            """, unsafe_allow_html=True)
           
            # Chi tiết so sánh
            st.markdown("""
            <div style="background: 435F7A; padding: 25px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); margin: 20px 0;">
                <h3 style="color: #435F7A; margin-top: 0;">📊 Phân Tích Chi Tiết</h3>
            """, unsafe_allow_html=True)
           
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chênh lệch giá", f"{abs(resid):.1f} Triệu", 
                         f"{resid:+.1f} Triệu" if selling_price > 0 else "N/A")
                st.metric("Tỷ lệ chênh lệch", f"{abs(resid)/pred*100:.1f}%")
               
            with col2:
                st.metric("Khuyến nghị", 
                         "Nên giảm giá" if resid > 0 else "Giá tốt" if resid < 0 else "Giá phù hợp")
                st.metric("Khả năng bán", 
                         "Khó" if resid > 0.2*pred else "Dễ" if resid < -0.1*pred else "Trung bình")
           
            st.markdown("</div>", unsafe_allow_html=True)
           
            # Explanation - Cập nhật màu sắc động
            if card_class == "normal":
                bg_color = "#d4edda"  # Xanh nhạt
                text_color = "#155724"
                border_color = "#c3e6cb"
                icon = "✅"
            elif card_class == "warning":
                bg_color = "#fff3cd"  # Vàng nhạt
                text_color = "#856404"
                border_color = "#ffeaa7"
                icon = "⚠️"
            else:  # danger
                bg_color = "#f8d7da"  # Đỏ nhạt
                text_color = "#721c24"
                border_color = "#f5c6cb"
                icon = "❌"

            # Explanation - Phiên bản nâng cao với icons
            st.markdown(f"""
            <div class="price-card {card_class}">
                <h2>💡 Khuyến Nghị Bán Xe</h2>
                <p style="font-size: 1.2rem; margin: 15px 0;">{explanation}</p>
                <div style="text-align: left; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 15px 0;">
                    <div style="display: flex; align-items: center; margin: 12px 0;">
                        <span style="font-size: 1.2rem; margin-right: 10px;">📸</span>
                        <span>Nên chụp ảnh thật rõ ràng, đầy đủ góc</span>
                    </div>
                    <div style="display: flex; align-items: center; margin: 12px 0;">
                        <span style="font-size: 1.2rem; margin-right: 10px;">📝</span>
                        <span>Mô tả chi tiết tình trạng xe, lịch sử bảo dưỡng</span>
                    </div>
                    <div style="display: flex; align-items: center; margin: 12px 0;">
                        <span style="font-size: 1.2rem; margin-right: 10px;">💰</span>
                        <span>Sẵn sàng thương lượng trong khoảng 5-10%</span>
                    </div>
                    <div style="display: flex; align-items: center; margin: 12px 0;">
                        <span style="font-size: 1.2rem; margin-right: 10px;">📄</span>
                        <span>Chuẩn bị đầy đủ giấy tờ: đăng ký, bảo hiểm</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
           
            # Save to admin if requested
            if save_flag:
                entry = {
                    "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                    "type": "seller_submission",
                    "Tiêu_đề": title,
                    "Mô_tả_chi_tiết": description,
                    "Thương hiệu": brand,
                    "Dòng xe": model_name,
                    "Năm đăng ký": year_reg,
                    "Số Km đã đi": km,
                    "Loại xe": loai,
                    "Dung tích xe": dungtich,
                    "Giá_muốn_bán": selling_price,
                    "Giá_dự_đoán": pred,
                    "verdict": verdict
                }
                pid = add_pending(entry)
                st.success(f"✅ Đã lưu submission (id={pid}) để admin duyệt.")

    else:  # Dự đoán hàng loạt
        st.markdown("""
        <div style="background: #435F7A; padding: 30px; border-radius: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
            <h3>📁 Upload File Dự Đoán Hàng Loạt</h3>
            <p>File cần có các cột: <strong>Thương_hiệu, Dòng_xe, Loại_xe, Dung_tích_xe, Năm_đăng_ký, Số_Km_đã_đi, Giá (tùy chọn)</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Tạo file mẫu
        sample_data = {
            "Thương_hiệu": ["Honda", "Yamaha", "SYM", "Piaggio", "Honda"],
            "Dòng_xe": ["Vision", "Exciter", "Attila", "Vespa", "SH"],
            "Loại_xe": ["Xe số", "Xe côn tay", "Xe tay ga", "Xe tay ga", "Xe tay ga"],
            "Dung_tích_xe": ["110", "150", "125", "150", "150"],
            "Năm_đăng_ký": [2020, 2019, 2021, 2020, 2022],
            "Số_Km_đã_đi": [15000, 20000, 10000, 8000, 5000],
            "Giá": [20.5, 35.0, 25.0, 80.0, 120.0]  # Giá tùy chọn
        }
        sample_df_csv = pd.DataFrame(sample_data)

        # Chuyển đổi thành CSV
        csv_sample = sample_df_csv.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Nút tải file mẫu
            st.download_button(
                label="📥 Tải file mẫu (CSV)",
                data=csv_sample,
                file_name="mau_du_lieu_xe_may.csv",
                mime="text/csv",
                use_container_width=True,
                help="Tải về file mẫu với định dạng CSV để nhập liệu. File đã bao gồm các cột cần thiết và dữ liệu mẫu."
            )

        with col2:
            st.info("""
            **💡 Hướng dẫn sử dụng file mẫu:**
            - Tải file mẫu về và mở bằng Excel hoặc Google Sheets
            - Điền thông tin xe của bạn vào các cột tương ứng
            - Cột **Giá** là tùy chọn, có thể để trống nếu chỉ muốn dự đoán
            - Lưu file dưới dạng CSV và upload lại hệ thống
            """)

        # Giữ nguyên code dự đoán hàng loạt
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
                    st.info(f"**Các cột cần có:** {', '.join(required_cols)}")
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
                           
                            # Download button for results
                            csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
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
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #435F7A; font-size: 2.2rem;">🔍 Kiểm Tra Bất Thường</h2>
        <p style="color: #7f8c8d; font-size: 1.1rem;">Phát hiện giá xe bất thường so với thị trường</p>
    </div>
    """, unsafe_allow_html=True)
   
    with st.form("anomaly_form"):
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("**🚗 Thông tin xe**")
            brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique()))
            model_name = st.text_input("Dòng xe", placeholder="Nhập dòng xe cụ thể")
            age = st.slider("Tuổi xe (năm)", 0, 50, 3)
            year_reg = CURRENT_YEAR - age
            km = st.number_input("Số Km đã đi", 0, 500000, 20000)
       
        with col2:
            st.markdown("**💰 Thông tin giá**")
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
            p25 = brand_data['Gia_trieu'].quantile(0.25)
            p75 = brand_data['Gia_trieu'].quantile(0.75)
            p90 = brand_data['Gia_trieu'].quantile(0.90)
            median_price = brand_data['Gia_trieu'].median()
           
            # Determine anomaly level
            if actual_price < p10:
                verdict = "Giá thấp bất thường"
                reason = "Thấp hơn 90% mẫu. Có thể xe bị lỗi / giấy tờ không rõ ràng."
                color = "danger"
                icon = "⚠️"
            elif actual_price > p90:
                verdict = "Giá cao bất thường"
                reason = "Cao hơn 90% mẫu. Nên kiểm tra thực tế hoặc thương lượng."
                color = "danger"
                icon = "⚠️"
            elif actual_price < p25:
                verdict = "Giá hơi thấp"
                reason = "Thấp hơn 75% mẫu. Có thể là cơ hội tốt nhưng cần kiểm tra kỹ."
                color = "warning"
                icon = "ℹ️"
            elif actual_price > p75:
                verdict = "Giá hơi cao"
                reason = "Cao hơn 75% mẫu. Có thể chấp nhận được nhưng nên thương lượng."
                color = "warning"
                icon = "ℹ️"
            else:
                verdict = "Giá bình thường"
                reason = "Giá nằm trong vùng an toàn so với thị trường."
                color = "normal"
                icon = "✅"
           
            # Display results
            st.markdown(f"""
            <div class="price-card {color}">
                <h2>{icon} {verdict}</h2>
                <p style="font-size: 1.1rem;">{reason}</p>
            </div>
            """, unsafe_allow_html=True)
           
            # Market statistics
            st.markdown("""
            <div style="background: #A364F5; padding: 25px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); margin: 20px 0;">
                <h3 style="color: #435F7A; margin-top: 0;">📊 Thống kê thị trường</h3>
            """, unsafe_allow_html=True)
            
                        # Thêm CSS để cải thiện giao diện metrics
            st.markdown("""
            <style>
                .custom-metric {
                    background: #A364F5;
                    padding: 20px;
                    border-radius: 15px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                    border-left: 4px solid #667eea;
                    margin-bottom: 20px;
                }
                .price-highlight {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                }
            </style>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            # ---------- Cột 1 ----------
            with col1:
                with st.container():
                    st.markdown(f"""
                    <div class="custom-metric">
                        <h3 style="color: #435F7A; margin: 0 0 10px 0; font-size: 1.1rem;">📊 Giá trung vị (Median)</h3>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #435F7A; margin: 0;">{median_price:.1f} Triệu</p>
                        <p style="color: #7f8c8d; font-size: 0.9rem; margin: 10px 0 0 0;">Giá trung bình của nhóm xe, 50% xe rẻ hơn và 50% xe đắt hơn mức này.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown(f"""
                    <div class="custom-metric">
                        <h3 style="color: #435F7A; margin: 0 0 10px 0; font-size: 1.1rem;">📉 Phân vị 25% (P25)</h3>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #435F7A; margin: 0;">{p25:.1f} Triệu</p>
                        <p style="color: #7f8c8d; font-size: 0.9rem; margin: 10px 0 0 0;">25% xe rẻ nhất nằm dưới mức giá này.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ---------- Cột 2 ----------
            with col2:
                with st.container():
                    st.markdown(f"""
                    <div class="custom-metric">
                        <h3 style="color: #435F7A; margin: 0 0 10px 0; font-size: 1.1rem;">📈 Phân vị 75% (P75)</h3> 
                        <p style="font-size: 1.5rem; font-weight: bold; color: #435F7A; margin: 0;">{p75:.1f} Triệu</p>
                        <p style="color: #7f8c8d; font-size: 0.9rem; margin: 10px 0 0 0;">75% xe rẻ hơn mức giá này, chỉ 25% xe đắt hơn.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Highlight giá của người dùng
                delta_value = ((actual_price - median_price) / median_price * 100) if median_price > 0 else 0
                
                with st.container():
                    st.markdown(f"""
                    <div class="price-highlight">
                        <h3 style="color: white; margin: 0 0 10px 0; font-size: 1.2rem;">💰 Giá của bạn</h3>
                        <p style="font-size: 1.8rem; font-weight: bold; color: white; margin: 0;">{actual_price:.1f} Triệu</p>
                        <p style="color: rgba(255,255,255,0.9); font-size: 1rem; margin: 10px 0 0 0;">
                            {f"{delta_value:+.1f}%" if median_price > 0 else "N/A"} so với trung vị
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ---------- Cột 3 ----------
            with col3:
                with st.container():
                    st.markdown(f"""
                    <div class="custom-metric">
                        <h3 style="color: #435F7A; margin: 0 0 10px 0; font-size: 1.1rem;">⬇️ Phân vị 10% (P10)</h3>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #435F7A; margin: 0;">{p10:.1f} Triệu</p>
                        <p style="color: #7f8c8d; font-size: 0.9rem; margin: 10px 0 0 0;">Nhóm xe rẻ nhất thị trường, chỉ 10% xe rẻ hơn mức giá này.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown(f"""
                    <div class="custom-metric">
                        <h3 style="color: #435F7A; margin: 0 0 10px 0; font-size: 1.1rem;">⬆️ Phân vị 90% (P90)</h3>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #435F7A; margin: 0;">{p90:.1f} Triệu</p>
                        <p style="color: #7f8c8d; font-size: 0.9rem; margin: 10px 0 0 0;">Giới hạn của nhóm xe rất đắt, chỉ còn 10% xe cao hơn mức giá này.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Thêm phần giải thích ngắn
            st.markdown("---")
            st.markdown("""
            <div style="background: #7510E0; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h4 style="color: #435F7A; margin-top: 0;">📖 Giải thích về phân vị giá:</h4>
                <p style="color: #5a6c7d; margin: 5px 0;">
                    • <strong>Phân vị 25% (P25):</strong> 25% xe có giá thấp hơn mức này<br>
                    • <strong>Phân vị 50% (Median):</strong> Giá trung bình - 50% xe rẻ hơn, 50% xe đắt hơn<br>
                    • <strong>Phân vị 75% (P75):</strong> 75% xe có giá thấp hơn mức này<br>
                    • <strong>Phân vị 10%/90%:</strong> Giới hạn dưới/trên của thị trường
                </p>
            </div>
            """, unsafe_allow_html=True)
                       
            st.markdown("</div>", unsafe_allow_html=True)
           
            # Recommendations
            st.markdown("""
            <div style="background: #1D56A3; padding: 25px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.08);">
                <h3 style="color: #435F7A; margin-top: 0;">💡 Khuyến nghị</h3>
            """, unsafe_allow_html=True)
           
            if color == "danger":
                st.warning("**CẢNH BÁO**: Giá xe có dấu hiệu bất thường rõ rệt. Nên:")
                st.write("- Kiểm tra kỹ lịch sử xe và giấy tờ")
                st.write("- Xem xét kỹ tình trạng thực tế")
                st.write("- Tham khảo ý kiến chuyên gia nếu cần")
            elif color == "warning":
                st.info("**LƯU Ý**: Giá xe có chút khác biệt so với thị trường. Cân nhắc:")
                st.write("- Thương lượng giá nếu cần thiết")
                st.write("- Kiểm tra lại các thông số kỹ thuật")
                st.write("- So sánh với các xe tương tự trên thị trường")
            else:
                st.success("**TỐT**: Giá xe nằm trong phạm vi hợp lý. Có thể:")
                st.write("- Tiếp tục đánh giá các yếu tố khác")
                st.write("- Kiểm tra tình trạng thực tế xe")
                st.write("- Xem xét mua nếu các yếu tố khác đều tốt")
           
            st.markdown("</div>", unsafe_allow_html=True)
       
        else:
            st.error("❌ Không tìm thấy dữ liệu cho thương hiệu này.")
# ----------------------
# PAGE: REPORTS & STATISTICS
# ----------------------
elif st.session_state.current_page == "reports":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #435F7A; font-size: 2.2rem;">📈 Báo Cáo & Thống Kê</h2>
        <p style="color: #7f8c8d; font-size: 1.1rem;">Phân tích dữ liệu và xu hướng thị trường</p>
    </div>
    """, unsafe_allow_html=True)
   
    tab1, tab2, tab3 = st.tabs(["📊 Thống Kê Tổng Quan", "📈 Phân Tích Xu Hướng", "🔍 Feature Importance"])
   
    with tab1:
        st.subheader("Thống Kê Dữ Liệu Mẫu")
        st.dataframe(sample_df.describe())
       
        col1, col2 = st.columns(2)
       
        with col1:
            # Price distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sample_df['Gia_trieu'].hist(bins=30, ax=ax, alpha=0.7, color='#667eea')
            ax.set_xlabel('Giá (Triệu VNĐ)')
            ax.set_ylabel('Số lượng')
            ax.set_title('Phân Phối Giá Xe')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
       
        with col2:
            # Top brands
            brand_counts = sample_df['Thương hiệu'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            brand_counts.plot(kind='barh', ax=ax, color='#764ba2', alpha=0.7)
            ax.set_xlabel('Số lượng')
            ax.set_title('Top 10 Thương Hiệu Phổ Biến')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
   
    with tab2:
        st.subheader("Phân Tích Xu Hướng Giá")
       
        # Year vs Price
        if 'Năm đăng ký' in sample_df.columns:
            year_price = sample_df.groupby('Năm đăng ký')['Gia_trieu'].mean().dropna()
            fig, ax = plt.subplots(figsize=(12, 6))
            year_price.plot(ax=ax, marker='o', color='#ff6b6b', linewidth=2)
            ax.set_xlabel('Năm Đăng Ký')
            ax.set_ylabel('Giá Trung Bình (Triệu VNĐ)')
            ax.set_title('Xu Hướng Giá Theo Năm Đăng Ký')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
   
    with tab3:
        st.subheader("Feature Importance")
        try:
            fi_df = pd.read_csv(FI_CSV)
            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(fi_df.head(15)))
            ax.barh(y_pos, fi_df['importance'].head(15), color='#667eea', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(fi_df['feature'].head(15))
            ax.set_xlabel('Importance')
            ax.set_title('Top 15 Features Quan Trọng Nhất')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
           
            st.dataframe(fi_df.head(20))
        except Exception as e:
            st.warning(f"Không thể load feature importance: {e}")
# ----------------------
# PAGE: ADMIN
# ----------------------
elif st.session_state.current_page == "admin":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #435F7A; font-size: 2.2rem;">🛠️ Quản Trị Viên</h2>
        <p style="color: #7f8c8d; font-size: 1.1rem;">Quản lý submissions và hệ thống</p>
    </div>
    """, unsafe_allow_html=True)
   
    admin_password = st.text_input("🔐 Mật khẩu quản trị", type="password")
   
    if admin_password == "admin123": # In production, use secure password hashing
        st.success("✅ Đăng nhập thành công!")
       
        tab1, tab2, tab3 = st.tabs(["📋 Submissions", "📊 Cập Nhật Dữ Liệu", "⚙️ Thông Tin Hệ Thống"])
       
        with tab1:
            if PENDING_PATH.exists():
                pending_df = pd.read_csv(PENDING_PATH)
                st.metric("Tổng Submissions", len(pending_df))
                st.dataframe(pending_df)
               
                if not pending_df.empty:
                    selected_id = st.selectbox("Chọn ID để thao tác", pending_df['id'].tolist())
                    col1, col2, col3 = st.columns(3)
                   
                    with col1:
                        if st.button("✅ Duyệt", use_container_width=True):
                            pending_df.loc[pending_df['id'] == selected_id, 'status'] = 'approved'
                            pending_df.to_csv(PENDING_PATH, index=False)
                            st.success("Đã duyệt submission!")
                            st.rerun()
                   
                    with col2:
                        if st.button("❌ Từ chối", use_container_width=True):
                            pending_df.loc[pending_df['id'] == selected_id, 'status'] = 'rejected'
                            pending_df.to_csv(PENDING_PATH, index=False)
                            st.warning("Đã từ chối submission!")
                            st.rerun()
                   
                    with col3:
                        if st.button("🗑️ Xóa", use_container_width=True):
                            pending_df = pending_df[pending_df['id'] != selected_id]
                            pending_df.to_csv(PENDING_PATH, index=False)
                            st.info("Đã xóa submission!")
                            st.rerun()
            else:
                st.info("📭 Chưa có submissions nào.")
       
        with tab2:
            st.subheader("📊 Cập Nhật Dữ Liệu Xe")
           
            update_method = st.radio(
                "Chọn phương thức cập nhật:",
                ["📝 Nhập thủ công", "📁 Upload file CSV"]
            )
           
            if update_method == "📝 Nhập thủ công":
                with st.form("manual_update_form"):
                    st.markdown("**Thông tin xe mới**")
                    col1, col2 = st.columns(2)
                   
                    with col1:
                        new_brand = st.text_input("Thương hiệu")
                        new_model = st.text_input("Dòng xe")
                        new_type = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique().tolist()))
                        new_cc = st.text_input("Dung tích xe")
                   
                    with col2:
                        new_year = st.number_input("Năm đăng ký", min_value=1990, max_value=CURRENT_YEAR, value=CURRENT_YEAR-3)
                        new_km = st.number_input("Số Km đã đi", min_value=0, value=20000)
                        new_price = st.number_input("Giá (Triệu VNĐ)", min_value=0.0, value=50.0)
                        new_condition = st.selectbox("Tình trạng", ["Đã sử dụng", "Mới"])
                   
                    if st.form_submit_button("💾 Thêm vào dữ liệu"):
                        new_data = {
                            "Thương hiệu": new_brand,
                            "Dòng xe": new_model,
                            "Loại xe": new_type,
                            "Dung tích xe": new_cc,
                            "Năm đăng ký": new_year,
                            "Số Km đã đi": new_km,
                            "Gia_trieu": new_price,
                            "Tình trạng": new_condition,
                            "Xuất xứ": "Việt Nam"
                        }
                       
                        # Thêm vào sample_data.csv
                        updated_df = pd.concat([sample_df, pd.DataFrame([new_data])], ignore_index=True)
                        updated_df.to_csv(SAMPLE_PATH, index=False)
                        st.success("✅ Đã thêm dữ liệu xe mới!")
                        st.rerun()
           
            else:  # Upload CSV
                st.markdown("""
                **📁 Upload file CSV để cập nhật dữ liệu**
                - File cần có các cột: Thương hiệu, Dòng xe, Loại xe, Dung tích xe, Năm đăng ký, Số Km đã đi, Gia_trieu
                - Dữ liệu mới sẽ được thêm vào cuối file
                """)
               
                uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
               
                if uploaded_file:
                    try:
                        new_data_df = pd.read_csv(uploaded_file)
                        required_cols = ["Thương hiệu", "Dòng xe", "Loại xe", "Dung tích xe", "Năm đăng ký", "Số Km đã đi", "Gia_trieu"]
                       
                        if all(col in new_data_df.columns for col in required_cols):
                            st.success(f"✅ File hợp lệ: {len(new_data_df)} dòng dữ liệu")
                            st.dataframe(new_data_df.head(5))
                           
                            if st.button("🚀 Cập nhật dữ liệu", use_container_width=True):
                                # Merge với dữ liệu hiện có
                                updated_df = pd.concat([sample_df, new_data_df], ignore_index=True)
                                updated_df.to_csv(SAMPLE_PATH, index=False)
                                st.success(f"✅ Đã cập nhật {len(new_data_df)} dòng dữ liệu mới!")
                                st.info("🔄 Vui lòng reload trang để cập nhật model với dữ liệu mới")
                        else:
                            missing = [col for col in required_cols if col not in new_data_df.columns]
                            st.error(f"❌ Thiếu cột: {', '.join(missing)}")
                   
                    except Exception as e:
                        st.error(f"❌ Lỗi khi đọc file: {str(e)}")
           
            # Hiển thị thống kê dữ liệu hiện tại
            st.markdown("---")
            st.subheader("📈 Thống Kê Dữ Liệu Hiện Tại")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số mẫu", len(sample_df))
            with col2:
                st.metric("Số thương hiệu", sample_df['Thương hiệu'].nunique())
            with col3:
                st.metric("Số dòng xe", sample_df['Dòng xe'].nunique())
       
        with tab3:
            st.subheader("Thông Tin Hệ Thống")
            col1, col2 = st.columns(2)
           
            with col1:
                st.metric("Model Status", "✅ Đã load" if model else "❌ Chưa load")
                st.metric("Sample Data Size", f"{len(sample_df):,} records")
                st.metric("Isolation Forest", "✅ Đã load" if iso else "❌ Chưa load")
           
            with col2:
                if LOG_PATH.exists():
                    logs_df = pd.read_csv(LOG_PATH)
                    st.metric("Total Predictions", f"{len(logs_df):,}")
                else:
                    st.metric("Total Predictions", "0")
               
                if PENDING_PATH.exists():
                    pending_df = pd.read_csv(PENDING_PATH)
                    pending_count = len(pending_df[pending_df['status'] == 'pending'])
                    st.metric("Pending Reviews", pending_count)
   
    elif admin_password != "":
        st.error("❌ Mật khẩu không đúng! (Thử lại với password: admin123)")
# ----------------------
# PAGE: LOGS
# ----------------------
elif st.session_state.current_page == "logs":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #435F7A; font-size: 2.2rem;">📋 Nhật Ký Hệ Thống</h2>
        <p style="color: #7f8c8d; font-size: 1.1rem;">Theo dõi lịch sử dự đoán và hoạt động</p>
    </div>
    """, unsafe_allow_html=True)
   
    if LOG_PATH.exists():
        logs_df = pd.read_csv(LOG_PATH)
        st.metric("Tổng số bản ghi", len(logs_df))
       
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            mode_filter = st.selectbox("Lọc theo chế độ", ["Tất cả", "single", "batch"])
        with col2:
            date_sort = st.selectbox("Sắp xếp theo", ["Mới nhất", "Cũ nhất"])
       
        # Apply filters
        filtered_logs = logs_df.copy()
        if mode_filter != "Tất cả":
            filtered_logs = filtered_logs[filtered_logs['mode'] == mode_filter]
       
        if date_sort == "Mới nhất":
            filtered_logs = filtered_logs.sort_values('timestamp', ascending=False)
        else:
            filtered_logs = filtered_logs.sort_values('timestamp', ascending=True)
       
        st.dataframe(filtered_logs.head(100))
       
        # Download button
        csv = filtered_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Export Logs CSV",
            data=csv,
            file_name="system_logs.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("📭 Chưa có logs nào được ghi lại.")
# ----------------------
# PAGE: TEAM INFO
# ----------------------
elif st.session_state.current_page == "team":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #435F7A; font-size: 2.2rem;">👨‍💻 Nhóm Thực Hiện</h2>
        <p style="color: #5a6c7d; font-size: 1.1rem;">Thông tin về nhóm phát triển dự án</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Thêm CSS cho ảnh hình tròn
    st.markdown("""
    <style>
    .circle-image {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0 auto 20px auto;
        display: block;
    }
    .circle-placeholder {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 3rem;
        margin: 0 auto 20px auto;
        border: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .member-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #435F7A;
        margin-bottom: 10px;
        text-align: center;
    }
    .member-container {
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Tạo 2 cột cho 2 thành viên
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="custom-container" style="text-align: center;">
            <h3 style="color: #435F7A; margin-bottom: 20px;">THÀNH VIÊN 1</h3>
        """, unsafe_allow_html=True)
        
        # Hiển thị hình ảnh thành viên 1 dạng hình tròn
        try:
            # Sử dụng st.image trực tiếp với CSS class
            st.markdown('<div class="member-container">', unsafe_allow_html=True)
            st.image("TB.jpg", width=180, use_column_width="auto", output_format="auto")
            st.markdown('<div class="member-name">Nguyen Thai Binh</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            # Fallback nếu không có hình
            st.markdown("""
            <div class="member-container">
                <div class="circle-placeholder" style="background: linear-gradient(135deg, #667eea 0%, #774CA3 100%);">
                    👨‍💻
                </div>
                <div class="member-name">Nguyen Thai Binh</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="text-align: left; padding: 0 20px;">
                <p><strong>📧 Email:</strong> thaibinh782k1@gmail.com</p>
                <p><strong>📚 Vai trò:</strong> Data Scientist & Developer</p>
                <p><strong>🔧 Công việc:</strong> 
                    <br>• Phát triển model ML
                    <br>• Xử lý dữ liệu
                    <br>• Triển khai hệ thống
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-container" style="text-align: center;">
            <h3 style="color: #435F7A; margin-bottom: 20px;">THÀNH VIÊN 2</h3>
        """, unsafe_allow_html=True)
        
        # Hiển thị hình ảnh thành viên 2 dạng hình tròn
        try:
            st.markdown('<div class="member-container">', unsafe_allow_html=True)
            st.image("DT.jpg", width=180, use_column_width="auto", output_format="auto")
            st.markdown('<div class="member-name">Nguyen Duy Thanh</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown("""
            <div class="member-container">
                <div class="circle-placeholder" style="background: linear-gradient(135deg, #00b09b 40%, #96c93d 60%);">
                    👨‍💻
                </div>
                <div class="member-name">Nguyen Duy Thanh</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="text-align: left; padding: 0 20px;">
                <p><strong>📧 Email:</strong> duythanh200620@gmail.com</p>
                <p><strong>📚 Vai trò:</strong> Data Analyst & Developer</p>
                <p><strong>🔧 Công việc:</strong> 
                    <br>• Phân tích dữ liệu
                    <br>• Phát triển giao diện
                    <br>• Testing & Deployment
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # Timeline dự án
    st.markdown("""
    <div class="custom-container">
        <h3 style="color: #435F7A; margin-top: 0;">📅 Timeline Dự Án</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 20px 0;">
            <div style="text-align: center; flex: 1;">
                <div style="background: #667eea; color: white; padding: 10px; border-radius: 50%; width: 50px; height: 50px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                    1
                </div>
                <p style="margin-top: 10px;"><strong>Tuần 1</strong><br>Phân tích yêu cầu<br>Thu thập dữ liệu</p>
            </div>
            <div style="flex: 1; height: 3px; background: #667eea;"></div>
            <div style="text-align: center; flex: 1;">
                <div style="background: #667eea; color: white; padding: 10px; border-radius: 50%; width: 50px; height: 50px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                    2
                </div>
                <p style="margin-top: 10px;"><strong>Tuần 2</strong><br>Xử lý dữ liệu<br>Xây dựng model</p>
            </div>
            <div style="flex: 1; height: 3px; background: #667eea;"></div>
            <div style="text-align: center; flex: 1;">
                <div style="background: #667eea; color: white; padding: 10px; border-radius: 50%; width: 50px; height: 50px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                    3
                </div>
                <p style="margin-top: 10px;"><strong>Tuần 3</strong><br>Phát triển giao diện<br>Testing</p>
            </div>
            <div style="flex: 1; height: 3px; background: #667eea;"></div>
            <div style="text-align: center; flex: 1;">
                <div style="background: #667eea; color: white; padding: 10px; border-radius: 50%; width: 50px; height: 50px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                    4
                </div>
                <p style="margin-top: 10px;"><strong>Tuần 4</strong><br>Deployment<br>Báo cáo</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------
# FOOTER
# ----------------------
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 40px 0 20px 0;">
    <hr style="border-color: #e0e6ed; margin-bottom: 20px;">
    <b>MotorPrice Pro - Hệ thống dự đoán giá xe máy cũ | Phiên bản 1.0 </b><br>
    ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE - MACHINE LEARNING<br>
</div>
""", unsafe_allow_html=True)



















































