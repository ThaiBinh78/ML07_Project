# demo_streamlit_app_admin.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------- CONFIG ----------
# Default: assume model + files are in same folder as this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_pipeline.pkl"
ISO_PATH = BASE_DIR / "isolation_forest.pkl"
FI_CSV = BASE_DIR / "feature_importances.csv"
SAMPLE = BASE_DIR / "sample_data.csv"

# File to persist user-submitted predictions for admin review
PENDING_PATH = BASE_DIR / "pending_listings.csv"
PUBLISHED_PATH = BASE_DIR / "published_listings.csv"

CURRENT_YEAR = 2025  # use fixed year as per project requirement / can use datetime.now().year

st.set_page_config(page_title="Dự đoán giá xe - Project (Admin)", layout="wide")

# ---------- Helpers ----------
@st.cache_resource
def load_models_and_data():
    model = joblib.load(MODEL_PATH)
    iso = joblib.load(ISO_PATH)
    fi = pd.read_csv(FI_CSV)
    sample = pd.read_csv(SAMPLE)
    return model, iso, fi, sample

def load_pending():
    if PENDING_PATH.exists():
        return pd.read_csv(PENDING_PATH)
    else:
        cols = ["id","timestamp","Thương hiệu","Dòng xe","Tuổi (năm)","Năm đăng ký",
                "Số Km đã đi","Loại xe","Dung_tich","Xuất xứ","Giá_thực","Giá_dự_đoán",
                "anomaly_score","iso_flag","status","notes"]
        return pd.DataFrame(columns=cols)

def save_pending(df):
    df.to_csv(PENDING_PATH, index=False)

def add_pending(entry: dict):
    df = load_pending()
    # ensure unique id
    entry_id = int(datetime.utcnow().timestamp() * 1000)
    entry["id"] = entry_id
    df = pd.concat([pd.DataFrame([entry]), df], ignore_index=True, sort=False)
    save_pending(df)
    return entry_id

def update_pending_status(entry_id, new_status, notes=""):
    df = load_pending()
    if "id" in df.columns:
        df.loc[df["id"] == entry_id, "status"] = new_status
        df.loc[df["id"] == entry_id, "notes"] = notes
        save_pending(df)
        return True
    return False

def human_currency(x):
    try:
        return f"{int(float(x)):,} ₫"
    except:
        return x

# ---------- Load ----------
# If model files missing, fail with friendly message
try:
    model, iso, fi, sample = load_models_and_data()
except Exception as e:
    st.error(f"Không thể load model / data. Kiểm tra MODEL_PATH/ISO_PATH/FI_CSV/SAMPLE. Chi tiết: {e}")
    st.stop()

# ---------- UI: Sidebar ----------
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Chọn chức năng",
    ["Business Problem", "New Prediction", "Anomaly Check", "Admin Dashboard", "Evaluation & Report", "Team Info"]
)

# ---------- HOME ----------
if menu == "Business Problem":
    st.image("xe_may_cu.jpg")
    st.title(" Ứng dụng Dự đoán giá & Phát hiện bất thường - Xe máy cũ")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        **Mục tiêu:** Hỗ trợ người mua & người bán định giá hợp lý cho xe máy cũ,
        đồng thời phát hiện các tin đăng có giá bất thường để admin rà soát.
        """)
        st.markdown("**Các tính năng:**")
        st.write("- Dự đoán giá bằng Random Forest (offline).")
        st.write("- Phát hiện bất thường kết hợp: residual-z, min/max violation, P10-P90, isolation forest.")
        st.write("- Giao diện người dùng đơn giản; giao diện admin để duyệt & thống kê.")
    with col2:
        st.metric("Kích thước mẫu (mẫu dùng trong app)", len(sample))
        try:
            n_trees = model.named_steps['rf'].n_estimators
            st.metric("Model", f"RandomForest ({n_trees} trees)")
        except:
            st.metric("Model", "RandomForest")

    st.markdown("---")
    st.subheader("Hướng dẫn nhanh")
    st.markdown("1. Người dùng vào `New Prediction` => nhập thông tin => nhận giá đề xuất. Kết quả được lưu chờ admin duyệt nếu người dùng gửi đăng.")
    st.markdown("2. Admin vào `Admin Dashboard` để xem các submissions, duyệt/bỏ, hoặc đánh dấu không đăng.")
    st.markdown("3. Bạn có thể deploy cả app và model lên Streamlit Cloud (đưa các file trong repo).")

# ---------- NEW PREDICTION (USER) ----------
elif menu == "New Prediction":
    st.image("xe_may.jpg")
    st.title(" Dự đoán giá")
    st.markdown("Nhập thông tin xe — app trả về mức giá đề xuất và cảnh báo nếu có dấu hiệu bất thường.")
    st.markdown("Giao diện tối giản, dễ dùng.")

    # form for user
    with st.form("predict_form", clear_on_submit=False):
        c1, c2 = st.columns([2,1])
        with c1:
            brand = st.selectbox("Thương hiệu", options=sorted(list(sample['Thương hiệu'].dropna().unique()) + ["unknown"]))
            model_name = st.text_input("Dòng xe (gõ hoặc để trống)", value="")
            # age slider (tuổi xe)
            age = st.slider("Tuổi xe (năm)", min_value=0, max_value=50, value=3, step=1,
                            help="Chọn số năm kể từ năm đăng ký đến hiện tại")
            year_registered = int(CURRENT_YEAR - age)
            st.write(f"Năm đăng ký tương ứng: **{year_registered}**")
            km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=20000, step=1000)
            loai = st.selectbox("Loại xe", options=sorted(list(sample['Loại xe'].dropna().unique()) + ["unknown"]))
        with c2:
            dungtich = st.number_input("Dung tích (ví dụ: 125)", min_value=50, max_value=2000, value=125)
            xuatxu = st.selectbox("Xuất xứ", options=sorted(list(sample['Xuất xứ'].dropna().unique()) + ["unknown"]))
            # optional price if user wants to propose price for listing
            gia_de_xuat = st.number_input("Nếu bạn muốn đăng: Giá mong muốn (VNĐ)", min_value=0, value=0, step=1000000)
            publish_intent = st.checkbox("Tôi muốn lưu kết quả để Admin duyệt (đăng bán)", value=False)

        submitted = st.form_submit_button("Tính giá & Gửi")

    if submitted:
        input_df = pd.DataFrame([{
            "Thương hiệu": brand,
            "Dòng xe": model_name if model_name.strip()!="" else "unknown",
            "Năm đăng ký": year_registered,
            "Số Km đã đi": km,
            "Loại xe": loai,
            "Dung_tich": dungtich,
            "Xuất xứ": xuatxu
        }])
        # predict
        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")
            pred = None

        # anomaly basic check using isolation
        preproc = model.named_steps['preproc']
        X_trans = preproc.transform(input_df)
        resid = (gia_de_xuat - pred) if gia_de_xuat>0 else 0.0
        iso_vec = np.hstack([X_trans, np.array([[resid]])])
        iso_flag = bool(iso.predict(iso_vec)[0] == -1)
        iso_score_raw = -iso.decision_function(iso_vec)[0]

        # show results
        st.markdown("### Kết quả")
        col_a, col_b = st.columns([1,2])
        with col_a:
            st.metric("Giá đề xuất (VNĐ)", human_currency(pred))
            if iso_flag:
                st.error(" Dấu hiệu bất thường (isolation forest). Vui lòng kiểm tra thông tin.")
            else:
                st.success(" Không phát hiện bất thường (isolation).")
        with col_b:
            st.subheader("So sánh ngắn")
            similar = sample[(sample['Thương hiệu']==brand)]
            if len(similar)>0:
                fig, ax = plt.subplots(figsize=(6,2.5))
                ax.hist(similar['Giá'], bins=25)
                ax.axvline(pred, color='red', linestyle='--', label='Predicted')
                ax.set_xlabel("Giá (VNĐ)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Không có mẫu cùng thương hiệu để so sánh.")

        # Save to pending if user wants to publish
        if publish_intent:
            entry = {
                "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "Thương hiệu": brand,
                "Dòng xe": model_name,
                "Tuổi (năm)": age,
                "Năm đăng ký": year_registered,
                "Số Km đã đi": km,
                "Loại xe": loai,
                "Dung_tich": dungtich,
                "Xuất xứ": xuatxu,
                "Giá_thực": int(gia_de_xuat) if gia_de_xuat>0 else np.nan,
                "Giá_dự_đoán": float(pred),
                "anomaly_score": float(min(100, max(0, iso_score_raw*100))),
                "iso_flag": int(iso_flag),
                "status": "pending",
                "notes": ""
            }
            new_id = add_pending(entry)
            st.success(f"Kết quả đã lưu và gửi cho Admin duyệt (id={new_id}). Admin sẽ xem và quyết định đăng hay không.")

# ---------- ANOMALY CHECK (USER) ----------
elif menu == "Anomaly Check":
    st.title(" Kiểm tra bất thường trước khi đăng")
    st.markdown("Nhập thông tin và Giá thực (bạn muốn đăng) — hệ thống trả về điểm bất thường (0-100) và gợi ý.")
    with st.form("anom_form"):
        brand = st.text_input("Thương hiệu", value="unknown")
        model_name = st.text_input("Dòng xe", value="unknown")
        age = st.slider("Tuổi xe (năm)", min_value=0, max_value=50, value=3)
        year_registered = int(CURRENT_YEAR - age)
        km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=20000, step=1000)
        loai = st.text_input("Loại xe", value="unknown")
        dungtich = st.number_input("Dung tích (ví dụ 125)", min_value=50, max_value=2000, value=125)
        xuatxu = st.text_input("Xuất xứ", value="unknown")
        gia_thuc = st.number_input("Giá thực (VNĐ)", min_value=0, value=20000000, step=1000000)
        submitted = st.form_submit_button("Check Anomaly")
    if submitted:
        input_df = pd.DataFrame([{
            "Thương hiệu": brand,
            "Dòng xe": model_name,
            "Năm đăng ký": year_registered,
            "Số Km đã đi": km,
            "Loại xe": loai,
            "Dung_tich": dungtich,
            "Xuất xứ": xuatxu
        }])
        pred = model.predict(input_df)[0]
        resid = gia_thuc - pred
        # compute simple residual z by brand IQR
        sample_brand = sample[sample['Thương hiệu']==brand]
        if len(sample_brand) >= 10:
            iqr = (sample_brand['Giá'].quantile(0.75) - sample_brand['Giá'].quantile(0.25)) or 1
            resid_z = abs(resid) / iqr
        else:
            resid_z = abs(resid) / max(1, sample['Giá'].std())

        min_price = sample_brand['Khoảng giá min'].min() if 'Khoảng giá min' in sample_brand.columns else np.nan
        max_price = sample_brand['Khoảng giá max'].max() if 'Khoảng giá max' in sample_brand.columns else np.nan
        violate_minmax = int(not np.isnan(min_price) and gia_thuc < min_price) or int(not np.isnan(max_price) and gia_thuc > max_price)
        p10 = sample_brand['Giá'].quantile(0.10) if len(sample_brand)>0 else np.nan
        p90 = sample_brand['Giá'].quantile(0.90) if len(sample_brand)>0 else np.nan
        outside_p10p90 = int((not np.isnan(p10) and gia_thuc < p10) or (not np.isnan(p90) and gia_thuc > p90))

        # isolation
        preproc = model.named_steps['preproc']
        X_trans = preproc.transform(input_df)
        iso_vec = np.hstack([X_trans, np.array([[resid]])])
        iso_pred = iso.predict(iso_vec)[0]
        iso_score_raw = -iso.decision_function(iso_vec)[0]
        iso_flag = int(iso_pred == -1)

        # combine
        w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
        score1 = min(1, resid_z/3) * 100
        score2 = violate_minmax * 100
        score3 = outside_p10p90 * 100
        score4 = min(1, iso_score_raw/0.5) * 100
        final_score = w1*score1 + w2*score2 + w3*score3 + w4*score4

        st.metric("Anomaly Score (0-100)", f"{final_score:.1f}")
        st.write("Chi tiết:")
        st.write({
            "Giá dự đoán": float(pred),
            "Resid (Giá_thực - Giá_dự_đoán)": float(resid),
            "resid_z": float(resid_z),
            "violate_minmax": int(violate_minmax),
            "outside_p10p90": int(outside_p10p90),
            "iso_flag": int(iso_flag),
            "iso_score_raw": float(iso_score_raw)
        })
        if final_score >= 50:
            st.error(" Hệ thống đánh dấu bất thường. Cân nhắc điều chỉnh giá hoặc gửi admin xác nhận.")
        else:
            st.success(" Không phát hiện bất thường nghiêm trọng.")

# ---------- ADMIN DASHBOARD ----------
elif menu == "Admin Dashboard":
    st.title(" Admin Dashboard — Duyệt các listing")
    st.markdown("Quản lý các submission từ người dùng, duyệt/không duyệt, xem thống kê model & dataset.")

    # Stats
    pending_df = load_pending()
    total_sub = len(pending_df)
    pending_count = len(pending_df[pending_df["status"]=="pending"]) if "status" in pending_df.columns else total_sub
    approved_count = len(pending_df[pending_df["status"]=="approved"]) if "status" in pending_df.columns else 0
    rejected_count = len(pending_df[pending_df["status"]=="rejected"]) if "status" in pending_df.columns else 0
    anomaly_count = len(pending_df[pending_df["iso_flag"]==1]) if "iso_flag" in pending_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng submissions", total_sub)
    c2.metric("Đang chờ", pending_count)
    c3.metric("Đã approve", approved_count)
    c4.metric("Flagged (iso)", anomaly_count)

    st.markdown("---")
    st.subheader("Danh sách chờ duyệt (mới nhất lên trên)")
    if total_sub == 0:
        st.info("Chưa có submission nào.")
    else:
        # show table with buttons to act on each row (select a row id to operate)
        st.dataframe(pending_df.sort_values("timestamp", ascending=False).head(200))

        st.markdown("**Duyệt thủ công**")
        # choose an id to operate
        ids = pending_df["id"].astype(str).tolist()
        pick = st.selectbox("Chọn submission id", options=["(chọn)"] + ids)
        if pick != "(chọn)":
            row = pending_df[pending_df["id"].astype(str)==pick].iloc[0]
            st.markdown("### Chi tiết submission")
            st.write(row.to_dict())
            colx, coly = st.columns(2)
            with colx:
                if st.button(" Approve (Cho đăng)"):
                    update_pending_status(int(pick), "approved", notes="Approved by admin")
                    st.success("Đã approve.")
                if st.button(" Reject (Không đăng)"):
                    update_pending_status(int(pick), "rejected", notes="Rejected by admin")
                    st.warning("Đã reject.")
            with coly:
                if st.button(" View similar in sample"):
                    # show similar samples
                    similar = sample[sample['Thương hiệu']==row['Thương hiệu']]
                    st.dataframe(similar.head(50))
                if st.button(" Delete entry"):
                    df = load_pending()
                    df = df[df["id"] != int(pick)]
                    save_pending(df)
                    st.info("Deleted.")

    st.markdown("---")
    st.subheader("Thông tin model & thuật toán")
    try:
        n_trees = model.named_steps['rf'].n_estimators
    except:
        n_trees = "unknown"
    st.markdown(f"- Model: RandomForest (trees = {n_trees})")
    st.markdown(f"- Training sample size (app sample): {len(sample)}")
    st.markdown("- Anomaly detector: IsolationForest (trained on transformed features + residual)")
    st.markdown("**Top feature importances:**")
    st.dataframe(fi.head(20))
    fig, ax = plt.subplots(figsize=(8,3))
    top20 = fi.head(20)
    ax.barh(top20['feature'][::-1], top20['importance'][::-1])
    ax.set_title("Top 20 feature importances")
    st.pyplot(fig)

# ---------- EVALUATION & REPORT ----------
elif menu == "Evaluation & Report":
    st.title(" Evaluation & Report")
    st.subheader("Feature importances")
    st.dataframe(fi.head(50))
    st.subheader("Sample data preview")
    st.dataframe(sample.head(200))

# ---------- TEAM INFO ----------
elif menu == "Team Info":
    st.title("Nhóm thực hiện")
    st.markdown("- Họ tên HV: Nguyen Thai Binh")
    st.markdown("- Email: thaibinh782k1@gmail.com")
    st.markdown("- File project và link deploy: https://github.com/ThaiBinh78/ML07_Project")
    st.markdown("- Ngày report: 22/11/2025")
