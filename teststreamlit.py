# app_motor_price.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import os
import traceback

# ----------------------
# CONFIG (Streamlit Cloud compatible)
# ----------------------
MODEL_PATH = "rf_pipeline.pkl"
ISO_PATH = "isolation_forest.pkl"
SAMPLE_PATH = "sample_data.csv"
FI_CSV = "feature_importances.csv"

PENDING_PATH = Path("pending_listings.csv")
LOG_PATH = Path("prediction_logs.csv")

CURRENT_YEAR = datetime.now().year

st.set_page_config(page_title="Dự đoán giá - Xe máy cũ", layout="wide")

# ----------------------
# Helpers
# ----------------------
@st.cache_resource
def load_models_and_sample(rf_path, iso_path, sample_path):
    """
    Load model, iso, sample. Normalize sample column names so downstream code is stable.
    """
    # load model & iso
    model = joblib.load(rf_path)
    iso = joblib.load(iso_path)

    # load sample
    sample = pd.read_csv(sample_path)

    # Normalize column names (handle variants)
    sample = sample.rename(columns=lambda x: x.strip())

    # unify price column to 'Gia_trieu' numeric (triệu)
    if 'Gia_trieu' not in sample.columns and 'Giá' in sample.columns:
        # assume 'Giá' maybe in million or exact? user used Gia_trieu in training
        # try to coerce to numeric
        sample['Gia_trieu'] = pd.to_numeric(sample['Giá'], errors='coerce')
    else:
        if 'Gia_trieu' in sample.columns:
            sample['Gia_trieu'] = pd.to_numeric(sample['Gia_trieu'], errors='coerce')

    # ensure Khoảng giá min/max are numeric if exist
    for col in ["Khoảng giá min", "Khoảng giá max", "Giá"]:
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors='coerce')

    return model, iso, sample

def ensure_cols_for_upload(df):
    required = [
        "Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe",
        "Năm_đăng_ký","Số_Km_đã_đi","Giá","Khoảng_giá_min","Khoảng_giá_max",
        "Tiêu_đề","Mô_tả_chi_tiết","Địa_chỉ"
    ]
    missing = [c for c in required if c not in df.columns]
    return missing

def add_pending(entry: dict):
    # ensure pending file exists or create
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
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
    else:
        logs = pd.DataFrame()
    logs = pd.concat([pd.DataFrame([record]), logs], ignore_index=True, sort=False)
    logs.to_csv(LOG_PATH, index=False)

def human_currency(x):
    # input x is in the same units as training (likely 'Gia_trieu' = million VND)
    try:
        v = float(x)
        # present nice format: use millions with commas
        return f"{v:,.2f} Triệu"
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
    - actual_price and pred are assumed same unit (Triệu)
    - X_trans_for_iso: 1-D or 2-D vector that already includes residual column appended
    """
    try:
        resid = (actual_price - pred) if (actual_price is not None and not pd.isna(actual_price)) else (0.0 - pred)
    except Exception:
        resid = 0.0 - pred

    # ensure sample brand selection works for different column name variants
    if 'Thương hiệu' in sample_df.columns:
        sample_brand = sample_df[sample_df['Thương hiệu'] == brand].copy()
    elif 'Thương_hiệu' in sample_df.columns:
        sample_brand = sample_df[sample_df['Thương_hiệu'] == brand].copy()
    else:
        sample_brand = pd.DataFrame()

    # resid_z
    if len(sample_brand) >= 10 and 'Gia_trieu' in sample_brand.columns:
        iqr = (sample_brand['Gia_trieu'].quantile(0.75) - sample_brand['Gia_trieu'].quantile(0.25)) or 1.0
        resid_z = abs(resid) / max(iqr, 1e-6)
    else:
        global_std = sample_df['Gia_trieu'].std() if 'Gia_trieu' in sample_df.columns else 1.0
        resid_z = abs(resid) / max(1.0, global_std)

    # min/max
    min_price = sample_brand['Khoảng giá min'].min() if ('Khoảng giá min' in sample_brand.columns and len(sample_brand)>0) else np.nan
    max_price = sample_brand['Khoảng giá max'].max() if ('Khoảng giá max' in sample_brand.columns and len(sample_brand)>0) else np.nan
    violate_minmax = int((not pd.isna(min_price) and (actual_price < min_price)) or (not pd.isna(max_price) and (actual_price > max_price)))

    # p10/p90
    p10 = sample_brand['Gia_trieu'].quantile(0.10) if (len(sample_brand)>0 and 'Gia_trieu' in sample_brand.columns) else np.nan
    p90 = sample_brand['Gia_trieu'].quantile(0.90) if (len(sample_brand)>0 and 'Gia_trieu' in sample_brand.columns) else np.nan
    outside_p10p90 = int((not pd.isna(p10) and actual_price < p10) or (not pd.isna(p90) and actual_price > p90))

    # isolation: X_trans_for_iso must be 1D or 2D array including residual column as last column
    iso_vec = X_trans_for_iso
    if hasattr(iso_vec, "toarray"):
        iso_vec = iso_vec.toarray()
    iso_vec = np.asarray(iso_vec)
    # shape normalize
    if iso_vec.ndim == 1:
        iso_vec = iso_vec.reshape(1, -1)

    try:
        iso_score_raw = - iso.decision_function(iso_vec)[0]
        iso_flag = int(iso.predict(iso_vec)[0] == -1)
    except Exception:
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
# Load models & sample (safe)
# ----------------------
try:
    if not Path(MODEL_PATH).exists() or not Path(ISO_PATH).exists() or not Path(SAMPLE_PATH).exists():
        missing = [p for p in [MODEL_PATH, ISO_PATH, SAMPLE_PATH] if not Path(p).exists()]
        raise FileNotFoundError(f"Missing files: {missing}. Make sure these files are in the same folder as this app.")
    model, iso, sample_df = load_models_and_sample(MODEL_PATH, ISO_PATH, SAMPLE_PATH)
except Exception as e:
    st.error("Không thể load model/sample. Kiểm tra đường dẫn & file có trong repo hay không.")
    st.write(str(e))
    # print traceback to logs for debug (not shown to users)
    st.write(traceback.format_exc())
    st.stop()

# ----------------------
# Sidebar menu (single page app with sidebar)
# ----------------------
st.sidebar.title("Menu")
# show banner only if exists
if Path("xe_may_cu.jpg").exists():
    st.sidebar.image("xe_may_cu.jpg", use_column_width=True)
page = st.sidebar.radio("Chọn mục", ["Business Problem", "Prediction", "Anomaly Check", "Admin Dashboard", "Logs", "Evaluation & Report", "Team Info"])

# ----------------------
# Business Problem (static)
# ----------------------
def render_business_problem():
    st.title("Business Problem")
    st.markdown("""
- **Mục tiêu:** Dự đoán giá bán hợp lý cho xe máy cũ (người mua/ người bán) và phát hiện các tin đăng có giá bất thường.
- **Input:** Thương hiệu, Dòng xe, Năm đăng ký, Số Km, Loại xe, Dung tích, Xuất xứ, (Giá thực - tùy chọn).
- **Output:** Giá dự đoán (Triệu VNĐ) + Anomaly Score (0-100) + Kết luận (Giá thấp bất thường / Giá cao bất thường / Bình thường).
- **Phương pháp:** RandomForest cho regression; IsolationForest + thống kê cho anomaly detection.
    """)
if page == "Business Problem":
    render_business_problem()

# ----------------------
# PREDICTION PAGE (single tab with two options)
# ----------------------
if page == "Prediction":
    st.title(" Dự đoán giá & Kiểm tra bất thường — Xe máy cũ")
    st.markdown("Chọn cách nhập: Nhập tay hoặc Upload file CSV/XLSX (12 cột chuẩn).")

    mode = st.radio("Chọn chế độ", ["Nhập tay", "Upload file (CSV/XLSX)"], index=0)

    if mode == "Nhập tay":
        st.subheader("Nhập chi tiết tin đăng")
        with st.form("form_single", clear_on_submit=False):
            col1, col2 = st.columns([2,1])
            with col1:
                title = st.text_input("Tiêu đề tin đăng", value="Bán SH Mode 125 chính chủ")
                description = st.text_area("Mô tả chi tiết", value="Xe đẹp, bao test, biển số TP, giá có thương lượng.")
                address = st.text_input("Địa chỉ", value="Quận 1, TP. Hồ Chí Minh")
                # safe selectbox with fallback
                brands = sample_df['Thương hiệu'].dropna().unique().tolist() if 'Thương hiệu' in sample_df.columns else []
                brands = sorted(brands) if brands else ['unknown']
                brand = st.selectbox("Thương hiệu", options=brands)
                model_name = st.text_input("Dòng xe", value="")
                loai_values = sample_df['Loại xe'].dropna().unique().tolist() if 'Loại xe' in sample_df.columns else []
                loai = st.selectbox("Loại xe", options=sorted(loai_values) if loai_values else ['unknown'])
            with col2:
                dungtich = st.text_input("Dung tích xe (ví dụ '100 - 175 cc' hoặc '125')", value="125")
                age = st.slider("Tuổi xe (năm)", 0, 50, 3)
                year_reg = int(CURRENT_YEAR - age)
                st.write(f"Năm đăng ký (tương ứng): {year_reg}")
                km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=20000, step=1000)
                price_input = st.number_input("Giá thực (Triệu VNĐ) — nếu muốn (tùy chọn)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
                price_min = st.number_input("Khoảng_giá_min (Triệu) — có thể bỏ trống", min_value=0.0, value=0.0, step=0.1, format="%.2f")
                price_max = st.number_input("Khoảng_giá_max (Triệu) — có thể bỏ trống", min_value=0.0, value=0.0, step=0.1, format="%.2f")

            publish = st.checkbox("Lưu để Admin duyệt (đăng bán)")
            submitted = st.form_submit_button("Predict & Check Anomaly")

        if submitted:
            # Build input df with correct training column names:
            input_df = pd.DataFrame([{
                "Thương hiệu": brand,
                "Dòng xe": model_name if model_name.strip()!="" else "unknown",
                "Năm đăng ký": year_reg,
                "Số Km đã đi": km,
                "Tình trạng": "Đã sử dụng",
                "Loại xe": loai,
                "Dung tích xe": dungtich,
                "Xuất xứ": "unknown"
            }])

            # sanitize types
            input_df["Năm đăng ký"] = pd.to_numeric(input_df["Năm đăng ký"], errors="coerce")
            input_df["Số Km đã đi"] = pd.to_numeric(input_df["Số Km đã đi"], errors="coerce")

            # predict
            try:
                pred = model.predict(input_df)[0]  # pred in same unit as training (Gia_trieu)
            except Exception as e:
                st.error("Lỗi khi dự đoán. Kiểm tra model pipeline.")
                st.write(str(e))
                st.stop()

            # transform features for ISO (and append residual)
            pre = None
            try:
                # try common names for preprocessor step
                if 'pre' in model.named_steps:
                    pre = model.named_steps['pre']
                elif 'preproc' in model.named_steps:
                    pre = model.named_steps['preproc']
                else:
                    # if pipeline saved differently, try first step that is ColumnTransformer
                    for name, step in model.named_steps.items():
                        if hasattr(step, "transform"):
                            pre = step
                            break
            except Exception:
                pre = None

            if pre is None:
                st.error("Không tìm thấy preprocessor trong pipeline. Kiểm tra rf_pipeline.pkl (phải chứa ColumnTransformer tại named_steps['pre']).")
                st.stop()

            X_trans = pre.transform(input_df)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            X_trans = np.asarray(X_trans)  # shape (1, n_features_trans)

            # compute residual for ISO training alignment: ISO expects features + residual (1 column)
            resid_val = (price_input - pred) if price_input > 0 else (0.0 - pred)
            resid_col = np.array(resid_val).reshape(1,1)
            iso_vec = np.hstack([X_trans, resid_col])

            # ensure iso_vec shape matches iso n_features
            try:
                expected = iso.n_features_in_
                if iso_vec.shape[1] != expected:
                    # try using pre.transform then append residual computed in units of training (Gia_trieu)
                    # If mismatch, attempt to warn but continue with best-effort reshape (pad/truncate)
                    st.warning(f"Warning: IsolationForest expects {expected} features but got {iso_vec.shape[1]}. Trying to adjust.")
                    if iso_vec.shape[1] < expected:
                        pad = np.zeros((1, expected - iso_vec.shape[1]))
                        iso_vec = np.hstack([iso_vec, pad])
                    else:
                        iso_vec = iso_vec[:, :expected]
            except Exception:
                # keep going; iso.predict will raise if incompatible
                pass

            # compute iso decision
            try:
                iso_flag = int(iso.predict(iso_vec)[0] == -1)
                iso_score_raw = float(-iso.decision_function(iso_vec)[0])
            except Exception:
                iso_flag = 0
                iso_score_raw = 0.0

            # compute anomaly score using helper (pass full iso_vec (1xN))
            final_score, details = compute_anomaly_score(sample_df=sample_df, brand=brand,
                                                         actual_price=(price_input if price_input>0 else np.nan),
                                                         pred=pred, iso=iso, X_trans_for_iso=iso_vec)

            # determine verdict
            verdict = "Bình thường"
            if final_score >= 50 and (details["resid"] < 0):
                verdict = "Giá thấp bất thường"
            elif final_score >= 50 and (details["resid"] > 0):
                verdict = "Giá cao bất thường"

            # display
            st.markdown("### Kết quả dự đoán")
            st.write(f"**Giá dự đoán:** {human_currency(pred)}")
            st.metric("Anomaly Score (0-100)", f"{final_score:.1f}")
            if verdict != "Bình thường":
                st.error(f" Kết luận: {verdict}")
            else:
                st.success(" Kết luận: Bình thường")

            st.markdown("**Lý do:**")
            reasons = []
            if details["resid_z"] > 1.5:
                reasons.append("- Residual Z cao (khác biệt lớn so với phân khúc).")
            if details["violate_minmax"]:
                reasons.append("- Vi phạm khoảng giá min/max của thương hiệu.")
            if details["outside_p10p90"]:
                reasons.append("- Giá nằm ngoài P10–P90 theo thương hiệu.")
            if details["iso_flag"]:
                reasons.append("- IsolationForest đánh dấu bất thường dựa trên vector đặc trưng + resid.")
            if not reasons:
                reasons.append("- Không phát hiện điểm bất thường rõ rệt.")
            for r in reasons:
                st.write(r)

            # detailed table
            detail_table = pd.DataFrame([{
                "Giá_dự_đoán (Triệu)": pred,
                "Giá_thực (Triệu nếu có)": (price_input if price_input>0 else np.nan),
                "Resid": details["resid"],
                "Resid_z": details["resid_z"],
                "Violate_minmax": details["violate_minmax"],
                "Outside_P10_P90": details["outside_p10p90"],
                "ISO_flag": details["iso_flag"],
                "ISO_score_raw": details["iso_score_raw"],
                "AnomalyScore": final_score
            }])
            st.dataframe(detail_table.T, width=900)

            # save pending if requested
            if publish:
                entry = {
                    "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                    "Tiêu_đề": title,
                    "Mô_tả_chi_tiết": description,
                    "Địa_chỉ": address,
                    "Thương hiệu": brand,
                    "Dòng xe": model_name,
                    "Năm đăng ký": year_reg,
                    "Số Km đã đi": km,
                    "Loại xe": loai,
                    "Dung tích xe": dungtich,
                    "Giá_thực": (price_input if price_input>0 else np.nan),
                    "Giá_dự_đoán": float(pred),
                    "anomaly_score": float(final_score),
                    "iso_flag": int(details["iso_flag"]),
                    "status": "pending",
                    "notes": ""
                }
                pid = add_pending(entry)
                st.success(f"Kết quả đã lưu (id={pid}) và chờ Admin duyệt.")

            # log
            log_record = {
                "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                "mode": "single",
                "title": title,
                "pred": float(pred),
                "price_input": (price_input if price_input>0 else np.nan),
                "anomaly_score": float(final_score),
                "verdict": verdict
            }
            log_prediction(log_record)

    else:
        # Batch upload
        st.subheader("Upload file CSV/XLSX (batch)")
        st.markdown("File cần có các cột: Thương_hiệu, Dòng_xe, Loại_xe, Dung_tích_xe, Năm_đăng_ký, Số_Km_đã_đi, Giá (tùy chọn), Khoảng_giá_min, Khoảng_giá_max, Tiêu_đề, Mô_tả_chi_tiết, Địa_chỉ")
        uploaded = st.file_uploader("Chọn file", type=["csv","xlsx"])
        if uploaded is not None:
            try:
                if str(uploaded.name).lower().endswith(".csv"):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = pd.read_excel(uploaded)
            except Exception as e:
                st.error("Không thể đọc file. Kiểm tra định dạng.")
                st.write(e)
                df_up = None

            if df_up is not None:
                missing = ensure_cols_for_upload(df_up)
                if missing:
                    st.error(f"File thiếu cột bắt buộc: {missing}")
                else:
                    # rename upload columns -> training schema
                    rename_map = {
                        "Thương_hiệu": "Thương hiệu",
                        "Dòng_xe": "Dòng xe",
                        "Loại_xe": "Loại xe",
                        "Dung_tích_xe": "Dung tích xe",
                        "Năm_đăng_ký": "Năm đăng ký",
                        "Số_Km_đã_đi": "Số Km đã đi",
                        "Giá": "Giá_thực",
                        "Khoảng_giá_min": "Khoảng giá min",
                        "Khoảng_giá_max": "Khoảng giá max",
                        "Tiêu_đề": "Tiêu_đề",
                        "Mô_tả_chi_tiết": "Mô_tả_chi_tiết",
                        "Địa_chỉ": "Địa_chỉ"
                    }
                    df_up = df_up.rename(columns=rename_map)

                    # build input for model
                    model_inputs = []
                    for _, row in df_up.iterrows():
                        input_row = {
                            "Thương hiệu": row["Thương hiệu"],
                            "Dòng xe": row["Dòng xe"] if pd.notna(row["Dòng xe"]) else "unknown",
                            "Năm đăng ký": int(row["Năm đăng ký"]) if pd.notna(row["Năm đăng ký"]) else CURRENT_YEAR,
                            "Số Km đã đi": int(row["Số Km đã đi"]) if pd.notna(row["Số Km đã đi"]) else 0,
                            "Tình trạng": row.get("Tình trạng", "Đã sử dụng"),
                            "Loại xe": row["Loại xe"],
                            "Dung tích xe": row["Dung tích xe"],
                            "Xuất xứ": row.get("Xuất xứ", "unknown")
                        }
                        model_inputs.append(input_row)
                    model_X = pd.DataFrame(model_inputs)

                    # predict batch
                    preds = model.predict(model_X)

                    # transform base features
                    pre = None
                    if 'pre' in model.named_steps:
                        pre = model.named_steps['pre']
                    elif 'preproc' in model.named_steps:
                        pre = model.named_steps['preproc']
                    else:
                        for name, step in model.named_steps.items():
                            if hasattr(step, "transform"):
                                pre = step
                                break
                    if pre is None:
                        st.error("Không tìm thấy preprocessor trong pipeline.")
                        st.stop()

                    X_trans = pre.transform(model_X)
                    if hasattr(X_trans, "toarray"):
                        X_trans = X_trans.toarray()
                    X_trans = np.asarray(X_trans)

                    # prepare results
                    results = []
                    for i in range(len(model_X)):
                        # use renamed column Giá_thực
                        actual_price = df_up.loc[i, "Giá_thực"] if "Giá_thực" in df_up.columns else np.nan
                        pred_i = float(preds[i])
                        resid_val = (actual_price - pred_i) if (pd.notna(actual_price) and actual_price>0) else (0.0 - pred_i)
                        iso_vec = np.hstack([X_trans[i].reshape(1,-1), np.array(resid_val).reshape(1,1)])
                        # ensure iso_vec shape matches iso
                        try:
                            expected = iso.n_features_in_
                            if iso_vec.shape[1] != expected:
                                if iso_vec.shape[1] < expected:
                                    iso_vec = np.hstack([iso_vec, np.zeros((1, expected - iso_vec.shape[1]))])
                                else:
                                    iso_vec = iso_vec[:, :expected]
                        except Exception:
                            pass
                        final_score, details = compute_anomaly_score(sample_df=sample_df,
                                                                     brand=model_X.loc[i, "Thương hiệu"],
                                                                     actual_price=(actual_price if pd.notna(actual_price) and actual_price>0 else np.nan),
                                                                     pred=pred_i, iso=iso, X_trans_for_iso=iso_vec)
                        verdict = "Bình thường"
                        if final_score >= 50 and (details["resid"] < 0):
                            verdict = "Giá thấp bất thường"
                        elif final_score >= 50 and (details["resid"] > 0):
                            verdict = "Giá cao bất thường"
                        results.append({
                            "Tiêu_đề": df_up.loc[i, "Tiêu_đề"] if "Tiêu_đề" in df_up.columns else "",
                            "Thương hiệu": model_X.loc[i, "Thương hiệu"],
                            "Dòng xe": model_X.loc[i, "Dòng xe"],
                            "Giá_thực": actual_price if pd.notna(actual_price) else np.nan,
                            "Giá_dự_đoán": pred_i,
                            "Resid": details["resid"],
                            "Resid_z": details["resid_z"],
                            "ISO_flag": details["iso_flag"],
                            "ISO_score_raw": details["iso_score_raw"],
                            "AnomalyScore": final_score,
                            "Verdict": verdict
                        })
                        # log entry
                        log_prediction({
                            "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                            "mode": "batch",
                            "file": uploaded.name,
                            "pred": float(pred_i),
                            "price_input": float(actual_price) if pd.notna(actual_price) else np.nan,
                            "anomaly_score": float(final_score),
                            "verdict": verdict
                        })

                    res_df = pd.DataFrame(results)
                    st.success("Xử lý xong — hiển thị kết quả")
                    st.dataframe(res_df)
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Export kết quả (CSV)", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# ----------------------
# ANOMALY CHECK (single input quick check)
# ----------------------
if page == "Anomaly Check":
    st.title("🔎 Kiểm tra bất thường (nhanh)")
    with st.form("anom_quick"):
        brand = st.text_input("Thương hiệu", value="unknown")
        model_name = st.text_input("Dòng xe", value="unknown")
        age = st.slider("Tuổi xe (năm)", min_value=0, max_value=50, value=3)
        year_registered = int(CURRENT_YEAR - age)
        km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=20000, step=1000)
        loai = st.text_input("Loại xe", value="unknown")
        dungtich = st.text_input("Dung tích xe", value="125")
        xuatxu = st.text_input("Xuất xứ", value="unknown")
        gia_thuc = st.number_input("Giá thực (Triệu VNĐ)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        submitted = st.form_submit_button("Check Anomaly")
    if submitted:
        input_df = pd.DataFrame([{
            "Thương hiệu": brand,
            "Dòng xe": model_name,
            "Năm đăng ký": year_registered,
            "Số Km đã đi": km,
            "Loại xe": loai,
            "Dung tích xe": dungtich,
            "Xuất xứ": xuatxu
        }])
        input_df["Năm đăng ký"] = pd.to_numeric(input_df["Năm đăng ký"], errors="coerce")
        input_df["Số Km đã đi"] = pd.to_numeric(input_df["Số Km đã đi"], errors="coerce")
        pred = model.predict(input_df)[0]
        # transform & append resid
        pre = model.named_steps.get('pre', model.named_steps.get('preproc', None))
        if pre is None:
            for name, step in model.named_steps.items():
                if hasattr(step, "transform"):
                    pre = step
                    break
        X_trans = pre.transform(input_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.asarray(X_trans)
        resid = (gia_thuc - pred) if gia_thuc>0 else (0.0 - pred)
        iso_vec = np.hstack([X_trans, np.array(resid).reshape(1,1)])
        try:
            expected = iso.n_features_in_
            if iso_vec.shape[1] != expected:
                if iso_vec.shape[1] < expected:
                    iso_vec = np.hstack([iso_vec, np.zeros((1, expected - iso_vec.shape[1]))])
                else:
                    iso_vec = iso_vec[:, :expected]
        except Exception:
            pass
        final_score, details = compute_anomaly_score(sample_df=sample_df, brand=brand, actual_price=(gia_thuc if gia_thuc>0 else np.nan), pred=pred, iso=iso, X_trans_for_iso=iso_vec)
        st.metric("Giá dự đoán (Triệu)", f"{pred:.2f}")
        st.metric("Anomaly Score (0-100)", f"{final_score:.1f}")
        if final_score >= 50 and details["resid"] < 0:
            st.error("Kết luận: Giá thấp bất thường")
        elif final_score >= 50 and details["resid"] > 0:
            st.error("Kết luận: Giá cao bất thường")
        else:
            st.success("Kết luận: Bình thường")
        st.write(details)

# ----------------------
# ADMIN DASHBOARD
# ----------------------
if page == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard")
    st.markdown("Duyệt các submissions từ người dùng")
    # show pending
    if PENDING_PATH.exists():
        pending = pd.read_csv(PENDING_PATH)
    else:
        pending = pd.DataFrame(columns=["id","timestamp","Thương hiệu","Dòng xe","Giá_thực","Giá_dự_đoán","anomaly_score","iso_flag","status","notes"])
    st.write(f"Tổng submissions: {len(pending)}")
    st.dataframe(pending.sort_values("timestamp", ascending=False).head(200))
    # operate
    if len(pending)>0:
        pick = st.selectbox("Chọn id để thao tác", options=["(chọn)"] + pending["id"].astype(str).tolist())
        if pick != "(chọn)":
            row = pending[pending["id"].astype(str)==pick].iloc[0]
            st.write(row.to_dict())
            if st.button("Approve"):
                pending.loc[pending["id"]==int(pick),"status"] = "approved"
                pending.to_csv(PENDING_PATH, index=False)
                st.success("Đã approve")
            if st.button("Reject"):
                pending.loc[pending["id"]==int(pick),"status"] = "rejected"
                pending.to_csv(PENDING_PATH, index=False)
                st.warning("Đã reject")
            if st.button("Delete"):
                pending = pending[pending["id"]!=int(pick)]
                pending.to_csv(PENDING_PATH, index=False)
                st.info("Đã xóa")

    st.markdown("---")
    st.subheader("Thông tin model")
    try:
        n_trees = model.named_steps['rf'].n_estimators
    except:
        n_trees = "unknown"
    st.write(f"- RandomForest trees: {n_trees}")
    st.write(f"- Training sample size (app sample): {len(sample_df)}")
    st.write("- Anomaly detector: IsolationForest trained on features + residual")
    if Path(FI_CSV).exists():
        st.dataframe(pd.read_csv(FI_CSV).head(30))
    else:
        st.info("feature_importances.csv not found in repo.")

# ----------------------
# LOGS PAGE
# ----------------------
if page == "Logs":
    st.title(" Logs hoạt động")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        st.write(f"Tổng bản ghi: {len(logs)}")
        st.dataframe(logs.sort_values("timestamp", ascending=False).head(500))
        st.download_button("Export Logs CSV", data=logs.to_csv(index=False).encode('utf-8'), file_name="prediction_logs.csv", mime="text/csv")
    else:
        st.info("Chưa có logs nào")

# ----------------------
# EVALUATION & REPORT
# ----------------------
if page == "Evaluation & Report":
    st.title(" Evaluation & Report")
    st.subheader("Sample data preview")
    st.dataframe(sample_df.head(200))
    st.subheader("Feature importances")
    try:
        if Path(FI_CSV).exists():
            fi = pd.read_csv(FI_CSV)
            st.dataframe(fi.head(50))
            fig, ax = plt.subplots(figsize=(8,4))
            ax.barh(fi['feature'].head(20)[::-1], fi['importance'].head(20)[::-1])
            st.pyplot(fig)
        else:
            st.info("Không tìm thấy file feature_importances.csv")
    except Exception as e:
        st.write("Không thể hiển thị feature importances:", e)

# ----------------------
# TEAM INFO
# ----------------------
if page == "Team Info":
    st.title("Nhóm thực hiện")
    st.markdown("- Họ tên HV: Nguyen Thai Binh")
    st.markdown("- Email: thaibinh782k1@gmail.com")
    st.markdown("- Repo: https://github.com/ThaiBinh78/ML07_Project")
    st.markdown("- Ngày report: 22/11/2025")
