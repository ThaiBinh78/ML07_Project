# app_motor_price.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io

# ----------------------
# CONFIG
# ----------------------
BASE_DIR = Path(__file__).resolve().parent
# If you want to use the uploaded models in /mnt/data, use those paths:
DEFAULT_RF_PATH = BASE_DIR / "rf_pipeline.pkl"
DEFAULT_ISO_PATH = BASE_DIR / "isolation_forest.pkl"
DEFAULT_SAMPLE = BASE_DIR / "sample_data.csv"


PENDING_PATH = BASE_DIR / "pending_listings.csv"

MODEL_PATH = DEFAULT_RF_PATH
ISO_PATH = DEFAULT_ISO_PATH
SAMPLE_PATH = DEFAULT_SAMPLE

PENDING_PATH = BASE_DIR / "pending_listings.csv"
LOG_PATH = BASE_DIR / "prediction_logs.csv"

CURRENT_YEAR = datetime.now().year

st.set_page_config(page_title="Dự đoán giá - Xe máy cũ", layout="wide")

# ----------------------
# Helpers
# ----------------------
@st.cache_resource
def load_models_and_sample(rf_path, iso_path, sample_path):
    model = joblib.load(rf_path)
    iso = joblib.load(iso_path)
    sample = pd.read_csv(sample_path)
    # sanitize sample numeric columns
    for col in ["Gia_trieu", "Giá", "Khoảng giá min", "Khoảng giá max"]:
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")
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
# Sidebar menu (single page app with sidebar)
# ----------------------
st.sidebar.title("Menu")
st.image("xe_may_cu.jpg")
page = st.sidebar.radio("Chọn mục", ["Prediction", "Admin Dashboard", "Logs", "Evaluation & Report", "Team Info"])

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
                brand = st.selectbox("Thương hiệu", options=sorted(sample_df['Thương hiệu'].dropna().unique().tolist()))
                model_name = st.text_input("Dòng xe", value="")
                loai = st.selectbox("Loại xe", options=sorted(sample_df['Loại xe'].dropna().unique().tolist()))
            with col2:
                dungtich = st.text_input("Dung tích xe (ví dụ '100 - 175 cc' hoặc '125')", value="125")
                age = st.slider("Tuổi xe (năm)", 0, 50, 3)
                year_reg = int(CURRENT_YEAR - age)
                st.write(f"Năm đăng ký (tương ứng): {year_reg}")
                km = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=20000, step=1000)
                price_input = st.number_input("Giá thực (VNĐ) — nếu muốn (tùy chọn)", min_value=0, value=0, step=100000)
                price_min = st.number_input("Khoảng_giá_min (VNĐ) — có thể bỏ trống", min_value=0, value=0, step=100000)
                price_max = st.number_input("Khoảng_giá_max (VNĐ) — có thể bỏ trống", min_value=0, value=0, step=100000)

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
            pred = model.predict(input_df)[0]
            # transform features for ISO (and append residual)
            pre = model.named_steps['pre']
            X_trans = pre.transform(input_df)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            X_trans = np.asarray(X_trans)
            # compute residual for ISO training alignment: ISO expects features + residual (1 column)
            resid_val = (price_input - pred) if price_input > 0 else (0 - pred)
            resid_col = np.array(resid_val).reshape(1,1)
            iso_vec = np.hstack([X_trans, resid_col])
            # compute iso decision
            iso_flag = int(iso.predict(iso_vec)[0] == -1)
            iso_score_raw = float(-iso.decision_function(iso_vec)[0])
            # compute anomaly score using helper
            final_score, details = compute_anomaly_score(sample_df=sample_df, brand=brand,
                                                         actual_price=(price_input if price_input>0 else np.nan),
                                                         pred=pred, iso=iso, X_trans_for_iso=iso_vec.flatten())
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
                "Giá_dự_đoán": pred,
                "Giá_thực (nếu có)": (price_input if price_input>0 else np.nan),
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
                    # map upload names to model names:
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
                    out_rows = []
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
                    pre = model.named_steps['pre']
                    X_trans = pre.transform(model_X)
                    if hasattr(X_trans, "toarray"):
                        X_trans = X_trans.toarray()
                    X_trans = np.asarray(X_trans)
                    # prepare results
                    results = []
                    for i in range(len(model_X)):
                        actual_price = df_up.loc[i, "Giá"] if "Giá" in df_up.columns else np.nan
                        pred_i = preds[i]
                        resid_val = (actual_price - pred_i) if (pd.notna(actual_price) and actual_price>0) else (0 - pred_i)
                        iso_vec = np.hstack([X_trans[i].reshape(1,-1), np.array(resid_val).reshape(1,1)])
                        final_score, details = compute_anomaly_score(sample_df=sample_df,
                                                                     brand=model_X.loc[i, "Thương hiệu"],
                                                                     actual_price=(actual_price if pd.notna(actual_price) and actual_price>0 else np.nan),
                                                                     pred=pred_i, iso=iso, X_trans_for_iso=iso_vec.flatten())
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
                        # optionally save pending if needed - we won't auto-save here
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
                    # display table + export
                    st.dataframe(res_df)
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Export kết quả (CSV)", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# ----------------------
# ADMIN DASHBOARD
# ----------------------
elif page == "Admin Dashboard":
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
    st.dataframe(pd.read_csv(BASE_DIR / "feature_importances.csv").head(30))

# ----------------------
# LOGS PAGE
# ----------------------
elif page == "Logs":
    st.title(" Logs hoạt động")
    if Path(LOG_PATH).exists():
        logs = pd.read_csv(LOG_PATH)
        st.write(f"Tổng bản ghi: {len(logs)}")
        st.dataframe(logs.sort_values("timestamp", ascending=False).head(500))
        st.download_button("Export Logs CSV", data=logs.to_csv(index=False).encode('utf-8'), file_name="prediction_logs.csv", mime="text/csv")
    else:
        st.info("Chưa có logs nào")

# ----------------------
# EVALUATION & REPORT
# ----------------------
elif page == "Evaluation & Report":
    st.title(" Evaluation & Report")
    st.subheader("Sample data preview")
    st.dataframe(sample_df.head(200))
    st.subheader("Feature importances")
    try:
        fi = pd.read_csv(BASE_DIR / "feature_importances.csv")
        st.dataframe(fi.head(50))
        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(fi['feature'].head(20)[::-1], fi['importance'].head(20)[::-1])
        st.pyplot(fig)
    except Exception as e:
        st.write("Không tìm thấy file feature_importances.csv", e)

# ----------------------
# TEAM INFO
# ----------------------
else:
    st.title("Nhóm thực hiện")
    st.markdown("- Họ tên HV: Nguyen Thai Binh")
    st.markdown("- Email: thaibinh782k1@gmail.com")
    st.markdown("- Repo: https://github.com/ThaiBinh78/ML07_Project")
    st.markdown("- Ngày report: 22/11/2025")
