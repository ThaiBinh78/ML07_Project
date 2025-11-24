# app_motor_price.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import traceback

# ----------------------
# CONFIG
# ----------------------
MODEL_PATH = Path("rf_pipeline.pkl")
ISO_PATH = Path("isolation_forest.pkl")
SAMPLE_PATH = Path("sample_data.csv")
FI_CSV = Path("feature_importances.csv")

PENDING_PATH = Path("pending_listings.csv")
LOG_PATH = Path("prediction_logs.csv")

CURRENT_YEAR = datetime.now().year

st.set_page_config(page_title="Dự đoán giá - Xe máy cũ", layout="wide")


# ----------------------
# Helpers
# ----------------------
@st.cache_resource
def load_models_and_sample(rf_path: Path, iso_path: Path, sample_path: Path):
    """
    Load pipeline, isolation forest, sample dataset. Normalize sample columns for robust downstream usage.
    """
    model = joblib.load(rf_path)
    iso = joblib.load(iso_path)
    sample = pd.read_csv(sample_path)
    sample = sample.rename(columns=lambda x: x.strip())
    # unify price column to 'Gia_trieu' if possible
    if 'Gia_trieu' not in sample.columns and 'Giá' in sample.columns:
        sample['Gia_trieu'] = pd.to_numeric(sample['Giá'], errors='coerce')
    elif 'Gia_trieu' in sample.columns:
        sample['Gia_trieu'] = pd.to_numeric(sample['Gia_trieu'], errors='coerce')
    for col in ["Khoảng giá min", "Khoảng giá max", "Giá"]:
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors='coerce')
    return model, iso, sample

def ensure_cols_for_upload(df: pd.DataFrame):
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
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
    else:
        logs = pd.DataFrame()
    logs = pd.concat([pd.DataFrame([record]), logs], ignore_index=True, sort=False)
    logs.to_csv(LOG_PATH, index=False)

def human_currency_trieu(x):
    try:
        v = float(x)
        return f"{v:,.2f} Triệu"
    except Exception:
        return x

def compute_anomaly_score(sample_df, brand, actual_price, pred, iso, X_trans_for_iso):
    """
    Returns numeric final_score (kept for internal use) and details dict.
    X_trans_for_iso: array or flattened vector that already includes residual as last column OR includes features only;
                     function will accept both (if no residual present, assumes residual computed outside and appended).
    """
    try:
        resid = (actual_price - pred) if (actual_price is not None and not pd.isna(actual_price)) else (0.0 - pred)
    except Exception:
        resid = 0.0 - pred

    # try find brand column
    if 'Thương hiệu' in sample_df.columns:
        sample_brand = sample_df[sample_df['Thương hiệu'] == brand].copy()
    elif 'Thương_hiệu' in sample_df.columns:
        sample_brand = sample_df[sample_df['Thương_hiệu'] == brand].copy()
    else:
        sample_brand = pd.DataFrame()

    # residual z: prefer brand IQR, else global std
    if len(sample_brand) >= 10 and 'Gia_trieu' in sample_brand.columns:
        iqr = (sample_brand['Gia_trieu'].quantile(0.75) - sample_brand['Gia_trieu'].quantile(0.25)) or 1.0
        resid_z = abs(resid) / max(iqr, 1e-6)
    else:
        global_std = sample_df['Gia_trieu'].std() if 'Gia_trieu' in sample_df.columns else 1.0
        resid_z = abs(resid) / max(1.0, global_std)

    # min/max check (brand-level)
    min_price = sample_brand['Khoảng giá min'].min() if ('Khoảng giá min' in sample_brand.columns and len(sample_brand)>0) else np.nan
    max_price = sample_brand['Khoảng giá max'].max() if ('Khoảng giá max' in sample_brand.columns and len(sample_brand)>0) else np.nan
    violate_minmax = int((not pd.isna(min_price) and (actual_price < min_price)) or (not pd.isna(max_price) and (actual_price > max_price)))

    # p10/p90
    p10 = sample_brand['Gia_trieu'].quantile(0.10) if (len(sample_brand)>0 and 'Gia_trieu' in sample_brand.columns) else np.nan
    p90 = sample_brand['Gia_trieu'].quantile(0.90) if (len(sample_brand)>0 and 'Gia_trieu' in sample_brand.columns) else np.nan
    outside_p10p90 = int((not pd.isna(p10) and actual_price < p10) or (not pd.isna(p90) and actual_price > p90))

    # iso: ensure iso_vec is 2D and shape matches iso.n_features_in_
    iso_vec = X_trans_for_iso
    if hasattr(iso_vec, "toarray"):
        iso_vec = iso_vec.toarray()
    iso_vec = np.asarray(iso_vec)
    if iso_vec.ndim == 1:
        iso_vec = iso_vec.reshape(1, -1)

    try:
        expected = iso.n_features_in_
        if iso_vec.shape[1] != expected:
            # pad with zeros or truncate
            if iso_vec.shape[1] < expected:
                pad = np.zeros((iso_vec.shape[0], expected - iso_vec.shape[1]))
                iso_vec = np.hstack([iso_vec, pad])
            else:
                iso_vec = iso_vec[:, :expected]
        iso_score_raw = - float(iso.decision_function(iso_vec)[0])
        iso_flag = int(iso.predict(iso_vec)[0] == -1)
    except Exception:
        iso_score_raw = 0.0
        iso_flag = 0

    # combine into final numeric score (kept for logging/back-end)
    w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
    score1 = min(1.0, resid_z / 3.0) * 100.0
    score2 = violate_minmax * 100.0
    score3 = outside_p10p90 * 100.0
    score4 = min(1.0, iso_score_raw / 0.5) * 100.0
    final_score = float(w1*score1 + w2*score2 + w3*score3 + w4*score4)

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
    missing = [str(p) for p in [MODEL_PATH, ISO_PATH, SAMPLE_PATH] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}. Make sure these files are in the same folder as this app.")
    model, iso, sample_df = load_models_and_sample(MODEL_PATH, ISO_PATH, SAMPLE_PATH)
except Exception as e:
    st.error("Không thể load model/sample. Kiểm tra đường dẫn & file có trong repo hay không.")
    st.write(str(e))
    st.write(traceback.format_exc())
    st.stop()

# ----------------------
# Sidebar & Menu
# ----------------------
st.sidebar.title("Menu")
if Path("xe_may_cu.jpg").exists():
    st.sidebar.image("xe_may_cu.jpg", use_column_width=True)
page = st.sidebar.radio("Chọn mục", ["Bài toán nghiệp vụ ", "Dự đoán giá", "Kiểm tra bất thường", "Chế độ quản trị viên", "Nhật ký hệ thống", "Đánh giá & Báo cáo kết quả", "Thông tin nhóm thực hiện"])

# ----------------------
# Bài toán nghiệp vụ 
# ----------------------
def render_business_problem():
    st.title("Bài toán nghiệp vụ ")
    st.markdown("""
- **Mục tiêu:** Dự đoán giá bán hợp lý cho xe máy cũ và phát hiện tin đăng giá bất thường.
- **Input:** Thương hiệu, Dòng xe, Năm đăng ký, Số Km, Loại xe, Dung tích, Xuất xứ, (Giá thực - tùy chọn).
- **Output:** Giá dự đoán (Triệu VNĐ) + Kết luận bằng lời (dạng tư vấn, dễ hiểu).
- **Phương pháp:** RandomForest cho dự đoán; IsolationForest + thống kê cho phát hiện bất thường.
    """)
if page == "Bài toán nghiệp vụ ":
    render_business_problem()

# ----------------------
# Prediction page
# ----------------------
if page == "Dự đoán giá":
    st.title("Dự đoán giá & Kiểm tra bất thường — Xe máy cũ")
    st.markdown("Chọn chế độ nhập: Nhập tay hoặc Upload file CSV/XLSX (cột chuẩn).")

    mode = st.radio("Chọn chế độ", ["Nhập tay", "Upload file (CSV/XLSX)"], index=0)

    if mode == "Nhập tay":
        st.subheader("Nhập chi tiết tin đăng")
        with st.form("form_single", clear_on_submit=False):
            c1, c2 = st.columns([2,1])
            with c1:
                title = st.text_input("Tiêu đề tin đăng", value="Bán SH Mode 125 chính chủ")
                description = st.text_area("Mô tả chi tiết", value="Xe đẹp, bao test, biển số TP, giá có thương lượng.")
                address = st.text_input("Địa chỉ", value="Quận 1, TP. Hồ Chí Minh")
                brands = sample_df['Thương hiệu'].dropna().unique().tolist() if 'Thương hiệu' in sample_df.columns else ['unknown']
                brand = st.selectbox("Thương hiệu", options=sorted(brands))
                model_name = st.text_input("Dòng xe", value="")
                loai_values = sample_df['Loại xe'].dropna().unique().tolist() if 'Loại xe' in sample_df.columns else ['unknown']
                loai = st.selectbox("Loại xe", options=sorted(loai_values))
            with c2:
                dungtich = st.text_input("Dung tích xe (ví dụ '100 - 175 cc' hoặc '125')", value="125")
                xuatxu = st.text_input("Xuất xứ", value="unknown")
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
            input_df = pd.DataFrame([{
                "Thương hiệu": brand,
                "Dòng xe": model_name if model_name.strip()!="" else "unknown",
                "Năm đăng ký": year_reg,
                "Số Km đã đi": km,
                "Tình trạng": "Đã sử dụng",
                "Loại xe": loai,
                "Dung tích xe": dungtich,
                "Xuất xứ": xuatxu
            }])
            input_df["Năm đăng ký"] = pd.to_numeric(input_df["Năm đăng ký"], errors="coerce")
            input_df["Số Km đã đi"] = pd.to_numeric(input_df["Số Km đã đi"], errors="coerce")

            # predict
            try:
                pred = float(model.predict(input_df)[0])
            except Exception as e:
                st.error("Lỗi khi dự đoán — kiểm tra pipeline model.")
                st.write(str(e))
                st.stop()

            # find preprocessor inside pipeline
            pre = None
            try:
                if 'pre' in model.named_steps:
                    pre = model.named_steps['pre']
                elif 'preproc' in model.named_steps:
                    pre = model.named_steps['preproc']
                else:
                    for name, step in model.named_steps.items():
                        if hasattr(step, "transform"):
                            pre = step
                            break
            except Exception:
                pre = None

            if pre is None:
                st.error("Không tìm thấy preprocessor trong pipeline. Kiểm tra rf_pipeline.pkl")
                st.stop()

            X_trans = pre.transform(input_df)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            X_trans = np.asarray(X_trans)  # shape (1, n_features)

            # residual (in same units as training, Gia_trieu)
            resid_val = (price_input - pred) if price_input > 0 else (0.0 - pred)
            iso_vec = np.hstack([X_trans, np.array(resid_val).reshape(1,1)])

            # adjust iso_vec to expected features
            try:
                expected = iso.n_features_in_
                if iso_vec.shape[1] != expected:
                    if iso_vec.shape[1] < expected:
                        iso_vec = np.hstack([iso_vec, np.zeros((1, expected - iso_vec.shape[1]))])
                    else:
                        iso_vec = iso_vec[:, :expected]
            except Exception:
                pass

            # compute anomaly numeric details (kept for logs)
            final_score, details = compute_anomaly_score(sample_df=sample_df, brand=brand,
                                                         actual_price=(price_input if price_input>0 else np.nan),
                                                         pred=pred, iso=iso, X_trans_for_iso=iso_vec)

            # determine verdict and user-friendly explanation (C-style: tư vấn + rủi ro)
            verdict = "Bình thường"
            if final_score >= 50 and (details["resid"] < 0):
                verdict = "Giá thấp bất thường"
            elif final_score >= 50 and (details["resid"] > 0):
                verdict = "Giá cao bất thường"

            # Build human-friendly explanation (per your choice C)
            if verdict == "Bình thường":
                explanation = ("Giá bạn nhập hiện nằm trong vùng an toàn cho dòng xe này. "
                               "Người mua và người bán có thể thương lượng thêm — mức giá này ít khả năng là lừa đảo.")
            else:
                if verdict == "Giá thấp bất thường":
                    explanation = ("Giá này thấp hơn thông thường. Nếu bạn là người bán, kiểm tra: biển số tỉnh, "
                                   "xe có sửa chữa/đã thay máy, odo bất thường, hoặc bạn nhập nhầm đơn vị (ngàn vs triệu). "
                                   "Nếu bạn là người mua, hãy cẩn trọng: yêu cầu xem trực tiếp và giấy tờ.")
                else:
                    explanation = ("Giá này cao bất thường so với thị trường. Kiểm tra: thông tin xe có đầy đủ không, "
                                   "người bán có bằng chứng chính chủ hay lịch sử sửa chữa rõ ràng hay không.")

            # Provide detailed bullet suggestions when abnormal (6–8 items)
            suggestions = []
            if verdict != "Bình thường":
                suggestions = [
                    "Kiểm tra lại: bạn đã nhập đúng đơn vị (Triệu) chưa (đôi khi nhập nhầm ngàn).",
                    "Kiểm tra biển số (tỉnh/thành) so với địa chỉ người bán.",
                    "Yêu cầu hình ảnh/giấy tờ chi tiết: chính chủ, đăng kiểm, hóa đơn sửa chữa (nếu có).",
                    "Kiểm tra odo / số km — odo cao khiến giá thấp là hợp lý.",
                    "Xem lịch sử thay thế lớn (thay máy, thay khung) — điều này ảnh hưởng lớn đến giá.",
                    "Người mua nên hẹn xem trực tiếp, thử xe; người bán nên thêm mô tả chi tiết & hình ảnh rõ ràng."
                ]
            else:
                suggestions = [
                    "Giữ mô tả rõ ràng (đời xe, km, chính chủ) để tăng tin cậy.",
                    "Đánh giá kỹ trước khi thương lượng — giá đang ở vùng an toàn."
                ]

            # Display to user (no numeric anomaly score shown)
            st.markdown("### Kết quả dự đoán")
            st.write(f"**Giá dự đoán:** {human_currency_trieu(pred)}")
            st.markdown(f"**Kết luận:** **{verdict}**")
            st.markdown("**Giải thích (dễ hiểu):**")
            st.write(explanation)
            st.markdown("**Lý do chi tiết:**")
            # include human-readable reasons derived from details
            reasons = []
            if details["resid_z"] > 1.5:
                reasons.append("Giá chênh lớn so với phân khúc (residual cao).")
            if details["violate_minmax"]:
                reasons.append("Giá nằm ngoài khoảng giá min/max của thương hiệu.")
            if details["outside_p10p90"]:
                reasons.append("Giá nằm ngoài vùng P10–P90 (khác biệt so với 90% mẫu).")
            if details["iso_flag"]:
                reasons.append("Mẫu có đặc điểm khác biệt (một mô hình phát hiện bất thường đánh dấu).")
            if not reasons:
                reasons.append("Không phát hiện điểm bất thường rõ rệt trong dữ liệu mẫu.")
            for r in reasons:
                st.write("- " + r)

            st.markdown("**Gợi ý / Hướng xử lý**")
            for s in suggestions:
                st.write("- " + s)

            # Detailed table (for power users)
            detail_table = pd.DataFrame([{
                "Giá_dự_đoán (Triệu)": pred,
                "Giá_thực nhập (Triệu nếu có)": (price_input if price_input>0 else np.nan),
                "Resid": details["resid"],
                "Resid_z": details["resid_z"],
                "Violate_minmax": details["violate_minmax"],
                "Outside_P10_P90": details["outside_p10p90"],
                "ISO_flag": details["iso_flag"],
                "ISO_score_raw": details["iso_score_raw"],
                "AnomalyScore_internal": final_score
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
                    "Xuất xứ": xuatxu,
                    "Giá_thực": (price_input if price_input>0 else np.nan),
                    "Giá_dự_đoán": float(pred),
                    "anomaly_score": float(final_score),
                    "iso_flag": int(details["iso_flag"]),
                    "status": "pending",
                    "notes": ""
                }
                pid = add_pending(entry)
                st.success(f"Kết quả đã lưu (id={pid}) và chờ Admin duyệt.")

            # log (keeps numeric score for admin / analysis)
            log_prediction({
                "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                "mode": "single",
                "title": title,
                "pred": float(pred),
                "price_input": (price_input if price_input>0 else np.nan),
                "anomaly_score": float(final_score),
                "verdict": verdict
            })

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
                    # build inputs
                    model_inputs = []
                    for _, row in df_up.iterrows():
                        model_inputs.append({
                            "Thương hiệu": row["Thương hiệu"],
                            "Dòng xe": row["Dòng xe"] if pd.notna(row["Dòng xe"]) else "unknown",
                            "Năm đăng ký": int(row["Năm đăng ký"]) if pd.notna(row["Năm đăng ký"]) else CURRENT_YEAR,
                            "Số Km đã đi": int(row["Số Km đã đi"]) if pd.notna(row["Số Km đã đi"]) else 0,
                            "Tình trạng": row.get("Tình trạng", "Đã sử dụng"),
                            "Loại xe": row["Loại xe"],
                            "Dung tích xe": row["Dung tích xe"],
                            "Xuất xứ": row.get("Xuất xứ", "unknown")
                        })
                    model_X = pd.DataFrame(model_inputs)
                    preds = model.predict(model_X)
                    # find preprocessor
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

                    results = []
                    for i in range(len(model_X)):
                        actual_price = df_up.loc[i, "Giá_thực"] if "Giá_thực" in df_up.columns else np.nan
                        pred_i = float(preds[i])
                        resid_val = (actual_price - pred_i) if (pd.notna(actual_price) and actual_price>0) else (0.0 - pred_i)
                        iso_vec = np.hstack([X_trans[i].reshape(1,-1), np.array(resid_val).reshape(1,1)])
                        # ensure iso_vec size matches
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

                        # Generate human explanation text (short)
                        if verdict == "Bình thường":
                            explanation = "Giá nằm trong vùng an toàn của mẫu."
                        elif verdict == "Giá thấp bất thường":
                            explanation = "Giá thấp hơn đa số mẫu; hãy kiểm tra kỹ giấy tờ và tình trạng xe."
                        else:
                            explanation = "Giá cao hơn đa số mẫu; kiểm tra mã tin và giấy tờ."

                        results.append({
                            "Tiêu_đề": df_up.loc[i, "Tiêu_đề"] if "Tiêu_đề" in df_up.columns else "",
                            "Thương hiệu": model_X.loc[i, "Thương hiệu"],
                            "Dòng xe": model_X.loc[i, "Dòng xe"],
                            "Giá_thực": actual_price if pd.notna(actual_price) else np.nan,
                            "Giá_dự_đoán": pred_i,
                            "Verdict": verdict,
                            "Explanation": explanation,
                            "AnomalyScore_internal": final_score
                        })
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
# Anomaly Check (quick)
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
        pred = float(model.predict(input_df)[0])
        # transform & append resid
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
        # human-friendly
        if final_score >= 50 and details["resid"] < 0:
            verdict = "Giá thấp bất thường"
            explanation = ("Giá thấp hơn bình thường — người mua nên cẩn trọng; người bán hãy kiểm tra lại thông tin.")
        elif final_score >= 50 and details["resid"] > 0:
            verdict = "Giá cao bất thường"
            explanation = ("Giá cao hơn bình thường — kiểm tra tính xác thực hồ sơ và giấy tờ.")
        else:
            verdict = "Bình thường"
            explanation = ("Giá nằm trong vùng an toàn; thường có thể đăng bán hoặc thương lượng.")
        st.metric("Giá dự đoán (Triệu)", f"{pred:.2f}")
        st.write("Kết luận:", verdict)
        st.write("Giải thích:", explanation)
        st.write("Chi tiết kỹ thuật (dành cho admin/ analyst):")
        st.json(details)

# ----------------------
# Admin Dashboard (Approve / Reject only)
# ----------------------
if page == "Chế độ quản trị viên":
    st.title(" Chế độ quản trị viên")
    st.markdown("Duyệt các submissions từ người dùng")
    if PENDING_PATH.exists():
        pending = pd.read_csv(PENDING_PATH)
    else:
        pending = pd.DataFrame(columns=["id","timestamp","Thương hiệu","Dòng xe","Giá_thực","Giá_dự_đoán","anomaly_score","iso_flag","status","notes"])
    st.write(f"Tổng submissions: {len(pending)}")
    st.dataframe(pending.sort_values("timestamp", ascending=False).head(200))
    if len(pending) > 0:
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
    st.markdown("---")
    st.subheader("Thông tin model")
    try:
        n_trees = model.named_steps['rf'].n_estimators
    except Exception:
        n_trees = "unknown"
    st.write(f"- RandomForest trees: {n_trees}")
    st.write(f"- Training sample size (app sample): {len(sample_df)}")
    st.write("- Anomaly detector: IsolationForest trained on features + residual")
    if FI_CSV.exists():
        st.dataframe(pd.read_csv(FI_CSV).head(30))
    else:
        st.info("feature_importances.csv not found in repo.")

# ----------------------
# Logs
# ----------------------
if page == "Nhật ký hệ thống":
    st.title("Nhật ký hệ thống hoạt động")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        st.write(f"Tổng bản ghi: {len(logs)}")
        st.dataframe(logs.sort_values("timestamp", ascending=False).head(500))
        st.download_button("Export Logs CSV", data=logs.to_csv(index=False).encode('utf-8'), file_name="prediction_logs.csv", mime="text/csv")
    else:
        st.info("Chưa có logs nào")

# ----------------------
# Evaluation & Report (6 plots, professional, minimal)
# ----------------------
if page == "Đánh giá & Báo cáo kết quả":
    st.title("Đánh giá & Báo cáo kết quả")
    st.subheader("Sample data preview")
    st.dataframe(sample_df.head(200))

    # Prepare data safe names
    price_col = 'Gia_trieu' if 'Gia_trieu' in sample_df.columns else ('Giá' if 'Giá' in sample_df.columns else None)
    if price_col is None:
        st.error("Sample data không có cột giá (Gia_trieu / Giá).")
    else:
        df = sample_df.copy()
        df = df.dropna(subset=[price_col])
        # 1. Histogram (distribution)
        st.markdown("### Phân bố giá tổng thể")
        fig1, ax1 = plt.subplots(figsize=(8,3))
        ax1.hist(df[price_col], bins=40)
        ax1.set_xlabel("Giá (Triệu)")
        ax1.set_ylabel("Số tin")
        st.pyplot(fig1)

        # 2. Boxplot by brand (top 12 brands)
        st.markdown("### Phân bố giá theo thương hiệu (boxplot các top brands)")
        top_brands = df['Thương hiệu'].value_counts().head(12).index.tolist() if 'Thương hiệu' in df.columns else []
        if top_brands:
            fig2, ax2 = plt.subplots(figsize=(10,4))
            data_to_plot = [df[df['Thương hiệu'] == b][price_col].dropna() for b in top_brands]
            ax2.boxplot(data_to_plot, vert=False, labels=top_brands)
            ax2.set_xlabel("Giá (Triệu)")
            st.pyplot(fig2)
        else:
            st.info("Không đủ dữ liệu Thương hiệu để vẽ boxplot.")

        # 3. Scatter Km vs Price with trendline
        if 'Số Km đã đi' in df.columns:
            st.markdown("### Mối tương quan: Số Km vs Giá")
            x = pd.to_numeric(df['Số Km đã đi'], errors='coerce')
            y = pd.to_numeric(df[price_col], errors='coerce')
            mask = (~x.isna()) & (~y.isna())
            if mask.sum() > 10:
                x1 = x[mask]
                y1 = y[mask]
                fig3, ax3 = plt.subplots(figsize=(8,4))
                ax3.scatter(x1, y1, alpha=0.4, s=10)
                # trendline (polyfit)
                m, b = np.polyfit(x1, y1, 1)
                xs = np.linspace(x1.min(), x1.max(), 100)
                ax3.plot(xs, m*xs + b, linewidth=2)
                ax3.set_xlabel("Số Km đã đi")
                ax3.set_ylabel("Giá (Triệu)")
                st.pyplot(fig3)
            else:
                st.info("Không đủ dữ liệu Km để vẽ biểu đồ tương quan.")

        # 4. Feature importances (group-level)
        st.markdown("### Feature importances (top features)")
        if FI_CSV.exists():
            fi = pd.read_csv(FI_CSV)
            top = fi.head(20)
            fig4, ax4 = plt.subplots(figsize=(8,4))
            ax4.barh(top['feature'][::-1], top['importance'][::-1])
            ax4.set_xlabel("Importance")
            st.pyplot(fig4)
        else:
            st.info("Không tìm thấy feature_importances.csv")

        # 5. Heatmap of numeric correlations
        st.markdown("### Heatmap tương quan các biến numeric")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().fillna(0)
            fig5, ax5 = plt.subplots(figsize=(8,6))
            im = ax5.matshow(corr, aspect='auto')
            ax5.set_xticks(range(len(numeric_cols)))
            ax5.set_yticks(range(len(numeric_cols)))
            ax5.set_xticklabels(numeric_cols, rotation=90)
            ax5.set_yticklabels(numeric_cols)
            fig5.colorbar(im, ax=ax5)
            st.pyplot(fig5)
        else:
            st.info("Không đủ biến numeric để vẽ heatmap.")

        # 6. Anomaly score distribution (internal)
        st.markdown("### Phân bố Anomaly Score (internal, cho admin)")
        if LOG_PATH.exists():
            logs = pd.read_csv(LOG_PATH)
            if 'anomaly_score' in logs.columns:
                fig6, ax6 = plt.subplots(figsize=(8,3))
                ax6.hist(logs['anomaly_score'].dropna(), bins=30)
                ax6.set_xlabel("Anomaly Score (internal)")
                st.pyplot(fig6)
            else:
                st.info("Chưa có trường anomaly_score trong logs.")
        else:
            st.info("Chưa có logs để vẽ phân bố anomaly score.")

# ----------------------
# Team Info
# ----------------------
if page == "Thông tin nhóm thực hiện":
    st.title("Nhóm thực hiện")
    st.markdown("- Họ tên HV: Nguyen Thai Binh")
    st.markdown("- Email: thaibinh782k1@gmail.com")
    st.markdown("- Repo: https://github.com/ThaiBinh78/ML07_Project")
    st.markdown("- Ngày report: 22/11/2025")

