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
# CONFIG (use uploaded files in /mnt/data if present)
# ----------------------
# If you uploaded models to the container, these are typical paths:
MODEL_PATH = Path("/mnt/data/rf_pipeline.pkl")
ISO_PATH = Path("/mnt/data/isolation_forest.pkl")
SAMPLE_PATH = Path("/mnt/data/sample_data.csv")
FI_CSV = Path("/mnt/data/feature_importances.csv")

# fallback to repo-local files if /mnt/data doesn't exist
if not MODEL_PATH.exists():
    MODEL_PATH = Path("rf_pipeline.pkl")
if not ISO_PATH.exists():
    ISO_PATH = Path("isolation_forest.pkl")
if not SAMPLE_PATH.exists():
    SAMPLE_PATH = Path("sample_data.csv")
if not FI_CSV.exists():
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
    Load model, iso, sample. Normalize sample column names so downstream code is stable.
    """
    model = joblib.load(str(rf_path))
    iso = joblib.load(str(iso_path))
    sample = pd.read_csv(str(sample_path))
    sample = sample.rename(columns=lambda x: x.strip())
    # unify price column to 'Gia_trieu' numeric (triệu)
    if 'Gia_trieu' not in sample.columns and 'Gia_trieu' not in sample.columns and 'Gia_trieu' not in sample.columns:
        if 'Gia_trieu' in sample.columns:
            sample['Gia_trieu'] = pd.to_numeric(sample['Gia_trieu'], errors='coerce')
    if 'Gia_trieu' not in sample.columns and 'Giá' in sample.columns:
        sample['Gia_trieu'] = pd.to_numeric(sample['Giá'], errors='coerce')
    # coerce min/max if present
    for col in ["Khoảng giá min", "Khoảng giá max", "Giá", "Gia_trieu"]:
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors='coerce')
    return model, iso, sample

def ensure_cols_for_upload(df: pd.DataFrame):
    required = [
        "Thương_hiệu","Dòng_xe","Loại_xe","Dung_tích_xe",
        "Năm_đăng_ký","Số_Km_đã_đi","Giá","Khoảng_giá_min","Khoảng_giá_max",
        "Tiêu_đề","Mô_tả_chi_tiết","Địa_chỉ","Xuất_xứ"
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
    except:
        return x

def compute_anomaly_score_v2(sample_df, brand, actual_price, pred, iso, X_trans_for_iso):
    """
    New, simpler and explainable anomaly score (0-100):
      - Price Gap % component (60%): gap_pct = |actual - pred| / pred * 100
          - If no actual_price (NaN): we don't compute gap component (weight redistributed)
      - P10/P90 component (20%): if actual outside [P10,P90] of brand
      - ISO component (20%): normalized iso_score_raw
    Returns final_score and details (friendly).
    """
    # ensure pred numeric
    pred_val = float(pred)
    # price gap
    has_price = (actual_price is not None) and (not pd.isna(actual_price))
    gap_pct = None
    score_gap = 0.0
    weight_gap = 0.6
    weight_p10p90 = 0.2
    weight_iso = 0.2

    if has_price and pred_val != 0:
        gap_pct = abs(float(actual_price) - pred_val) / abs(pred_val) * 100.0
        # map gap_pct to 0-100 (cap at 200% for safety)
        score_gap = min(100.0, gap_pct)  # 100 means >=100% gap
    else:
        # If no price provided, remove gap component and renormalize weights
        weight_gap = 0.0
        total_remain = weight_p10p90 + weight_iso
        if total_remain > 0:
            weight_p10p90 = weight_p10p90 / total_remain
            weight_iso = weight_iso / total_remain

    # brand distribution
    sample_brand = pd.DataFrame()
    if 'Thương hiệu' in sample_df.columns:
        sample_brand = sample_df[sample_df['Thương hiệu'] == brand].copy()
    if 'Thương_hiệu' in sample_df.columns and sample_brand.empty:
        sample_brand = sample_df[sample_df['Thương_hiệu'] == brand].copy()

    # P10/P90 flag
    p10 = np.nan
    p90 = np.nan
    score_p10p90 = 0.0
    if len(sample_brand) > 0 and 'Gia_trieu' in sample_brand.columns:
        p10 = sample_brand['Gia_trieu'].quantile(0.10)
        p90 = sample_brand['Gia_trieu'].quantile(0.90)
        if has_price and (not pd.isna(p10)) and (not pd.isna(p90)):
            if actual_price < p10:
                # lower tail -> map distance to score (smaller actual -> higher score)
                frac = (p10 - actual_price) / max(1.0, p10)
                score_p10p90 = min(100.0, frac * 100.0)
            elif actual_price > p90:
                frac = (actual_price - p90) / max(1.0, p90)
                score_p10p90 = min(100.0, frac * 100.0)
    else:
        # insufficient brand data -> fallback 0
        score_p10p90 = 0.0

    # ISO: compute raw score (higher -> more anomalous); normalize to 0..100 by heuristic
    iso_score_raw = 0.0
    iso_flag = 0
    try:
        iso_vec = X_trans_for_iso
        if hasattr(iso_vec, "toarray"):
            iso_vec = iso_vec.toarray()
        iso_vec = np.asarray(iso_vec)
        if iso_vec.ndim == 1:
            iso_vec = iso_vec.reshape(1, -1)
        iso_score_raw = - iso.decision_function(iso_vec)[0]
        iso_flag = int(iso.predict(iso_vec)[0] == -1)
    except Exception:
        iso_score_raw = 0.0
        iso_flag = 0
    # normalize iso_score_raw: assume typical raw range ~ [0..1], scale *100 and cap
    score_iso = min(100.0, max(0.0, iso_score_raw * 100.0))

    # final weighted score
    final_score = weight_gap * (score_gap) + weight_p10p90 * (score_p10p90) + weight_iso * (score_iso)

    # Compose friendly explanation
    reasons = []
    if has_price and gap_pct is not None:
        if gap_pct >= 50:
            reasons.append(f"- Giá thực lệch so với dự đoán {gap_pct:.0f}% (lớn).")
        elif gap_pct >= 20:
            reasons.append(f"- Giá thực lệch so với dự đoán {gap_pct:.0f}% (khá đáng chú ý).")
    if score_p10p90 > 0:
        if actual_price < p10:
            reasons.append(f"- Giá thấp hơn P10 của thương hiệu (P10={p10:.2f} Triệu).")
        else:
            reasons.append(f"- Giá cao hơn P90 của thương hiệu (P90={p90:.2f} Triệu).")
    if iso_flag:
        reasons.append(f"- Mẫu tin có đặc điểm lạ so với dữ liệu huấn luyện (IsolationForest).")
    if not reasons:
        reasons.append("- Không phát hiện điểm bất thường rõ rệt dựa trên 3 tiêu chí.")

    details = {
        "has_price": bool(has_price),
        "gap_pct": (gap_pct if gap_pct is not None else np.nan),
        "score_gap": score_gap,
        "score_p10p90": score_p10p90,
        "score_iso": score_iso,
        "iso_flag": int(iso_flag),
        "iso_score_raw": float(iso_score_raw),
        "final_score": float(final_score),
        "explanations": reasons
    }

    return float(final_score), details

# ----------------------
# Load models & sample (safe)
# ----------------------
try:
    missing = [p for p in [MODEL_PATH, ISO_PATH, SAMPLE_PATH] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {[str(x) for x in missing]}. Make sure these files are in the same folder as this app or in /mnt/data.")
    model, iso, sample_df = load_models_and_sample(MODEL_PATH, ISO_PATH, SAMPLE_PATH)
except Exception as e:
    st.error("Không thể load model/sample. Kiểm tra đường dẫn & file có trong repo hay không.")
    st.write(str(e))
    st.write(traceback.format_exc())
    st.stop()

# ----------------------
# Sidebar & Pages
# ----------------------
st.sidebar.title("Menu")
if Path("xe_may_cu.jpg").exists():
    st.sidebar.image("xe_may_cu.jpg", use_column_width=True)
page = st.sidebar.radio("Chọn mục", ["Business Problem", "Prediction", "Anomaly Check", "Admin Dashboard", "Logs", "Evaluation & Report", "Team Info"])

# Business Problem
def render_business_problem():
    st.title("Business Problem")
    st.markdown("""
- **Mục tiêu:** Dự đoán giá bán hợp lý cho xe máy cũ và phát hiện tin đăng có giá bất thường.
- **Input:** Thương hiệu, Dòng xe, Năm đăng ký, Số Km, Loại xe, Dung tích, Xuất xứ, (Giá thực - tùy chọn).
- **Output:** Giá dự đoán (Triệu VNĐ), **Price Risk Score (0-100)**, Kết luận (Giá thấp bất thường / Giá cao bất thường / Bình thường).
- **Phương pháp:** RandomForest (regression) + IsolationForest + thống kê P10/P90.
    """)
if page == "Business Problem":
    render_business_problem()

# Prediction page
if page == "Prediction":
    st.title(" Dự đoán giá & Kiểm tra bất thường — Xe máy cũ")
    st.markdown("Chọn: Nhập tay hoặc Upload file CSV/XLSX (cần cột Xuất_xứ cho quốc gia).")

    mode = st.radio("Chọn chế độ", ["Nhập tay", "Upload file (CSV/XLSX)"], index=0)

    if mode == "Nhập tay":
        st.subheader("Nhập chi tiết tin đăng")
        with st.form("form_single", clear_on_submit=False):
            col1, col2 = st.columns([2,1])
            with col1:
                title = st.text_input("Tiêu đề tin đăng", value="Bán SH Mode 125 chính chủ")
                description = st.text_area("Mô tả chi tiết", value="Xe đẹp, bao test, biển số TP, giá có thương lượng.")
                address = st.text_input("Địa chỉ", value="Quận 1, TP. Hồ Chí Minh")
                brands = sample_df['Thương hiệu'].dropna().unique().tolist() if 'Thương hiệu' in sample_df.columns else []
                brands = sorted(brands) if brands else ['unknown']
                brand = st.selectbox("Thương hiệu", options=brands)
                model_name = st.text_input("Dòng xe", value="")
                loai_values = sample_df['Loại xe'].dropna().unique().tolist() if 'Loại xe' in sample_df.columns else []
                loai = st.selectbox("Loại xe", options=sorted(loai_values) if loai_values else ['unknown'])
            with col2:
                dungtich = st.text_input("Dung tích xe (ví dụ '100 - 175 cc' hoặc '125')", value="125")
                xuatxu = st.text_input("Xuất xứ (Quốc gia)", value="Việt Nam")
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

            try:
                pred = model.predict(input_df)[0]
            except Exception as e:
                st.error("Lỗi khi dự đoán. Kiểm tra model pipeline.")
                st.write(str(e))
                st.stop()

            # find preprocessor
            pre = model.named_steps.get('pre', model.named_steps.get('preproc', None))
            if pre is None:
                for name, step in model.named_steps.items():
                    if hasattr(step, "transform"):
                        pre = step
                        break
            if pre is None:
                st.error("Không tìm thấy preprocessor trong pipeline. Kiểm tra rf_pipeline.pkl.")
                st.stop()

            X_trans = pre.transform(input_df)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            X_trans = np.asarray(X_trans)
            resid_val = (price_input - pred) if price_input > 0 else (0.0 - pred)
            iso_vec = np.hstack([X_trans, np.array(resid_val).reshape(1,1)])
            # ensure iso vec dims
            try:
                expected = iso.n_features_in_
                if iso_vec.shape[1] != expected:
                    st.warning(f"ISO expects {expected} features but got {iso_vec.shape[1]}. Padding/truncating.")
                    if iso_vec.shape[1] < expected:
                        iso_vec = np.hstack([iso_vec, np.zeros((1, expected - iso_vec.shape[1]))])
                    else:
                        iso_vec = iso_vec[:, :expected]
            except Exception:
                pass

            final_score, details = compute_anomaly_score_v2(sample_df=sample_df, brand=brand,
                                                            actual_price=(price_input if price_input>0 else np.nan),
                                                            pred=pred, iso=iso, X_trans_for_iso=iso_vec)

            verdict = "Bình thường"
            if final_score >= 50 and details["gap_pct"] is not None and (float(price_input) < float(pred)):
                verdict = "Giá thấp bất thường"
            elif final_score >= 50 and details["gap_pct"] is not None and (float(price_input) > float(pred)):
                verdict = "Giá cao bất thường"

            # display user-friendly summary
            st.header("KẾT QUẢ TÓM TẮT")
            st.write(f"- **Giá dự đoán:** {human_currency_trieu(pred)}")
            if details["has_price"]:
                st.write(f"- **Giá thực bạn nhập:** {human_currency_trieu(price_input)}")
                st.write(f"- **Price gap:** {details['gap_pct']:.1f}%")
            else:
                st.write("- Bạn **không nhập** giá thực — chỉ hiện giá dự đoán và đánh giá rủi ro thị trường.")
            st.metric("Price Risk Score (0=low → 100=high)", f"{details['final_score']:.1f}")
            if verdict != "Bình thường":
                st.error(f"🔴 Kết luận: {verdict}")
            else:
                st.success("✅ Kết luận: Bình thường")

            st.markdown("**Giải thích chi tiết:**")
            for s in details["explanations"]:
                st.write(s)

            # detail table
            detail_table = pd.DataFrame([{
                "Giá_dự_đoán (Triệu)": pred,
                "Giá_thực (Triệu nếu có)": (price_input if price_input>0 else np.nan),
                "Gap_pct": details["gap_pct"],
                "Score_gap": details["score_gap"],
                "Score_P10P90": details["score_p10p90"],
                "Score_ISO": details["score_iso"],
                "ISO_flag": details["iso_flag"],
                "ISO_score_raw": details["iso_score_raw"],
                "FinalScore": details["final_score"]
            }])
            st.dataframe(detail_table.T, width=900)

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
                    "anomaly_score": float(details["final_score"]),
                    "iso_flag": int(details["iso_flag"]),
                    "status": "pending",
                    "notes": ""
                }
                pid = add_pending(entry)
                st.success(f"Đã lưu id={pid} chờ Admin duyệt.")

            log_prediction({
                "timestamp": datetime.now().isoformat(sep=' ', timespec='seconds'),
                "mode": "single",
                "title": title,
                "pred": float(pred),
                "price_input": (price_input if price_input>0 else np.nan),
                "anomaly_score": float(details["final_score"]),
                "verdict": verdict
            })

    else:
        # Batch upload
        st.subheader("Upload file CSV/XLSX (batch)")
        st.markdown("File cần có cột: Thương_hiệu, Dòng_xe, Loại_xe, Dung_tích_xe, Năm_đăng_ký, Số_Km_đã_đi, Giá (tùy chọn), Khoảng_giá_min, Khoảng_giá_max, Tiêu_đề, Mô_tả_chi_tiết, Địa_chỉ, Xuất_xứ")
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
                        "Địa_chỉ": "Địa_chỉ",
                        "Xuất_xứ": "Xuất xứ"
                    }
                    df_up = df_up.rename(columns=rename_map)
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
                    preds = model.predict(model_X)
                    pre = model.named_steps.get('pre', model.named_steps.get('preproc', None))
                    if pre is None:
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
                        try:
                            expected = iso.n_features_in_
                            if iso_vec.shape[1] != expected:
                                if iso_vec.shape[1] < expected:
                                    iso_vec = np.hstack([iso_vec, np.zeros((1, expected - iso_vec.shape[1]))])
                                else:
                                    iso_vec = iso_vec[:, :expected]
                        except Exception:
                            pass
                        final_score, details = compute_anomaly_score_v2(sample_df=sample_df,
                                                                     brand=model_X.loc[i, "Thương hiệu"],
                                                                     actual_price=(actual_price if pd.notna(actual_price) and actual_price>0 else np.nan),
                                                                     pred=pred_i, iso=iso, X_trans_for_iso=iso_vec)
                        verdict = "Bình thường"
                        if final_score >= 50 and (details["gap_pct"] is not None) and (float(actual_price) < pred_i):
                            verdict = "Giá thấp bất thường"
                        elif final_score >= 50 and (details["gap_pct"] is not None) and (float(actual_price) > pred_i):
                            verdict = "Giá cao bất thường"
                        results.append({
                            "Tiêu_đề": df_up.loc[i, "Tiêu_đề"] if "Tiêu_đề" in df_up.columns else "",
                            "Thương hiệu": model_X.loc[i, "Thương hiệu"],
                            "Dòng xe": model_X.loc[i, "Dòng xe"],
                            "Xuất xứ": model_X.loc[i, "Xuất xứ"],
                            "Giá_thực": actual_price if pd.notna(actual_price) else np.nan,
                            "Giá_dự_đoán": pred_i,
                            "Resid": details["gap_pct"],
                            "ISO_flag": details["iso_flag"],
                            "AnomalyScore": final_score,
                            "Verdict": verdict
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
                    st.download_button("Export kết quả (CSV)", data=res_df.to_csv(index=False).encode('utf-8'), file_name="batch_predictions.csv", mime="text/csv")

# Anomaly check quick page
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
        final_score, details = compute_anomaly_score_v2(sample_df=sample_df, brand=brand,
                                                        actual_price=(gia_thuc if gia_thuc>0 else np.nan),
                                                        pred=pred, iso=iso, X_trans_for_iso=iso_vec)
        st.metric("Giá dự đoán (Triệu)", f"{pred:.2f}")
        st.metric("Price Risk Score (0-100)", f"{final_score:.1f}")
        if final_score >= 50 and details["gap_pct"] is not None and details["gap_pct"]>0 and (gia_thuc < pred):
            st.error("Kết luận: Giá thấp bất thường")
        elif final_score >= 50 and details["gap_pct"] is not None and details["gap_pct"]>0 and (gia_thuc > pred):
            st.error("Kết luận: Giá cao bất thường")
        else:
            st.success("Kết luận: Bình thường")
        st.write("Giải thích:")
        for line in details["explanations"]:
            st.write(line)

# Admin / Logs / Eval pages (unchanged structure)
if page == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard")
    if PENDING_PATH.exists():
        pending = pd.read_csv(PENDING_PATH)
    else:
        pending = pd.DataFrame(columns=["id","timestamp","Thương hiệu","Dòng xe","Giá_thực","Giá_dự_đoán","anomaly_score","iso_flag","status","notes"])
    st.write(f"Tổng submissions: {len(pending)}")
    st.dataframe(pending.sort_values("timestamp", ascending=False).head(200))
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
    try:
        n_trees = model.named_steps['rf'].n_estimators
    except:
        n_trees = "unknown"
    st.write(f"- RandomForest trees: {n_trees}")
    st.write(f"- Training sample size (app sample): {len(sample_df)}")
    st.write("- Anomaly detector: IsolationForest trained on features + residual")
    if FI_CSV.exists():
        st.dataframe(pd.read_csv(FI_CSV).head(30))
    else:
        st.info("feature_importances.csv not found in repo.")

if page == "Logs":
    st.title(" Logs hoạt động")
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        st.write(f"Tổng bản ghi: {len(logs)}")
        st.dataframe(logs.sort_values("timestamp", ascending=False).head(500))
        st.download_button("Export Logs CSV", data=logs.to_csv(index=False).encode('utf-8'), file_name="prediction_logs.csv", mime="text/csv")
    else:
        st.info("Chưa có logs nào")

if page == "Evaluation & Report":
    st.title(" Evaluation & Report")
    st.subheader("Sample data preview")
    st.dataframe(sample_df.head(200))
    st.subheader("Feature importances")
    try:
        if FI_CSV.exists():
            fi = pd.read_csv(FI_CSV)
            st.dataframe(fi.head(50))
            fig, ax = plt.subplots(figsize=(8,4))
            ax.barh(fi['feature'].head(20)[::-1], fi['importance'].head(20)[::-1])
            st.pyplot(fig)
        else:
            st.info("Không tìm thấy file feature_importances.csv")
    except Exception as e:
        st.write("Không thể hiển thị feature importances:", e)

if page == "Team Info":
    st.title("Nhóm thực hiện")
    st.markdown("- Họ tên HV: Nguyen Thai Binh")
    st.markdown("- Email: thaibinh782k1@gmail.com")
    st.markdown("- Repo: https://github.com/ThaiBinh78/ML07_Project")
    st.markdown("- Ngày report: 22/11/2025")
