# ==============================================================================
# BIPV 통합 관제 시스템 (Streamlit Cloud / 로컬 실행용)
# streamlit run 260212_a.py → 링크만 열면 대시보드 표시
# ==============================================================================
__version__ = "5.28"

import os
import sys

import pandas as pd
import numpy as np
import requests
import urllib.parse
import streamlit as st
import plotly.graph_objects as go
import pvlib
from pvlib.location import Location
from datetime import datetime, timedelta

# XGBoost: 모델 파일 있으면 사용, 없으면 규칙 기반
try:
    import joblib
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# ==============================================================================
# 1. 기본 환경 설정 (상수)
# ==============================================================================
KMA_SERVICE_KEY = "c6ffb5b520437f3e6983a55234e73701fce509cbb3153c9473ebbe5756a1da00"

LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
NX, NY = 60, 127
DEFAULT_CAPACITY = 300
DEFAULT_EFFICIENCY = 18.7
DEFAULT_LOSS = 0.85
DEFAULT_KEPCO = 210
DEFAULT_UNIT_COUNT = 1

# v5.26: 루버 1개 가로·세로 기준, 개수 2배 → 면적 2배. 10개 합쳐 1.6 m² (300W 18.7% 기준)
DEFAULT_LOUVER_COUNT = 10
DEFAULT_WIDTH_MM = 1000.0   # 루버 1개 가로 1 m (mm)
DEFAULT_HEIGHT_MM = 160.0   # 루버 1개 세로 0.16 m → 10개 시 1.6 m²
# 파이프라인용 각도 상한 (고정)
ANGLE_CAP_DEG_DEFAULT = 90.0

# XGBoost 모델 파일 경로 (스크립트와 같은 폴더 또는 현재 디렉터리)
XGB_MODEL_FILENAME = "bipv_xgboost_model.pkl"
# 학습 시 사용한 feature 순서/이름 (모델에 맞게 수정 가능)
XGB_FEATURE_NAMES = ["hour", "month", "zenith", "azimuth", "ghi", "dni", "dhi", "cloud_cover"]


@st.cache_resource
def load_xgb_model():
    """XGBoost 모델 로드. 실패 시 None 반환."""
    if not _XGB_AVAILABLE:
        return None
    for base in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
        path = os.path.join(base, XGB_MODEL_FILENAME)
        if os.path.isfile(path):
            try:
                return joblib.load(path)
            except Exception:
                return None
    return None


def predict_angles_xgb(model, times, zenith_arr, azimuth_arr, ghi_real, dni_arr, dhi_arr, cloud_series, angle_cap_deg):
    """XGBoost로 시간별 각도 예측. 반환: (angles,) 길이 24, ghi<10 구간은 0°."""
    n = len(times)
    month = times.month.values
    hour = times.hour.values
    X = pd.DataFrame({
        "hour": hour,
        "month": month,
        "zenith": zenith_arr,
        "azimuth": azimuth_arr,
        "ghi": ghi_real,
        "dni": dni_arr,
        "dhi": dhi_arr,
        "cloud_cover": cloud_series,
    }, columns=XGB_FEATURE_NAMES)
    # 학습 시 사용한 컬럼 순서 맞추기
    if hasattr(model, "feature_names_in_"):
        cols = [c for c in model.feature_names_in_ if c in X.columns]
        if cols:
            X = X[cols]
    try:
        pred = model.predict(X)
    except Exception:
        return None
    pred = np.asarray(pred).ravel()
    pred = np.clip(pred, 0, min(90, angle_cap_deg))
    pred[ghi_real < 10] = 0
    return pred.astype(float)


def poa_with_iam(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi,
                 a_r=0.16, diffuse_iam_mode="none"):
    """입사각 손실(IAM) 적용 POA. 직달만 IAM, 산란은 미적용(default)."""
    irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solar_zenith, solar_azimuth=solar_azimuth
    )
    poa_direct = np.nan_to_num(irrad["poa_direct"], nan=0.0)
    poa_diffuse = np.nan_to_num(irrad["poa_diffuse"], nan=0.0)
    aoi = pvlib.irradiance.aoi(
        surface_tilt, surface_azimuth,
        solar_zenith, solar_azimuth
    )
    aoi = np.clip(np.asarray(aoi, dtype=float), 0, 90)
    try:
        iam = pvlib.iam.martin_ruiz(aoi, a_r=a_r)
    except AttributeError:
        iam = pvlib.irradiance.iam.martin_ruiz(aoi, a_r=a_r)
    if diffuse_iam_mode == "none":
        poa_eff = poa_direct * iam + poa_diffuse
    else:
        poa_eff = poa_direct * iam + poa_diffuse * np.clip(iam, 0.3, 1.0)
    return poa_eff


def brute_force_angles_per_hour(zenith, azimuth, dni, ghi, dhi, angle_cap_deg, ghi_real,
                                 step_deg=1):
    """브루트포스: 매 시간마다 0° vs 5° vs ... vs 60°를 싸움 붙여 이기는 각도를 선택.
    IAM 적용으로 비스듬히 받을 때 손실 반영 → 아침 9시엔 0~15°, 정오엔 60°, 오후 4시엔 다시 닫기.
    결과: AI 제어가 고정형(0°/60°)보다 발전량이 높게 나옴. ghi_real < 10 이면 0도(닫힘).
    반환: (angles,) 길이 24.
    """
    n = len(zenith)
    zenith = np.asarray(zenith)
    azimuth = np.asarray(azimuth)
    dni = np.asarray(dni)
    ghi = np.asarray(ghi)
    dhi = np.asarray(dhi)
    ghi_real = np.asarray(ghi_real)
    angles_out = np.zeros(n)
    candidates = np.arange(0, min(90, angle_cap_deg) + 1e-9, step_deg)
    candidates = np.unique(np.round(candidates).astype(int))
    for i in range(n):
        if ghi_real[i] < 10:
            angles_out[i] = 0
            continue
        best_power = -1
        best_angle = 0
        z_i, az_i = zenith[i], azimuth[i]
        d_i, g_i, dh_i = dni[i], ghi[i], dhi[i]
        for a in candidates:
            tilt = 90 - a
            poa_eff = poa_with_iam(tilt, 180, z_i, az_i, d_i, g_i, dh_i)
            power_i = float(np.asarray(poa_eff).ravel()[0])
            if power_i > best_power:
                best_power = power_i
                best_angle = a
        angles_out[i] = best_angle
    return angles_out


def solar_tracking_angles(zenith, ghi_real, angle_cap_deg=60):
    """v5.16 스타일: 태양 추적 각도. (90°−천정각)으로 시간대별로 자연스럽게 각도 변화.
    일사량이 낮은 구간(< 10 W/m²)은 0°(닫힘). 상한 angle_cap_deg(기본 90°).
    (브루트포스는 매시 최대 발전 각도만 골라서 정오 전후로 '거의 최대각'만 나오는 경향이 있음.)
    반환: (angles,) 길이 n, float.
    """
    z = np.asarray(zenith, dtype=float).ravel()
    g = np.asarray(ghi_real, dtype=float).ravel()
    max_angle = min(90, angle_cap_deg)
    angles = np.clip(90 - z, 0, max_angle)
    angles[g < 10] = 0
    return angles


site = Location(LAT, LON, tz=TZ)

# ==============================================================================
# 2. 기상 데이터 가져오기
# ==============================================================================
def get_kma_forecast():
    decoded_key = urllib.parse.unquote(KMA_SERVICE_KEY)
    base_date = datetime.now().strftime("%Y%m%d")
    now_hour = datetime.now().hour
    
    available_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    base_time_int = max([h for h in available_hours if h <= now_hour] or [23])
    if base_time_int == 23 and now_hour < 2:
        base_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    base_time = f"{base_time_int:02d}00"

    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {"serviceKey": decoded_key, "numOfRows": "1000", "dataType": "JSON",
              "base_date": base_date, "base_time": base_time, "nx": NX, "ny": NY}

    try:
        res = requests.get(url, params=params).json()
        items = res["response"]["body"]["items"]["item"]
        df = pd.DataFrame(items)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
        df_tom = df[df["fcstDate"] == tomorrow]
        df_tom = df_tom.drop_duplicates(subset=['fcstDate', 'fcstTime', 'category'])
        return df_tom.pivot(index="fcstTime", columns="category", values="fcstValue"), tomorrow
    except Exception as e:
        print(f"■ 기상청 데이터 오류: {e}")
        return None, None


# ==============================================================================
# 3. Streamlit 대시보드 (링크만 열면 실행)
# ==============================================================================
def _poa_with_iam_app(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi, a_r=0.16):
    """대시보드 내 IAM 적용 POA."""
    irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        dni=dni, ghi=ghi, dhi=dhi,
        solar_zenith=solar_zenith, solar_azimuth=solar_azimuth
    )
    poa_direct = np.nan_to_num(irrad["poa_direct"], nan=0.0)
    poa_diffuse = np.nan_to_num(irrad["poa_diffuse"], nan=0.0)
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    aoi = np.clip(np.asarray(aoi, dtype=float), 0, 90)
    try:
        iam = pvlib.iam.martin_ruiz(aoi, a_r=a_r)
    except AttributeError:
        iam = pvlib.irradiance.iam.martin_ruiz(aoi, a_r=a_r)
    return poa_direct * iam + poa_diffuse


def run_app():
    st.set_page_config(page_title="BIPV Dashboard", layout="wide")

    # 기상청 예보 (기본: 내일)
    kma, tomorrow = get_kma_forecast()
    if kma is None:
        st.error("기상청 예보를 불러올 수 없습니다. 잠시 후 새로고침 해 주세요.")
        return

    default_sunshine_hours = 10
    tomorrow_dt = datetime.strptime(tomorrow, "%Y%m%d")

    # 사이드바
    st.sidebar.title("■ 통합 환경 설정")
    st.sidebar.subheader("1. 시간 및 날짜")
    sim_date = st.sidebar.date_input("시뮬레이션 날짜", tomorrow_dt)
    sunshine_hours = default_sunshine_hours

    st.sidebar.subheader("2. 설치 면적 (루버 1개 가로×세로, 개수)")
    st.sidebar.caption("루버 1개 가로 1m·세로 0.16m, 10개 시 1.6 m². 개수 2배면 면적 2배.")
    width_mm = st.sidebar.number_input("루버 1개 가로 (mm)", min_value=100.0, value=float(DEFAULT_WIDTH_MM), step=100.0)
    height_mm = st.sidebar.number_input("루버 1개 세로 (mm)", min_value=100.0, value=float(DEFAULT_HEIGHT_MM), step=100.0)
    louver_count = st.sidebar.number_input("루버 개수 (개)", min_value=1, value=DEFAULT_LOUVER_COUNT, step=1)
    ref_area = DEFAULT_WIDTH_MM * DEFAULT_HEIGHT_MM * DEFAULT_LOUVER_COUNT if DEFAULT_LOUVER_COUNT > 0 else 1.0
    user_area = width_mm * height_mm * louver_count if louver_count > 0 else ref_area
    area_scale = user_area / ref_area if ref_area > 0 else 1.0

    st.sidebar.subheader("3. 패널 스펙")
    unit_count = st.sidebar.number_input("설치 유닛 수 (개)", min_value=1, value=DEFAULT_UNIT_COUNT)
    capacity_w = st.sidebar.number_input("패널 용량 (W)", value=DEFAULT_CAPACITY)
    target_eff = st.sidebar.number_input("패널 효율 (%)", value=DEFAULT_EFFICIENCY, step=0.1)
    kepco_rate = st.sidebar.number_input("전기 요금 (원/kWh)", value=DEFAULT_KEPCO)

    _sim_d = sim_date.strftime("%Y-%m-%d") if hasattr(sim_date, "strftime") else str(sim_date)
    times = pd.date_range(start=f"{_sim_d} 00:00", periods=24, freq="h", tz=TZ)
    solpos = site.get_solarposition(times)
    clearsky = site.get_clearsky(times)
    zenith_arr = np.asarray(solpos["apparent_zenith"].values, dtype=float)
    azimuth_arr = np.asarray(solpos["azimuth"].values, dtype=float)

    # 날짜가 내일이면 기상청 구름 반영, 아니면 클리어스카이만
    if _sim_d.replace("-", "") == tomorrow:
        kma_reindex = kma.reindex(times.strftime("%H00"))
        cloud_series = kma_reindex["SKY"].apply(lambda x: 0.0 if x == "1" else (0.5 if x == "3" else 1.0)).astype(float).values
    else:
        cloud_series = np.zeros(24)
    ghi_real = np.asarray(clearsky["ghi"].values, dtype=float) * (1.0 - (cloud_series * 0.65))
    _dni = pvlib.irradiance.dirint(ghi_real, solpos["apparent_zenith"], times)
    dni_arr = np.asarray(_dni.fillna(0).values, dtype=float).ravel()
    dhi_arr = (ghi_real - dni_arr * np.cos(np.radians(zenith_arr))).clip(0).astype(float).ravel()

    solar_noon_idx = int(np.argmin(zenith_arr))
    half = int(sunshine_hours) // 2
    op_start = max(0, min(solar_noon_idx - half, 7))
    op_end = min(23, solar_noon_idx + half + (int(sunshine_hours) % 2))
    op_hours = (op_start, op_end)

    eff_factor = float(target_eff) / DEFAULT_EFFICIENCY
    angle_cap_deg = ANGLE_CAP_DEG_DEFAULT
    default_loss = DEFAULT_LOSS
    _angle_cap = min(90, angle_cap_deg)

    # 각도: XGBoost 모델 있으면 사용, 없거나 실패 시 규칙 기반(90°−천정각)
    xgb_model = load_xgb_model() if _XGB_AVAILABLE else None
    xgb_angles = None
    if xgb_model is not None:
        xgb_angles = predict_angles_xgb(
            xgb_model, times, zenith_arr, azimuth_arr, ghi_real, dni_arr, dhi_arr, cloud_series, angle_cap_deg
        )
    if xgb_angles is not None:
        current_ai_angles = xgb_angles
        angle_mode = "XGBoost"
    else:
        current_ai_angles = np.where(ghi_real < 10, 0, np.clip(90 - zenith_arr, 0, _angle_cap).astype(float))
        angle_mode = "규칙 기반"

    def calc_power(angles_list):
        tilt = 90 - np.array(angles_list, dtype=float)
        poa_eff = _poa_with_iam_app(tilt, np.full_like(tilt, 180), zenith_arr, azimuth_arr, dni_arr, ghi_real, dhi_arr)
        mask = (times.hour >= op_hours[0]) & (times.hour <= op_hours[1])
        base_wh = (poa_eff[mask] / 1000 * capacity_w * unit_count * eff_factor * default_loss).sum()
        return base_wh * area_scale

    pow_ai = calc_power(current_ai_angles)
    pow_fix_0 = calc_power([0] * 24)
    rev_ai = (pow_ai / 1000) * kepco_rate

    weather_status = "맑음" if np.mean(cloud_series) < 0.3 else ("구름많음" if np.mean(cloud_series) < 0.8 else "흐림")

    # 메인 영역
    st.title("■ BIPV 통합 관제 대시보드 v5.28")
    st.markdown(f"**날짜:** {_sim_d} | **날씨:** {weather_status} | **각도:** {angle_mode}")

    c1, c2 = st.columns(2)
    c1.metric("AI 제어", f"{int(rev_ai):,} 원  {pow_ai/1000:.2f} kWh", "당일", help="태양 추적 각도로 고정형 대비 우위")
    c2.metric("고정 0°", f"{int((pow_fix_0/1000)*kepco_rate):,} 원  {pow_fix_0/1000:.2f} kWh", "당일")

    # 연간 (클리어스카이)
    last_year = int(sim_date.strftime("%Y")) - 1 if hasattr(sim_date, "strftime") else (datetime.now().year - 1)
    times_y = pd.date_range(start=f"{last_year}-01-01 00:00", end=f"{last_year}-12-31 23:00", freq="h", tz=TZ)
    solpos_y = site.get_solarposition(times_y)
    clearsky_y = site.get_clearsky(times_y)
    ghi_y = np.asarray(clearsky_y["ghi"].values, dtype=float)
    dni_y = pvlib.irradiance.dirint(ghi_y, solpos_y["apparent_zenith"], times_y).fillna(0).values
    dhi_y = (ghi_y - dni_y * np.cos(np.radians(solpos_y["apparent_zenith"].values))).clip(0)
    zen_y = solpos_y["apparent_zenith"].values
    az_y = solpos_y["azimuth"].values
    angles_ai_y = None
    if xgb_model is not None:
        cloud_y = np.zeros(len(times_y))
        angles_ai_y = predict_angles_xgb(xgb_model, times_y, zen_y, az_y, ghi_y, dni_y, dhi_y, cloud_y, angle_cap_deg)
    if angles_ai_y is None:
        angles_ai_y = np.where(ghi_y < 10, 0, np.clip(90 - zen_y, 0, _angle_cap).astype(float))

    def energy_wh_year(angles):
        tilt = 90 - np.asarray(angles, dtype=float)
        poa_eff = _poa_with_iam_app(tilt, np.full_like(tilt, 180), zen_y, az_y, dni_y, ghi_y, dhi_y)
        return (poa_eff / 1000 * capacity_w * unit_count * eff_factor * default_loss).sum() * area_scale

    ann_wh_ai = energy_wh_year(angles_ai_y)
    ann_wh_fix0 = energy_wh_year(np.zeros_like(angles_ai_y))
    ann_kwh_ai = ann_wh_ai / 1000
    ann_kwh_fix0 = ann_wh_fix0 / 1000
    ann_rev_ai = ann_kwh_ai * kepco_rate
    ann_rev_fix0 = ann_kwh_fix0 * kepco_rate
    c3, c4 = st.columns(2)
    c3.metric("연간 AI 제어", f"{int(ann_rev_ai):,} 원  {ann_kwh_ai:.1f} kWh", f"{last_year}년 1년 클리어스카이")
    c4.metric("연간 고정 0°", f"{int(ann_rev_fix0):,} 원  {ann_kwh_fix0:.1f} kWh", f"{last_year}년 1년 클리어스카이")

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("■ 제어 스케줄")
        mask_plot = (times.hour >= 6) & (times.hour <= 19)
        x_plot = times[mask_plot].strftime("%H:%M")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_plot, y=ghi_real[mask_plot],
            name="일사량 (W/m²)", marker_color="orange", opacity=0.3, yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=x_plot, y=current_ai_angles[mask_plot],
            name="AI 각도", line=dict(color="blue", width=4), yaxis="y2"
        ))
        fig.update_layout(
            yaxis=dict(title="일사량 (W/m²)", showgrid=False),
            yaxis2=dict(title="각도 (°)", overlaying="y", side="right", range=[0, 90], showgrid=True),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)
        df_schedule = pd.DataFrame({
            "시간": times[mask_plot].strftime("%H:%M").tolist(),
            "AI 각도(°)": current_ai_angles[mask_plot].astype(int).tolist(),
            "일사량 (W/m²)": np.round(ghi_real[mask_plot], 1).tolist(),
        })
        st.dataframe(df_schedule, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("■ 발전량 비교")
        comp_data = [
            {"Mode": "AI 제어", "Val": pow_ai, "Color": "#1a73e8", "Opacity": 1.0},
            {"Mode": "고정(0°)", "Val": pow_fix_0, "Color": "gray", "Opacity": 0.5},
        ]
        df_comp = pd.DataFrame(comp_data)
        fig_bar = go.Figure(data=[go.Bar(
            x=df_comp["Mode"], y=df_comp["Val"],
            marker_color=df_comp["Color"], marker_opacity=df_comp["Opacity"],
            text=[f"{v:.0f}Wh" for v in df_comp["Val"]], textposition="auto"
        )])
        fig_bar.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_bar, use_container_width=True)


# Streamlit이 이 파일을 실행할 때 대시보드 표시
run_app()
