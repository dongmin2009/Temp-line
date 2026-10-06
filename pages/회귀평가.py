import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="서울 연평균기온 회귀평가",
    page_icon="🌡️",
    layout="wide",
)

CSV_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
CUTOFF_YEAR = 2025
MIN_DAYS = 300
BASE_YEAR = 1908


@st.cache_data
def load_yearly_data():
    df = pd.read_csv(CSV_URL, encoding="utf-8")

    required = {"날짜", "지점", "평균기온", "최저기온", "최고기온"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {', '.join(sorted(missing))}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= CUTOFF_YEAR].copy()

    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 1년 관측일수가 300일 이상인 연도만 사용
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].copy()
    yearly["지난연수"] = yearly["연도"] - BASE_YEAR

    return yearly.sort_values("연도").reset_index(drop=True)


def fit_linear_regression(data):
    x = data["지난연수"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    if len(data) < 2:
        raise ValueError("회귀분석을 하려면 최소 2개의 연도가 필요합니다.")

    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def predict(slope, intercept, years):
    years = np.asarray(years, dtype=float)
    x = years - BASE_YEAR
    return slope * x + intercept


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return float(mae), float(mse), float(r2)


def make_result(name, requested_start, requested_end, train, test):
    slope, intercept = fit_linear_regression(train)
    pred = predict(slope, intercept, test["연도"])

    mae, mse, r2 = evaluate(test["연평균기온"], pred)

    return {
        "모델": name,
        "요청 학습기간": f"{requested_start}–{requested_end}",
        "실제 학습 시작": int(train["연도"].min()),
        "실제 학습 종료": int(train["연도"].max()),
        "학습 연도 수": len(train),
        "기울기 (°C/년)": slope,
        "절편": intercept,
        "MAE (°C)": mae,
        "MSE (°C²)": mse,
        "R²": r2,
        "slope": slope,
        "intercept": intercept,
    }


st.title("🌡️ 서울 연평균기온 선형회귀 및 예측 평가")
st.caption(
    "서울 기상자료를 이용해 연평균기온의 선형회귀를 수행하고 "
    "50년·100년 학습모델의 2006–2025년 예측 성능을 비교합니다."
)

try:
    yearly = load_yearly_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

if yearly.empty:
    st.error("조건을 만족하는 연도 데이터가 없습니다.")
    st.stop()

# ---------------------------------------------------------------------
# 1. 데이터 요약
# ---------------------------------------------------------------------
st.header("1. 분석 데이터")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("사용 연도 수", f"{len(yearly)}년")

with col2:
    st.metric("시작 연도", f"{yearly['연도'].min()}년")

with col3:
    st.metric("종료 연도", f"{yearly['연도'].max()}년")

with col4:
    st.metric("관측일수 기준", f"{MIN_DAYS}일 이상")

st.info(
    f"2025년까지의 자료 중 연간 관측일수가 {MIN_DAYS}일 이상인 연도만 사용했습니다. "
    f"회귀분석의 독립변수는 기준연도 {BASE_YEAR}년으로부터의 경과연수입니다."
)

# ---------------------------------------------------------------------
# 2. 전체 데이터 회귀분석
# ---------------------------------------------------------------------
st.header("2. 전체 데이터 선형회귀")

overall_slope, overall_intercept = fit_linear_regression(yearly)
overall_pred = predict(overall_slope, overall_intercept, yearly["연도"])

overall_mae, overall_mse, overall_r2 = evaluate(
    yearly["연평균기온"], overall_pred
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("기울기", f"{overall_slope:.4f} °C/년")

with c2:
    st.metric("연간 변화량", f"{overall_slope * 10:.3f} °C/10년")

with c3:
    st.metric("MAE", f"{overall_mae:.3f} °C")

with c4:
    st.metric("R²", f"{overall_r2:.3f}")

st.write(
    f"회귀식: **예측 평균기온 = {overall_slope:.4f} × (연도 − {BASE_YEAR}) "
    f"+ {overall_intercept:.4f}**"
)

fig_overall = go.Figure()

fig_overall.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        hovertemplate="연도: %{x}<br>평균기온: %{y:.2f} °C<extra></extra>",
    )
)

plot_years = np.arange(1900, 2101)
plot_pred = predict(overall_slope, overall_intercept, plot_years)

fig_overall.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred,
        mode="lines",
        name="전체 회귀선",
        hovertemplate="연도: %{x}<br>회귀 예측: %{y:.2f} °C<extra></extra>",
    )
)

fig_overall.update_layout(
    title="서울 연평균기온과 전체 선형회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
)

st.plotly_chart(fig_overall, use_container_width=True)

# ---------------------------------------------------------------------
# 3. 50년 / 100년 학습모델과 공통 테스트
# ---------------------------------------------------------------------
st.header("3. 50년 모델과 100년 모델의 예측 성능 비교")

TRAIN_END = 2005
TEST_START = 2006
TEST_END = 2025

train_50 = yearly[
    (yearly["연도"] >= 1956) & (yearly["연도"] <= TRAIN_END)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) & (yearly["연도"] <= TRAIN_END)
].copy()

test = yearly[
    (yearly["연도"] >= TEST_START) & (yearly["연도"] <= TEST_END)
].copy()

if test.empty:
    st.error("2006–2025년 테스트 데이터가 없습니다.")
    st.stop()

if train_50.empty:
    st.error("1956–2005년 학습 데이터가 없습니다.")
    st.stop()

if train_100.empty:
    st.error("1906–2005년 학습 데이터가 없습니다.")
    st.stop()

if train_100["연도"].min() > 1906:
    st.warning(
        f"원자료에서 1906년부터 모든 연도가 존재하지 않아 "
        f"100년 모델의 실제 학습 시작연도는 {train_100['연도'].min()}년입니다."
    )

result_50 = make_result(
    "50년 학습모델",
    1956,
    2005,
    train_50,
    test,
)

result_100 = make_result(
    "100년 학습모델",
    1906,
    2005,
    train_100,
    test,
)

results = pd.DataFrame([result_50, result_100])

display_results = results[
    [
        "모델",
        "요청 학습기간",
        "실제 학습 시작",
        "실제 학습 종료",
        "학습 연도 수",
        "기울기 (°C/년)",
        "MAE (°C)",
        "MSE (°C²)",
        "R²",
    ]
].copy()

for col in ["기울기 (°C/년)", "MAE (°C)", "MSE (°C²)", "R²"]:
    display_results[col] = display_results[col].astype(float).round(4)

st.dataframe(display_results, use_container_width=True, hide_index=True)

st.subheader("모델별 상세 지표")

m1, m2 = st.columns(2)

with m1:
    st.markdown("### 50년 학습모델")
    st.metric("기울기", f"{result_50['slope']:.5f} °C/년")
    st.metric("MAE", f"{result_50['MAE (°C)']:.3f} °C")
    st.metric("MSE", f"{result_50['MSE (°C²)']:.3f}")
    st.metric("R²", f"{result_50['R²']:.3f}")

with m2:
    st.markdown("### 100년 학습모델")
    st.metric("기울기", f"{result_100['slope']:.5f} °C/년")
    st.metric("MAE", f"{result_100['MAE (°C)']:.3f} °C")
    st.metric("MSE", f"{result_100['MSE (°C²)']:.3f}")
    st.metric("R²", f"{result_100['R²']:.3f}")

# ---------------------------------------------------------------------
# 4. 테스트 구간 실제값 vs 예측값
# ---------------------------------------------------------------------
st.header("4. 2006–2025년 테스트 구간 예측")

pred_50 = predict(result_50["slope"], result_50["intercept"], test["연도"])
pred_100 = predict(result_100["slope"], result_100["intercept"], test["연도"])

fig_test = go.Figure()

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        hovertemplate="연도: %{x}<br>실제: %{y:.2f} °C<extra></extra>",
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 모델 예측",
        hovertemplate="연도: %{x}<br>50년 예측: %{y:.2f} °C<extra></extra>",
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 모델 예측",
        hovertemplate="연도: %{x}<br>100년 예측: %{y:.2f} °C<extra></extra>",
    )
)

fig_test.update_layout(
    title="2006–2025년 실제값과 두 회귀모델의 예측값",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="x unified",
)

st.plotly_chart(fig_test, use_container_width=True)

# ---------------------------------------------------------------------
# 5. 성능 차이 해석
# ---------------------------------------------------------------------
st.header("5. 50년 모델과 100년 모델 비교 해석")

mae_diff = result_50["MAE (°C)"] - result_100["MAE (°C)"]
mse_diff = result_50["MSE (°C²)"] - result_100["MSE (°C²)"]
r2_diff = result_50["R²"] - result_100["R²"]
slope_diff = result_50["slope"] - result_100["slope"]

if mae_diff < 0:
    mae_comment = "50년 모델의 MAE가 더 작아 테스트 구간의 평균적인 예측 오차가 더 작습니다."
elif mae_diff > 0:
    mae_comment = "100년 모델의 MAE가 더 작아 테스트 구간의 평균적인 예측 오차가 더 작습니다."
else:
    mae_comment = "두 모델의 MAE가 같습니다."

if r2_diff > 0:
    r2_comment = "50년 모델의 R²가 더 높습니다."
elif r2_diff < 0:
    r2_comment = "100년 모델의 R²가 더 높습니다."
else:
    r2_comment = "두 모델의 R²가 같습니다."

st.markdown(
    f"""
- **기울기 차이:** 50년 모델 − 100년 모델 = **{slope_diff:.5f} °C/년**
- **MAE 차이:** 50년 모델 − 100년 모델 = **{mae_diff:.4f} °C**
- **MSE 차이:** 50년 모델 − 100년 모델 = **{mse_diff:.4f} °C²**
- **R² 차이:** 50년 모델 − 100년 모델 = **{r2_diff:.4f}**
- **MAE 기준:** {mae_comment}
- **R² 기준:** {r2_comment}
"""
)

st.info(
    "50년 모델은 비교적 최근의 기온 변화 추세에 집중하고, "
    "100년 모델은 더 긴 기간의 장기 추세를 반영합니다. "
    "따라서 두 모델의 기울기가 다를 수 있으며, 어느 모델이 더 좋은지는 "
    "공통 테스트 구간(2006–2025)의 MAE, MSE, R²를 함께 비교해 판단합니다."
)

# ---------------------------------------------------------------------
# 6. 연도 슬라이더를 이용한 예측
# ---------------------------------------------------------------------
st.header("6. 원하는 연도의 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

overall_selected = predict(
    overall_slope,
    overall_intercept,
    [selected_year],
)[0]

model50_selected = predict(
    result_50["slope"],
    result_50["intercept"],
    [selected_year],
)[0]

model100_selected = predict(
    result_100["slope"],
    result_100["intercept"],
    [selected_year],
)[0]

p1, p2, p3 = st.columns(3)

with p1:
    st.metric("전체 모델", f"{overall_selected:.2f} °C")

with p2:
    st.metric("50년 모델", f"{model50_selected:.2f} °C")

with p3:
    st.metric("100년 모델", f"{model100_selected:.2f} °C")

prediction_table = pd.DataFrame(
    {
        "모델": ["전체 모델", "50년 모델", "100년 모델"],
        "예측 연도": [selected_year] * 3,
        "예측 연평균기온 (°C)": [
            overall_selected,
            model50_selected,
            model100_selected,
        ],
    }
)

prediction_table["예측 연평균기온 (°C)"] = prediction_table[
    "예측 연평균기온 (°C)"
].round(2)

st.dataframe(prediction_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# 7. 연도별 원자료
# ---------------------------------------------------------------------
with st.expander("연도별 분석 데이터 보기"):
    st.dataframe(
        yearly[
            ["연도", "연평균기온", "관측일수", "지난연수"]
        ].round(
            {"연평균기온": 2}
        ),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "데이터 출처: 기상청 서울 관측자료(seoul.csv) / 분석 기준: 2025년까지, 연 300일 이상 관측된 연도"
)
