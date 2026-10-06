import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    return df.dropna(subset=["날짜", "연도", "평균기온"])


@st.cache_data
def make_yearly_data(df):
    # 수업 기준 기간: 2025년까지.
    # 관측일이 300일 미만인 해는 제외한다.
    yearly = (
        df[df["연도"] <= 2025]
        .groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    # 회귀의 독립 변수: 1908년부터 지난 연수
    yearly["지난연수"] = yearly["연도"] - 1908

    # 전체 유효 연도에 대한 최소제곱 회귀
    slope, intercept = np.polyfit(
        yearly["지난연수"], yearly["평균기온"], 1
    )
    yearly["회귀기온"] = intercept + slope * yearly["지난연수"]
    corr = yearly["지난연수"].corr(yearly["평균기온"])

    return yearly, slope, intercept, corr


def fit_and_evaluate(train_df, test_df):
    """연도(1908년부터 지난 연수)로 선형회귀하고 테스트 성능을 계산한다."""
    x_train = train_df["지난연수"].to_numpy()
    y_train = train_df["평균기온"].to_numpy()
    x_test = test_df["지난연수"].to_numpy()
    y_test = test_df["평균기온"].to_numpy()

    slope, intercept = np.polyfit(x_train, y_train, 1)
    pred = intercept + slope * x_test

    errors = y_test - pred
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "slope": slope,
        "intercept": intercept,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "pred": pred,
    }


try:
    df = load_data()
    yearly, slope, intercept, corr = make_yearly_data(df)
except Exception as e:
    st.error("서울 기온 데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

st.title("🌡️ 기온 예측기")
st.caption("서울 일평균기온 데이터를 이용한 연평균기온 회귀 분석")

# 회귀식: x = 1908년부터 지난 연수
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)

st.info(
    f"회귀선을 만든 해의 개수: **{n_years}개**  |  "
    f"시작 연도: **{start_year}년**  |  끝 연도: **{end_year}년**"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("회귀선 상관계수", f"{corr:.3f}")
with col2:
    st.metric("회귀선 기울기", f"{slope:.4f} °C/년")
with col3:
    st.metric("회귀선 절편", f"{intercept:.2f} °C")

st.subheader("훈련 기간별 선형회귀 성능 비교")

# 공통 테스트: 최근 20년(2006~2025).
# 동일한 연도별 품질 기준(관측일수 300일 이상)을 적용한다.
train_50 = yearly[(yearly["연도"] >= 1956) & (yearly["연도"] <= 2005)].copy()
train_100 = yearly[(yearly["연도"] >= 1906) & (yearly["연도"] <= 2005)].copy()
test = yearly[(yearly["연도"] >= 2006) & (yearly["연도"] <= 2025)].copy()

result_50 = fit_and_evaluate(train_50, test)
result_100 = fit_and_evaluate(train_100, test)

# 1906년 자료가 없으므로 실제 100년 학습에는 유효한 1907년 이후 자료만 들어갈 수 있다.
available_100_start = int(train_100["연도"].min())

comparison = pd.DataFrame(
    [
        {
            "모델": "최근 50년",
            "훈련기간": "1956~2005",
            "실제 훈련 연도 수": len(train_50),
            "기울기 (°C/년)": result_50["slope"],
            "MAE (°C)": result_50["mae"],
            "MSE (°C²)": result_50["mse"],
            "R²": result_50["r2"],
        },
        {
            "모델": "최근 100년",
            "훈련기간": "1906~2005",
            "실제 훈련 시작": available_100_start,
            "실제 훈련 연도 수": len(train_100),
            "기울기 (°C/년)": result_100["slope"],
            "MAE (°C)": result_100["mae"],
            "MSE (°C²)": result_100["mse"],
            "R²": result_100["r2"],
        },
    ]
)

st.markdown(
    f"**공통 테스트 데이터:** 2006~2025년, 관측일수 300일 이상인 연도 "
    f"({len(test)}개 연도)"
)
if available_100_start > 1906:
    st.warning(
        f"100년 학습 구간은 요청하신 1906~2005년으로 지정했지만, "
        f"원본 데이터가 1907년부터 시작하므로 실제 유효 학습은 "
        f"{available_100_start}~2005년입니다."
    )

show_cols = [
    "모델", "훈련기간", "실제 훈련 연도 수",
    "기울기 (°C/년)", "MAE (°C)", "MSE (°C²)", "R²"
]
st.dataframe(
    comparison[show_cols].style.format({
        "기울기 (°C/년)": "{:.4f}",
        "MAE (°C)": "{:.3f}",
        "MSE (°C²)": "{:.3f}",
        "R²": "{:.3f}",
    }),
    hide_index=True,
    width="stretch",
)

slope_diff = result_50["slope"] - result_100["slope"]
mae_diff = result_50["mae"] - result_100["mae"]
mse_diff = result_50["mse"] - result_100["mse"]
r2_diff = result_50["r2"] - result_100["r2"]

st.markdown(
    f"""
**비교 해석**

- **기울기:** 50년 모델은 **{result_50["slope"]:.4f} °C/년**, 
  100년 모델은 **{result_100["slope"]:.4f} °C/년**입니다. 
  (50년 − 100년 차이: **{slope_diff:+.4f} °C/년**)
- **MAE:** 50년 모델 **{result_50["mae"]:.3f} °C**, 
  100년 모델 **{result_100["mae"]:.3f} °C** 
  (50년 − 100년: **{mae_diff:+.3f} °C**)
- **MSE:** 50년 모델 **{result_50["mse"]:.3f} °C²**, 
  100년 모델 **{result_100["mse"]:.3f} °C²** 
  (50년 − 100년: **{mse_diff:+.3f} °C²**)
- **R²:** 50년 모델 **{result_50["r2"]:.3f}**, 
  100년 모델 **{result_100["r2"]:.3f}** 
  (50년 − 100년: **{r2_diff:+.3f}**)

테스트 성능은 **MAE·MSE는 낮을수록**, **R²는 높을수록** 좋습니다.
"""
)

st.subheader("연도별 평균기온과 회귀 직선")

# 회귀 직선은 1900~2100년까지 표시하여 슬라이더의 전체 범위를 보여 준다.
line_years = np.arange(1900, 2101)
line_x = line_years - 1908
line_temps = intercept + slope * line_x

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="회귀 직선",
        hovertemplate="연도: %{x}년<br>회귀기온: %{y:.2f} °C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=10,
        range=[1900, 2100],
    ),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=560,
)

st.plotly_chart(fig, width="stretch")

# 두 훈련 모델의 테스트 구간 예측 비교
fig_test = go.Figure()
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers+lines",
        name="실제 연평균기온",
        hovertemplate="연도: %{x}년<br>실제: %{y:.2f} °C<extra></extra>",
    )
)
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=result_50["pred"],
        mode="lines",
        name="50년 학습 예측",
        hovertemplate="연도: %{x}년<br>50년 예측: %{y:.2f} °C<extra></extra>",
    )
)
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=result_100["pred"],
        mode="lines",
        name="100년 학습 예측",
        hovertemplate="연도: %{x}년<br>100년 예측: %{y:.2f} °C<extra></extra>",
    )
)
fig_test.update_layout(
    title="공통 테스트 기간(2006~2025) 실제값과 모델 예측",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(tickmode="linear", dtick=2),
    hovermode="x unified",
    height=500,
)
st.plotly_chart(fig_test, width="stretch")

st.subheader("연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=min(2025, end_year),
    step=1,
)

predicted_temp = intercept + slope * (selected_year - 1908)

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C",
)

st.caption(
    "예상 기온은 2025년까지의 관측자료 중 관측일수가 300일 이상인 연도의 "
    "연평균기온을 사용한 선형 회귀 결과입니다. "
    "1900~1907년은 회귀 학습자료에서 제외됩니다."
)

with st.expander("분석에 사용된 연도별 자료 보기"):
    display_df = yearly.copy()
    display_df["평균기온"] = display_df["평균기온"].round(2)
    display_df["회귀기온"] = display_df["회귀기온"].round(2)
    st.dataframe(
        display_df[["연도", "관측일수", "평균기온", "회귀기온"]],
        hide_index=True,
        width="stretch",
    )
