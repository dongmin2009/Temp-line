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

    # 최소제곱 회귀 직선
    slope, intercept = np.polyfit(
        yearly["지난연수"], yearly["평균기온"], 1
    )
    yearly["회귀기온"] = (
        intercept + slope * yearly["지난연수"]
    )

    # 상관계수
    corr = yearly["지난연수"].corr(yearly["평균기온"])

    return yearly, slope, intercept, corr


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
