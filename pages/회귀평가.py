
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(
    page_title="서울 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide",
)


@st.cache_data
def load_yearly_data():
    """서울 일별 기온 자료를 읽어 연평균기온 자료로 변환한다."""
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {missing}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year

    # 수업 기준 기간: 2025년까지
    df = df[df["연도"] <= 2025]

    # 연도별 연평균기온과 관측일수
    yearly = (
        df.groupby("연도", as_index=False)
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
    )

    # 관측일수가 300일 미만인 해 제외
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    # 독립변수: 1908년부터 지난 연수
    yearly["지난연수"] = yearly["연도"] - 1908

    return yearly.sort_values("연도").reset_index(drop=True)


def fit_linear_regression(train):
    """최소제곱법으로 y = intercept + slope*x를 적합한다."""
    x = train["지난연수"].to_numpy(dtype=float)
    y = train["평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def predict(slope, intercept, data):
    return intercept + slope * data["지난연수"].to_numpy(dtype=float)


def evaluate(y_true, y_pred):
    """MAE, MSE, R²를 계산한다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return mae, mse, r2


def model_result(name, train_start, train_end, train, test):
    slope, intercept = fit_linear_regression(train)
    pred = predict(slope, intercept, test)
    mae, mse, r2 = evaluate(test["평균기온"], pred)

    return {
        "모델": name,
        "요청한 학습기간": f"{train_start}~{train_end}",
        "실제 학습 시작": int(train["연도"].min()),
        "실제 학습 끝": int(train["연도"].max()),
        "학습 연도 수": len(train),
        "기울기": slope,
        "절편": intercept,
        "MAE": mae,
        "MSE": mse,
        "R²": r2,
        "예측값": pred,
    }


try:
    yearly = load_yearly_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------------------
# 제목
# ---------------------------------------------------------------------
st.title("🌡️ 서울 연평균기온 선형회귀 분석")
st.caption(
    "1908년부터 지난 연수를 독립변수로 사용하여 서울 연평균기온을 선형회귀로 분석합니다."
)

st.info(
    f"분석에 사용된 유효 연도: **{yearly['연도'].min()}~{yearly['연도'].max()}년**, "
    f"총 **{len(yearly)}개 연도**"
)


# ---------------------------------------------------------------------
# 1. 전체 데이터 회귀
# ---------------------------------------------------------------------
st.header("1. 전체 유효 데이터에 대한 선형회귀")

slope_all, intercept_all = fit_linear_regression(yearly)
yearly["회귀기온"] = predict(slope_all, intercept_all, yearly)

_, _, r2_all = evaluate(yearly["평균기온"], yearly["회귀기온"])
corr_all = yearly["지난연수"].corr(yearly["평균기온"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("회귀선 기울기", f"{slope_all:.4f} °C/년")
c2.metric("상관계수", f"{corr_all:.3f}")
c3.metric("R²", f"{r2_all:.3f}")
c4.metric("사용 연도 수", f"{len(yearly)}개")

fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

line_years = np.arange(1900, 2101)
line_x = line_years - 1908
line_y = intercept_all + slope_all * line_x

fig_all.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="전체 데이터 회귀선",
        hovertemplate="%{x}년<br>회귀기온: %{y:.2f} °C<extra></extra>",
    )
)

fig_all.update_layout(
    title="서울 연평균기온과 전체 데이터 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100], dtick=10),
    hovermode="x unified",
    height=550,
)

st.plotly_chart(fig_all, width="stretch")


# ---------------------------------------------------------------------
# 2. 훈련/테스트 데이터 분리
# ---------------------------------------------------------------------
st.header("2. 훈련 데이터와 테스트 데이터")

st.markdown(
    """
- **최근 50년 훈련:** 1956~2005년
- **최근 100년 훈련:** 1906~2005년
- **공통 테스트:** 2006~2025년
- 모든 구간에서 **관측일수 300일 미만인 연도는 제외**
"""
)

test = yearly[(yearly["연도"] >= 2006) & (yearly["연도"] <= 2025)].copy()
train_50 = yearly[(yearly["연도"] >= 1956) & (yearly["연도"] <= 2005)].copy()
train_100 = yearly[(yearly["연도"] >= 1906) & (yearly["연도"] <= 2005)].copy()

if test.empty:
    st.error("2006~2025년 테스트 데이터가 없습니다.")
    st.stop()

if train_50.empty or train_100.empty:
    st.error("훈련 데이터가 충분하지 않습니다.")
    st.stop()

# 1906년 자료가 없는 경우 실제 유효 시작연도를 표시
if train_100["연도"].min() > 1906:
    st.warning(
        f"원본 자료에서 1906년 데이터가 없어 100년 학습모델의 실제 학습은 "
        f"{int(train_100['연도'].min())}년부터 시작합니다."
    )

st.write(
    f"공통 테스트 데이터는 유효한 **{len(test)}개 연도**이며, "
    f"{int(test['연도'].min())}~{int(test['연도'].max())}년입니다."
)


# ---------------------------------------------------------------------
# 3. 두 모델 학습 및 평가
# ---------------------------------------------------------------------
st.header("3. 최근 50년 vs 최근 100년 모델 평가")

result_50 = model_result(
    "최근 50년 모델", 1956, 2005, train_50, test
)

result_100 = model_result(
    "최근 100년 모델", 1906, 2005, train_100, test
)

results = pd.DataFrame(
    [
        {k: v for k, v in result_50.items() if k != "예측값"},
        {k: v for k, v in result_100.items() if k != "예측값"},
    ]
)

st.subheader("성능 비교")

display = results[
    [
        "모델",
        "요청한 학습기간",
        "실제 학습 시작",
        "실제 학습 끝",
        "학습 연도 수",
        "기울기",
        "MAE",
        "MSE",
        "R²",
    ]
].copy()

st.dataframe(
    display.style.format(
        {
            "기울기": "{:.4f}",
            "MAE": "{:.3f}",
            "MSE": "{:.3f}",
            "R²": "{:.3f}",
        }
    ),
    hide_index=True,
    width="stretch",
)

# 핵심 수치를 카드로 표시
a, b, c = st.columns(3)

with a:
    st.metric(
        "50년 모델 기울기",
        f"{result_50['slope']:.4f} °C/년",
        delta=f"{result_50['slope'] - result_100['slope']:+.4f} (50년-100년)",
    )

with b:
    st.metric(
        "50년 모델 MAE",
        f"{result_50['mae']:.3f} °C",
        delta=f"{result_50['mae'] - result_100['mae']:+.3f}",
        delta_color="inverse",
    )

with c:
    st.metric(
        "50년 모델 R²",
        f"{result_50['r2']:.3f}",
        delta=f"{result_50['r2'] - result_100['r2']:+.3f}",
    )


# ---------------------------------------------------------------------
# 4. 테스트 데이터 실제값 vs 예측값
# ---------------------------------------------------------------------
st.subheader("공통 테스트 기간: 실제값과 예측값")

fig_test = go.Figure()

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        hovertemplate="%{x}년<br>실제: %{y:.2f} °C<extra></extra>",
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=result_50["예측값"],
        mode="lines",
        name="50년 학습 회귀선",
        hovertemplate="%{x}년<br>50년 예측: %{y:.2f} °C<extra></extra>",
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=result_100["예측값"],
        mode="lines",
        name="100년 학습 회귀선",
        hovertemplate="%{x}년<br>100년 예측: %{y:.2f} °C<extra></extra>",
    )
)

fig_test.update_layout(
    title="2006~2025년 테스트 데이터 예측 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(dtick=2),
    hovermode="x unified",
    height=550,
)

st.plotly_chart(fig_test, width="stretch")


# ---------------------------------------------------------------------
# 5. 해석
# ---------------------------------------------------------------------
st.header("4. 결과 해석")

slope_diff = result_50["slope"] - result_100["slope"]

if result_50["mae"] < result_100["mae"]:
    mae_comment = "50년 모델의 MAE가 더 낮아 평균적인 예측 오차가 작습니다."
else:
    mae_comment = "100년 모델의 MAE가 더 낮아 평균적인 예측 오차가 작습니다."

if result_50["mse"] < result_100["mse"]:
    mse_comment = "50년 모델의 MSE가 더 낮아 큰 오차까지 고려한 예측 성능이 더 좋습니다."
else:
    mse_comment = "100년 모델의 MSE가 더 낮아 큰 오차까지 고려한 예측 성능이 더 좋습니다."

if result_50["r2"] > result_100["r2"]:
    r2_comment = "50년 모델의 R²가 더 높아 테스트 기간의 기온 변동을 더 잘 설명합니다."
else:
    r2_comment = "100년 모델의 R²가 더 높아 테스트 기간의 기온 변동을 더 잘 설명합니다."

st.markdown(
    f"""
### 기울기 비교
- 최근 50년 모델: **{result_50['slope']:.4f} °C/년**
- 최근 100년 모델: **{result_100['slope']:.4f} °C/년**
- 차이(50년 − 100년): **{slope_diff:+.4f} °C/년**

따라서 두 학습기간을 사용했을 때 장기적인 연평균기온 증가 추세의 기울기가
서로 다르게 나타납니다.

### 테스트 성능 비교
- **MAE:** 50년 = **{result_50['mae']:.3f} °C**, 100년 = **{result_100['mae']:.3f} °C**
  → {mae_comment}
- **MSE:** 50년 = **{result_50['mse']:.3f} °C²**, 100년 = **{result_100['mse']:.3f} °C²**
  → {mse_comment}
- **R²:** 50년 = **{result_50['r2']:.3f}**, 100년 = **{result_100['r2']:.3f}**
  → {r2_comment}

**해석 기준:** MAE와 MSE는 작을수록 좋고, R²는 클수록 좋습니다.
"""
)


# ---------------------------------------------------------------------
# 6. 선택 연도 예상 기온
# ---------------------------------------------------------------------
st.header("5. 회귀선으로 미래 연도 예상하기")

selected_year = st.slider(
    "예상할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

pred_all = intercept_all + slope_all * (selected_year - 1908)
pred_50 = result_50["intercept"] + result_50["slope"] * (selected_year - 1908)
pred_100 = result_100["intercept"] + result_100["slope"] * (selected_year - 1908)

p1, p2, p3 = st.columns(3)

p1.metric("전체 데이터 회귀", f"{pred_all:.2f} °C")
p2.metric("최근 50년 회귀", f"{pred_50:.2f} °C")
p3.metric("최근 100년 회귀", f"{pred_100:.2f} °C")


# ---------------------------------------------------------------------
# 7. 데이터 확인
# ---------------------------------------------------------------------
with st.expander("연도별 연평균기온 데이터 확인"):
    show = yearly[
        ["연도", "관측일수", "평균기온", "지난연수", "회귀기온"]
    ].copy()
    show["평균기온"] = show["평균기온"].round(2)
    show["회귀기온"] = show["회귀기온"].round(2)
    st.dataframe(show, hide_index=True, width="stretch")
