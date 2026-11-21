import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="🌱 우리 학교 극지식물 개체별 생육 차이 분석")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")


# ============================================================
# 데이터 로딩 (파일 1회만 읽고 재사용)
# ============================================================
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


@st.cache_data
def load_data(env_file, growth_file):
    env_raw = read_csv(env_file) if env_file is not None else pd.DataFrame()
    growth_raw = read_csv(growth_file) if growth_file is not None else pd.DataFrame()
    env_df = std_cols(env_raw) if not env_raw.empty else env_raw
    growth_df = std_cols(growth_raw) if not growth_raw.empty else growth_raw
    return env_df, growth_df, env_raw, growth_raw


def school_name(file):
    if file is None:
        return "우리 학교"
    return file.name.split("_")[0]


env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv", help="UTF-8 또는 CP949 인코딩")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv", help="UTF-8 또는 CP949 인코딩")

if env_file is None and growth_file is None:
    st.info("아직 데이터가 없습니다. 사이드바에서 환경·생육 CSV 파일을 올려 주세요.")

with st.spinner("CSV 파일을 읽고 있습니다..."):
    try:
        env_df, growth_df, env_raw, growth_raw = load_data(env_file, growth_file)
    except Exception:
        env_df, growth_df, env_raw, growth_raw = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        st.warning("⚠️ CSV 파일을 읽지 못했습니다. 파일 형식을 확인해 주세요.")

school = school_name(env_file)
if not school:
    school = school_name(growth_file) if growth_file is not None else "우리 학교"

bar_metrics = {
    "평균 생중량": "생중량",
    "평균 잎 수": "잎 수",
    "평균 지상부 길이": "지상부 길이",
    "평균 지하부 길이": "지하부 길이",
}

with st.sidebar:
    st.header("📊 설정")
    st.caption(f"학교: {school}")
    metrics = st.multiselect(
        "생육 비교 항목 선택",
        ["평균 생중량", "평균 잎 수", "평균 지상부 길이", "평균 지하부 길이"],
        default=["평균 생중량"],
    )

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.subheader("연구 배경 및 목적")
        st.write(
            f"우리 학교({school})에서 수집한 환경 데이터와 생육 결과 데이터를 이용해, "
            f"극지 식물의 개체별 생육 차이가 어떤 환경 요인과 관계있는지 살펴봅니다. "
            f"온도·습도 같은 환경 조건이 개체의 생장(생물량)에 미치는 영향을 비교하는 것이 목적입니다."
        )

        st.markdown("### 데이터 요약")
        if growth_df is not None and not growth_df.empty:
            summary_rows = [
                {"항목": "총 개체 수", "값": len(growth_df)},
            ]
            if "생중량" in growth_df.columns:
                summary_rows.append({"항목": "평균 생중량(g)", "값": f"{growth_df['생중량'].mean():.2f}"})
            if "잎 수" in growth_df.columns:
                summary_rows.append({"항목": "평균 잎 수", "값": f"{growth_df['잎 수'].mean():.1f}"})
            if "지상부 길이" in growth_df.columns:
                summary_rows.append({"항목": "평균 지상부 길이(mm)", "값": f"{growth_df['지상부 길이'].mean():.1f}"})
            if "지하부 길이" in growth_df.columns:
                summary_rows.append({"항목": "평균 지하부 길이(mm)", "값": f"{growth_df['지하부 길이'].mean():.1f}"})
        else:
            summary_rows = [{"항목": "총 개체 수", "값": "데이터 없음"}]
        period = "측정 기간 없음"
        if env_df is not None and "time" in env_df.columns:
            t = env_df["time"].dropna()
            if len(t) > 0:
                period = f"{t.min().strftime('%Y-%m-%d')} ~ {t.max().strftime('%Y-%m-%d')}"
        summary_rows.append({"항목": "측정 기간", "값": period})
        summary_rows.append({"항목": "측정 항목", "값": "환경: 온도·습도·pH·EC / 생육: 잎 수·길이·질량·생중량"})
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")

        st.markdown("### 주요 지표")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("총 개체 수", len(growth_df) if growth_df is not None and not growth_df.empty else "—")
        with c2:
            if env_df is not None and "temperature" in env_df.columns and not env_df["temperature"].isna().all():
                st.metric("평균 온도", f"{env_df['temperature'].mean():.1f} °C")
            else:
                st.metric("평균 온도", "—")
        with c3:
            if env_df is not None and "humidity" in env_df.columns and not env_df["humidity"].isna().all():
                st.metric("평균 습도", f"{env_df['humidity'].mean():.1f} %")
            else:
                st.metric("평균 습도", "—")
        with c4:
            if growth_df is not None and "생중량" in growth_df.columns and not growth_df["생중량"].isna().all():
                st.metric("평균 생중량", f"{growth_df['생중량'].mean():.2f} g")
            else:
                st.metric("평균 생중량", "—")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        st.subheader(f"환경 데이터 ({school})")
        if env_df is None or env_df.empty:
            st.info("환경 데이터가 없습니다. 사이드바에서 환경 데이터 CSV 파일을 올려 주세요.")
        else:
            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_num_cols:
                summary = env_df[env_num_cols].agg(["mean", "min", "max"]).T
                summary.columns = ["평균", "최소", "최대"]
                st.dataframe(summary.round(2), width="stretch")
            else:
                st.info("수치 기반 환경 열이 없습니다.")

            if env_num_cols and "time" in env_df.columns and not env_df["time"].isna().all():
                daily = env_df.groupby(env_df["time"].dt.date)[env_num_cols].mean().reset_index()
                daily.columns = ["날짜"] + env_num_cols
                daily["날짜"] = pd.to_datetime(daily["날짜"])
                has_ph = "ph" in env_num_cols
                has_ec = "ec" in env_num_cols
                n_rows = 2 if (has_ph and has_ec) else 1
                fig = make_subplots(
                    rows=n_rows,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1 if n_rows == 2 else 0.08,
                    subplot_titles=(
                        ("온도 변화", "습도 변화", "pH 변화", "EC 변화")
                        if n_rows == 2
                        else ("온도 변화", "습도 변화")
                    ),
                )
                if "temperature" in daily.columns:
                    fig.add_trace(
                        go.Scatter(x=daily["날짜"], y=daily["temperature"], mode="lines", name="온도(°C)"),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(x=daily["날짜"], y=daily["humidity"], mode="lines", name="습도(%)"),
                        row=1,
                        col=1,
                    )
                if n_rows == 2:
                    if has_ph:
                        fig.add_trace(
                            go.Scatter(x=daily["날짜"], y=daily["ph"], mode="lines", name="pH", line=dict(dash="dash")),
                            row=2,
                            col=1,
                        )
                    if has_ec:
                        fig.add_trace(
                            go.Scatter(x=daily["날짜"], y=daily["ec"], mode="lines", name="EC", line=dict(dash="dash")),
                            row=2,
                            col=1,
                        )
                fig.update_layout(
                    height=500,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                )
                st.plotly_chart(fig, width="stretch")

                with st.expander("📂 환경 데이터 원본 테이블"):
                    st.dataframe(env_raw, width="stretch")
                    btn_env = env_raw.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "환경 데이터 CSV 다운로드",
                        btn_env,
                        file_name=f"{school}_환경데이터.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        st.subheader(f"생육 결과 ({school})")
        if growth_df is None or growth_df.empty:
            st.info("생육 데이터가 없습니다. 사이드바에서 생육 결과 CSV 파일을 올려 주세요.")
        else:
            st.markdown("### 핵심 결과")
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) > 0:
                    mean_gw = gw.mean()
                    max_idx = gw.idxmax()
                    min_idx = gw.idxmin()
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("평균 생중량", f"{mean_gw:.2f} g")
                    with c2:
                        st.metric("최대 개체", f"{growth_df.loc[max_idx, '개체번호']}")
                    with c3:
                        st.metric("최소 개체", f"{growth_df.loc[min_idx, '개체번호']}")
                else:
                    st.info("생중량 값이 없습니다.")
            else:
                st.info("생중량 열이 없습니다.")

            st.markdown("### 개체별 생중량 분포와 차이")
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    stats_rows = [
                        {"항목": "평균", "값": f"{gw.mean():.2f} g"},
                        {"항목": "표준편차", "값": f"{gw.std():.2f} g"},
                        {"항목": "변동계수 CV(%)", "값": f"{gw.std() / gw.mean() * 100:.1f} %"},
                    ]
                    st.dataframe(pd.DataFrame(stats_rows), width="stretch")
                else:
                    st.info("생중량 통계는 데이터가 2개 이상 있어야 표시됩니다.")

            st.markdown("### 상관관계 분석")
            if "생중량" in growth_df.columns:
                if "잎 수" in growth_df.columns:
                    sub1 = growth_df[["잎 수", "생중량"]].dropna()
                    if len(sub1) >= 2:
                        corr1 = sub1["잎 수"].corr(sub1["생중량"])
                        fig1 = go.Figure()
                        fig1.add_trace(
                            go.Scatter(x=sub1["잎 수"], y=sub1["생중량"], mode="markers", name="개체")
                        )
                        coeffs1 = np.polyfit(sub1["잎 수"], sub1["생중량"], 1)
                        xs1 = np.linspace(sub1["잎 수"].min(), sub1["잎 수"].max(), 100)
                        fig1.add_trace(
                            go.Scatter(
                                x=xs1,
                                y=np.poly1d(coeffs1)(xs1),
                                mode="lines",
                                name="추세선",
                            )
                        )
                        fig1.update_layout(
                            title="잎 수 vs 생중량",
                            xaxis_title="잎 수(장)",
                            yaxis_title="생중량(g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                        st.plotly_chart(fig1, width="stretch")
                        st.caption(f"상관계수: {corr1:,.2f}")
                    else:
                        st.info("잎 수-생중량 상관관계는 데이터가 2개 이상 필요합니다.")
                else:
                    st.info("잎 수 열이 없습니다.")

                if "지상부 길이" in growth_df.columns:
                    sub2 = growth_df[["지상부 길이", "생중량"]].dropna()
                    if len(sub2) >= 2:
                        corr2 = sub2["지상부 길이"].corr(sub2["생중량"])
                        fig2 = go.Figure()
                        fig2.add_trace(
                            go.Scatter(x=sub2["지상부 길이"], y=sub2["생중량"], mode="markers", name="개체")
                        )
                        coeffs2 = np.polyfit(sub2["지상부 길이"], sub2["생중량"], 1)
                        xs2 = np.linspace(sub2["지상부 길이"].min(), sub2["지상부 길이"].max(), 100)
                        fig2.add_trace(
                            go.Scatter(
                                x=xs2,
                                y=np.poly1d(coeffs2)(xs2),
                                mode="lines",
                                name="추세선",
                            )
                        )
                        fig2.update_layout(
                            title="지상부 길이 vs 생중량",
                            xaxis_title="지상부 길이(mm)",
                            yaxis_title="생중량(g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                        st.plotly_chart(fig2, width="stretch")
                        st.caption(f"상관계수: {corr2:,.2f}")
                    else:
                        st.info("지상부 길이-생중량 상관관계는 데이터가 2개 이상 필요합니다.")
                else:
                    st.info("지상부 길이 열이 없습니다.")

            st.markdown("### 생중량 상위 25% vs 하위 25% 개체 비교")
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 4:
                    hi_mask = gw >= gw.quantile(0.75)
                    lo_mask = gw <= gw.quantile(0.25)
                    compare_rows = []
                    for disp, col in bar_metrics.items():
                        if disp not in metrics or col not in growth_df.columns:
                            continue
                        compare_rows.append(
                            {
                                "항목": disp,
                                "상위 25%": f"{growth_df.loc[hi_mask, col].mean():.2f}",
                                "하위 25%": f"{growth_df.loc[lo_mask, col].mean():.2f}",
                            }
                        )
                    if compare_rows:
                        st.dataframe(pd.DataFrame(compare_rows), width="stretch")
                    else:
                        st.info("비교할 생육 항목 열이 없습니다.")
                else:
                    st.info("개체 수가 4개 미만이라 상위·하위 25% 비교가 불가능합니다.")

            st.markdown("### 개체별 생육 비교")
            if growth_df is not None and not growth_df.empty and "개체번호" in growth_df.columns:
                tmp = growth_df.reset_index(drop=True)
                bar_cols = [(disp, col) for disp, col in bar_metrics.items() if disp in metrics and col in tmp.columns]
                if bar_cols:
                    cols = st.columns(2)
                    for i, (disp, col) in enumerate(bar_cols):
                        with cols[i % 2]:
                            fig = go.Figure()
                            if disp == "평균 생중량":
                                hi = tmp[col] >= tmp[col].quantile(0.75)
                                lo = tmp[col] <= tmp[col].quantile(0.25)
                                colors = pd.Series(index=tmp.index, dtype="object")
                                colors[hi] = "#2ecc71"
                                colors[lo] = "#e74c3c"
                                colors = colors.fillna("#bdc3c7")
                                fig.add_trace(
                                    go.Bar(
                                        x=tmp.index,
                                        y=tmp[col],
                                        marker_color=colors.tolist(),
                                        text=tmp["개체번호"].astype(str),
                                        textposition="outside",
                                    )
                                )
                            else:
                                fig.add_trace(
                                    go.Bar(
                                        x=tmp.index,
                                        y=tmp[col],
                                        text=tmp["개체번호"].astype(str),
                                        textposition="outside",
                                    )
                                )
                            fig.update_layout(
                                title=f"{disp}별 개체 비교",
                                xaxis_title="개체 (순서)",
                                yaxis_title=disp,
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            )
                            fig.update_traces(width=0.7)
                            st.plotly_chart(fig, width="stretch")
                else:
                    st.info("선택한 생육 비교 항목을 표시할 데이터가 없습니다.")
            else:
                st.info("개체 정보를 읽을 수 없습니다.")

            st.markdown("### 생중량 분포")
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) >= 2:
                    fig = px.violin(growth_df, y="생중량", box=True, points="outliers", title="생중량 분포")
                    fig.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("생중량 분포 표시에는 데이터가 2개 이상 필요합니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
