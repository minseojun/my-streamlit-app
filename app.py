# app.py
# -*- coding: utf-8 -*-

import os
import io
import json
import uuid
import sqlite3
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# 쿠키 (권장) : pip install streamlit-cookies-manager
from streamlit_cookies_manager import EncryptedCookieManager  # type: ignore


# -----------------------------
# 기본 설정
# -----------------------------
APP_TITLE = "FAILOG"
DB_PATH = "failog.sqlite3"
COOKIE_PREFIX = "failog"
COOKIE_NAME_USER_ID = "user_id"
COOKIE_MAX_AGE_DAYS = 365 * 3  # 3년

st.set_page_config(page_title=APP_TITLE, page_icon="🧩", layout="wide")


# -----------------------------
# 쿠키 기반 user_id
# -----------------------------
def init_cookie_manager() -> EncryptedCookieManager:
    """
    쿠키는 Streamlit이 기본 제공하지 않아서 외부 컴포넌트로 처리.
    EncryptedCookieManager는 prefix + password로 암호화 쿠키를 관리함.

    권장: .streamlit/secrets.toml
      COOKIE_PASSWORD="아무거나-충분히-긴-문자열"
    """
    cookie_password = None
    try:
        cookie_password = st.secrets.get("COOKIE_PASSWORD")
    except Exception:
        cookie_password = None

    if not cookie_password:
        # 개발 편의용 fallback (서버 재시작 시 바뀌면 쿠키 해독 실패 가능)
        # 실제 배포에서는 secrets.toml로 고정 비밀번호를 꼭 넣는 걸 권장.
        cookie_password = os.environ.get("COOKIE_PASSWORD", "DEV_ONLY_CHANGE_ME_PLEASE_SET_SECRETS")

    cookies = EncryptedCookieManager(prefix=COOKIE_PREFIX, password=cookie_password)
    if not cookies.ready():
        # 쿠키 초기화가 아직 안 되었으면 stop (다음 rerun에서 ready())
        st.stop()
    return cookies


def get_or_create_user_id(cookies: EncryptedCookieManager) -> str:
    uid = cookies.get(COOKIE_NAME_USER_ID)
    if uid and isinstance(uid, str) and len(uid) >= 8:
        return uid

    uid = str(uuid.uuid4())
    cookies[COOKIE_NAME_USER_ID] = uid
    cookies.save()  # 즉시 저장
    return uid


# -----------------------------
# DB
# -----------------------------
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fail_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,       -- ISO8601
            log_date TEXT NOT NULL,         -- YYYY-MM-DD (사용자가 선택)
            title TEXT NOT NULL,
            cause TEXT NOT NULL,            -- 원인(분류)
            detail TEXT,
            emotion TEXT,
            action_plan TEXT,
            weather_json TEXT               -- 스냅샷(옵션)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fail_logs_user_date ON fail_logs(user_id, log_date)")
    conn.commit()


def db_insert_log(
    conn: sqlite3.Connection,
    user_id: str,
    log_date: date,
    title: str,
    cause: str,
    detail: str,
    emotion: str,
    action_plan: str,
    weather: Optional[Dict[str, Any]] = None,
) -> str:
    log_id = str(uuid.uuid4())
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO fail_logs (id, user_id, created_at, log_date, title, cause, detail, emotion, action_plan, weather_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            log_id,
            user_id,
            datetime.now().isoformat(timespec="seconds"),
            log_date.isoformat(),
            title.strip(),
            cause.strip(),
            detail.strip(),
            emotion.strip(),
            action_plan.strip(),
            json.dumps(weather, ensure_ascii=False) if weather else None,
        ),
    )
    conn.commit()
    return log_id


def db_fetch_logs(conn: sqlite3.Connection, user_id: str, start: date, end: date) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM fail_logs
        WHERE user_id = ?
          AND log_date >= ?
          AND log_date <= ?
        ORDER BY log_date ASC, created_at ASC
        """,
        (user_id, start.isoformat(), end.isoformat()),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    # 파싱
    df["log_date"] = pd.to_datetime(df["log_date"]).dt.date
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["weekday"] = pd.to_datetime(df["log_date"]).dt.day_name()
    return df


# -----------------------------
# Open-Meteo (키 필요 없음)
# -----------------------------
WEATHER_CODE_MAP = {
    # 간단히 대표만 매핑 (원하면 더 확장 가능)
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "착빙 안개",
    51: "이슬비(약)",
    53: "이슬비(중)",
    55: "이슬비(강)",
    61: "비(약)",
    63: "비(중)",
    65: "비(강)",
    71: "눈(약)",
    73: "눈(중)",
    75: "눈(강)",
    80: "소나기(약)",
    81: "소나기(중)",
    82: "소나기(강)",
    95: "뇌우",
}


def fetch_weather_snapshot(lat: float, lon: float, tz: str = "Asia/Seoul") -> Dict[str, Any]:
    """
    Open-Meteo forecast endpoint 호출.
    문서: /v1/forecast (no API key)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        # current는 필요한 것만 최소로
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        # daily 요약(주간 리포트에 유용)
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
        "forecast_days": 7,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    # 보기 좋게 정리
    current = data.get("current", {})
    code = current.get("weather_code")
    current_desc = WEATHER_CODE_MAP.get(code, f"code:{code}")

    snapshot = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "lat": lat,
        "lon": lon,
        "timezone": tz,
        "current": {
            "temperature_2m": current.get("temperature_2m"),
            "relative_humidity_2m": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed_10m": current.get("wind_speed_10m"),
            "weather_code": code,
            "weather_desc": current_desc,
            "time": current.get("time"),
        },
        "daily": data.get("daily", {}),
        "daily_units": data.get("daily_units", {}),
    }
    return snapshot


# -----------------------------
# 시각화
# -----------------------------
def plot_counts_bar(series: pd.Series, title: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts = series.value_counts().sort_values(ascending=False)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_ylabel("건수")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_daily_trend(df: pd.DataFrame, title: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    g = df.groupby("log_date")["id"].count().reset_index(name="count")
    ax.plot(g["log_date"], g["count"], marker="o")
    ax.set_title(title)
    ax.set_ylabel("건수")
    ax.set_xlabel("날짜")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -----------------------------
# PDF 리포트 (ReportLab)
# -----------------------------
def make_week_range(anchor: date) -> Tuple[date, date]:
    # 월요일~일요일
    start = anchor - timedelta(days=anchor.weekday())
    end = start + timedelta(days=6)
    return start, end


def build_weekly_pdf_bytes(
    user_id: str,
    week_start: date,
    week_end: date,
    df_week: pd.DataFrame,
    location_label: str,
    weather_snapshot: Optional[Dict[str, Any]],
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def draw_title(text: str, y: float) -> float:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(20 * mm, y, text)
        return y - 10 * mm

    def draw_text(text: str, y: float, size: int = 10) -> float:
        c.setFont("Helvetica", size)
        for line in text.split("\n"):
            c.drawString(20 * mm, y, line[:120])
            y -= 5 * mm
        return y - 2 * mm

    y = height - 20 * mm
    y = draw_title(f"{APP_TITLE} 주간 리포트", y)
    y = draw_text(f"- User: {user_id}", y)
    y = draw_text(f"- 기간: {week_start.isoformat()} ~ {week_end.isoformat()}", y)
    y = draw_text(f"- 위치: {location_label}", y)

    if weather_snapshot:
        cur = weather_snapshot.get("current", {})
        y = draw_text(
            "날씨 스냅샷(현재): "
            f'{cur.get("weather_desc","")}, '
            f'{cur.get("temperature_2m","?")}°C, '
            f'습도 {cur.get("relative_humidity_2m","?")}%, '
            f'강수 {cur.get("precipitation","?")}, '
            f'바람 {cur.get("wind_speed_10m","?")}',
            y,
        )

    y -= 2 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "요약")
    y -= 8 * mm

    total = int(df_week.shape[0]) if not df_week.empty else 0
    top_cause = "-"
    if total > 0:
        top_cause = df_week["cause"].value_counts().index[0]

    y = draw_text(f"- 총 기록 수: {total}건", y)
    y = draw_text(f"- 가장 많은 원인: {top_cause}", y)

    # 차트 2개: 요일/원인
    if total > 0:
        fig_wd = plot_counts_bar(df_week["weekday"], "요일별 기록 수")
        fig_cause = plot_counts_bar(df_week["cause"], "원인별 기록 수")
        img_wd = ImageReader(io.BytesIO(fig_to_png_bytes(fig_wd)))
        img_cause = ImageReader(io.BytesIO(fig_to_png_bytes(fig_cause)))

        # 배치
        chart_w = (width - 40 * mm)
        chart_h = 60 * mm

        y -= 5 * mm
        if y - chart_h < 20 * mm:
            c.showPage()
            y = height - 20 * mm

        c.drawImage(img_wd, 20 * mm, y - chart_h, width=chart_w, height=chart_h, preserveAspectRatio=True, anchor="nw")
        y -= (chart_h + 10 * mm)

        if y - chart_h < 20 * mm:
            c.showPage()
            y = height - 20 * mm

        c.drawImage(
            img_cause, 20 * mm, y - chart_h, width=chart_w, height=chart_h, preserveAspectRatio=True, anchor="nw"
        )
        y -= (chart_h + 10 * mm)

    # 상세 목록(최대 20개 정도)
    if total > 0:
        c.setFont("Helvetica-Bold", 12)
        if y < 35 * mm:
            c.showPage()
            y = height - 20 * mm
        c.drawString(20 * mm, y, "상세 기록(최근/주간)")
        y -= 8 * mm

        df_show = df_week.sort_values(["log_date", "created_at"]).tail(20)
        c.setFont("Helvetica", 9)

        for _, r in df_show.iterrows():
            line = f'{r["log_date"]} | {r["cause"]} | {r["title"]}'
            if y < 20 * mm:
                c.showPage()
                y = height - 20 * mm
                c.setFont("Helvetica", 9)
            c.drawString(20 * mm, y, line[:130])
            y -= 5 * mm

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


# -----------------------------
# UI
# -----------------------------
@dataclass
class LocationPreset:
    label: str
    lat: float
    lon: float


LOCATION_PRESETS = [
    LocationPreset("서울", 37.5665, 126.9780),
    LocationPreset("부산", 35.1796, 129.0756),
    LocationPreset("대전", 36.3504, 127.3845),
    LocationPreset("광주", 35.1595, 126.8526),
    LocationPreset("제주", 33.4996, 126.5312),
]


def main() -> None:
    cookies = init_cookie_manager()
    user_id = get_or_create_user_id(cookies)

    conn = db_connect()
    db_init(conn)

    st.title("🧩 FAILOG")
    st.caption("쿠키 기반 user_id 고정 + Open-Meteo 날씨 + 주간 PDF 리포트 + 트렌드 대시보드")

    with st.sidebar:
        st.subheader("세션")
        st.code(f"user_id = {user_id}", language="text")
        st.divider()

        st.subheader("날씨 위치")
        preset_labels = [p.label for p in LOCATION_PRESETS] + ["직접 입력"]
        pick = st.selectbox("위치 선택", preset_labels, index=0)

        if pick != "직접 입력":
            p = next(x for x in LOCATION_PRESETS if x.label == pick)
            lat, lon = p.lat, p.lon
            location_label = p.label
        else:
            lat = st.number_input("위도(lat)", value=37.5665, format="%.6f")
            lon = st.number_input("경도(lon)", value=126.9780, format="%.6f")
            location_label = f"custom({lat:.4f},{lon:.4f})"

        st.caption("날씨 데이터: Open-Meteo (키 불필요)")

    tab1, tab2, tab3, tab4 = st.tabs(["✍️ 기록하기", "📊 대시보드", "🌤️ 날씨", "🧾 주간 PDF 리포트"])

    # ---- 기록하기
    with tab1:
        st.subheader("오늘의 FAILOG 기록")
        colA, colB = st.columns([1, 1])

        with colA:
            log_date = st.date_input("날짜", value=date.today())
            title = st.text_input("제목(한 줄)", placeholder="예: 발표 준비를 미루다가 밤샘함")
            cause = st.selectbox(
                "원인(분류)",
                ["시간관리", "집중/산만", "커뮤니케이션", "체력/수면", "감정/스트레스", "기술/환경", "기타"],
            )

        with colB:
            emotion = st.text_input("감정(선택)", placeholder="예: 불안, 짜증, 무기력")
            detail = st.text_area("상세(무슨 일이 있었는지)", height=140)
            action_plan = st.text_area("다음엔 어떻게 할지(액션 플랜)", height=110)

        colX, colY = st.columns([1, 1])
        with colX:
            attach_weather = st.checkbox("기록에 현재 날씨 스냅샷 저장", value=True)
        with colY:
            st.write("")

        if st.button("저장", type="primary", use_container_width=True):
            weather = None
            if attach_weather:
                try:
                    weather = fetch_weather_snapshot(lat, lon)
                except Exception as e:
                    st.warning(f"날씨 불러오기 실패(기록은 저장됨): {e}")

            if not title.strip():
                st.error("제목은 필수야.")
            else:
                db_insert_log(
                    conn=conn,
                    user_id=user_id,
                    log_date=log_date,
                    title=title,
                    cause=cause,
                    detail=detail,
                    emotion=emotion,
                    action_plan=action_plan,
                    weather=weather,
                )
                st.success("저장 완료! (새로고침해도 user_id가 고정이면 기록이 안 사라져.)")

        st.divider()
        st.subheader("최근 20개")
        df_recent = db_fetch_logs(conn, user_id, date.today() - timedelta(days=90), date.today())
        if df_recent.empty:
            st.info("아직 기록이 없어. 첫 로그를 저장해봐!")
        else:
            st.dataframe(
                df_recent.sort_values(["created_at"], ascending=False).head(20)[
                    ["log_date", "cause", "title", "emotion", "created_at"]
                ],
                use_container_width=True,
            )

    # ---- 대시보드
    with tab2:
        st.subheader("트렌드 대시보드")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            days = st.selectbox("기간", [7, 14, 30, 60, 90], index=2)
        with col2:
            show_table = st.checkbox("원본 테이블 보기", value=False)
        with col3:
            st.write("")

        start = date.today() - timedelta(days=int(days) - 1)
        end = date.today()
        df = db_fetch_logs(conn, user_id, start, end)

        if df.empty:
            st.info("선택한 기간에 데이터가 없어.")
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("기록 수", f"{len(df)}")
            k2.metric("원인 종류 수", f"{df['cause'].nunique()}")
            k3.metric("가장 많은 원인", df["cause"].value_counts().index[0])

            cA, cB = st.columns(2)
            with cA:
                fig = plot_counts_bar(df["weekday"], "요일별 기록 수")
                st.pyplot(fig, clear_figure=True, use_container_width=True)
            with cB:
                fig = plot_counts_bar(df["cause"], "원인별 기록 수")
                st.pyplot(fig, clear_figure=True, use_container_width=True)

            fig = plot_daily_trend(df, "일자별 기록 트렌드")
            st.pyplot(fig, clear_figure=True, use_container_width=True)

            if show_table:
                st.dataframe(
                    df.sort_values(["log_date", "created_at"], ascending=[False, False])[
                        ["log_date", "weekday", "cause", "title", "emotion"]
                    ],
                    use_container_width=True,
                )

    # ---- 날씨
    with tab3:
        st.subheader("Open-Meteo 날씨")
        st.caption("선택한 위치 기준으로 현재 + 7일 요약을 보여줘.")

        if st.button("날씨 새로 불러오기", use_container_width=True):
            st.session_state["weather_snapshot"] = None

        if "weather_snapshot" not in st.session_state or st.session_state["weather_snapshot"] is None:
            try:
                st.session_state["weather_snapshot"] = fetch_weather_snapshot(lat, lon)
            except Exception as e:
                st.error(f"날씨 불러오기 실패: {e}")
                st.stop()

        snap = st.session_state["weather_snapshot"]
        cur = snap.get("current", {})
        st.write(
            f"**현재:** {cur.get('weather_desc','')} / {cur.get('temperature_2m','?')}°C / "
            f"습도 {cur.get('relative_humidity_2m','?')}% / 강수 {cur.get('precipitation','?')} / "
            f"바람 {cur.get('wind_speed_10m','?')} (time={cur.get('time','?')})"
        )

        daily = snap.get("daily", {})
        if daily and "time" in daily:
            dfd = pd.DataFrame(daily)
            # 코드 -> 설명
            if "weather_code" in dfd.columns:
                dfd["weather_desc"] = dfd["weather_code"].apply(lambda x: WEATHER_CODE_MAP.get(int(x), f"code:{x}"))
            st.dataframe(dfd, use_container_width=True)
        else:
            st.info("일간 데이터가 비어있어.")

    # ---- 주간 PDF 리포트
    with tab4:
        st.subheader("주간 PDF 리포트")
        st.caption("선택한 주(월~일)의 요약 + 차트 + 상세 목록을 PDF로 내보내.")

        anchor = st.date_input("주 선택(아무 날짜나 찍으면 그 주로 묶음)", value=date.today(), key="week_anchor")
        week_start, week_end = make_week_range(anchor)

        df_week = db_fetch_logs(conn, user_id, week_start, week_end)

        st.write(f"**기간:** {week_start.isoformat()} ~ {week_end.isoformat()}")
        if df_week.empty:
            st.info("이 주에는 기록이 없어. 기록부터 남기고 리포트를 뽑아봐!")
        else:
            st.dataframe(
                df_week.sort_values(["log_date", "created_at"], ascending=[True, True])[
                    ["log_date", "weekday", "cause", "title", "emotion"]
                ],
                use_container_width=True,
            )

            colL, colR = st.columns([1, 1])
            with colL:
                fig = plot_counts_bar(df_week["weekday"], "요일별 기록 수(주간)")
                st.pyplot(fig, clear_figure=True, use_container_width=True)
            with colR:
                fig = plot_counts_bar(df_week["cause"], "원인별 기록 수(주간)")
                st.pyplot(fig, clear_figure=True, use_container_width=True)

            # PDF 생성 버튼 + 다운로드
            if st.button("PDF 생성", type="primary", use_container_width=True):
                # PDF에 넣을 날씨 스냅샷(현재)
                weather_for_pdf = None
                try:
                    weather_for_pdf = fetch_weather_snapshot(lat, lon)
                except Exception:
                    weather_for_pdf = None

                pdf_bytes = build_weekly_pdf_bytes(
                    user_id=user_id,
                    week_start=week_start,
                    week_end=week_end,
                    df_week=df_week,
                    location_label=location_label,
                    weather_snapshot=weather_for_pdf,
                )
                st.session_state["latest_pdf"] = pdf_bytes
                st.success("PDF 준비 완료! 아래에서 다운로드해.")

            if "latest_pdf" in st.session_state and st.session_state["latest_pdf"]:
                fname = f"failog_weekly_{week_start.isoformat()}_{week_end.isoformat()}.pdf"
                st.download_button(
                    "PDF 다운로드",
                    data=st.session_state["latest_pdf"],
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                    # 다운로드 클릭 시 앱 rerun 경쟁상황을 줄이려면 ignore 권장(문서 참고)
                    on_click="ignore",
                )

    st.divider()
    st.caption("Tip: 배포 환경에서는 st.secrets에 COOKIE_PASSWORD를 꼭 설정해줘.")


if __name__ == "__main__":
    main()
