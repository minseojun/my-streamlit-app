# app.py
# ------------------------------------------------------------
# FAILOG (Device-separated, no-login) - COOKIE VERSION (READY-SAFE)
# ✅ Same browser/device: refresh/reopen keeps everything (user_id + settings)
# ✅ Different browser/device: completely different app instance (different user_id)
#
# Key fix:
# - localStorage 완전 제거
# - streamlit-cookies-manager의 CookiesNotReady 루프 방지:
#   - ready() 되기 전에는 쿠키 get/set/del 절대 호출하지 않음
#   - ready 전에는 임시 session uid로 UI를 "끝까지" 렌더해 ready가 뜨게 함
#   - ready 되는 순간 cookie uid를 고정 저장하고 rerun
#
# Features added (per your request):
# A) Open-Meteo 날씨 연동 (키 필요 없음)
# B) 주간 PDF 리포트 내보내기 + 대시보드(요일/원인 트렌드)
#
# Install:
#   pip install streamlit pandas altair openai requests reportlab matplotlib streamlit-cookies-manager
#   (optional) pip install streamlit-autorefresh
#
# Run:
#   streamlit run app.py
# ------------------------------------------------------------

import json
import re
import sqlite3
import uuid
import io
import os
import glob
from datetime import date, datetime, timedelta, time
from typing import Optional, List, Dict, Any

import pandas as pd
import streamlit as st
import altair as alt
from zoneinfo import ZoneInfo
import requests

# Cookies
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_cookies_manager.cookie_manager import CookiesNotReady

# Optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# charts for PDF images
import matplotlib.pyplot as plt


KST = ZoneInfo("Asia/Seoul")
DB_PATH = "planner.db"
ACCENT = "#A0C4F2"

# ============================================================
# ✅ COOKIES (READY-SAFE)
# ============================================================
# IMPORTANT:
# - 아래 password는 반드시 프로젝트에서 "충분히 긴 랜덤 문자열"로 바꾸세요.
# - OpenAI 키를 쿠키에 저장하는 것은 보안상 민감합니다. "쿠키 저장"을 켠 경우에만 저장됩니다.
COOKIE_PASSWORD = "CHANGE_THIS_TO_A_RANDOM_LONG_SECRET_32CHARS_PLUS"

def cookies() -> EncryptedCookieManager:
    if "cookie_mgr" not in st.session_state:
        st.session_state["cookie_mgr"] = EncryptedCookieManager(
            prefix="failog_",
            password=COOKIE_PASSWORD,
        )
    return st.session_state["cookie_mgr"]

def cookie_ready() -> bool:
    try:
        return cookies().ready()
    except Exception:
        return False

def ck_get(key: str, default: str = "") -> str:
    # ready 전에는 절대 mgr.get 호출 금지
    if not cookie_ready():
        return default
    mgr = cookies()
    try:
        v = mgr.get(key)
        return default if v is None else str(v)
    except CookiesNotReady:
        return default
    except Exception:
        return default

def ck_set(key: str, value: str):
    # ready 전에는 절대 mgr[...] / save 호출 금지
    if not cookie_ready():
        return
    mgr = cookies()
    try:
        mgr[key] = str(value if value is not None else "")
        mgr.save()
    except CookiesNotReady:
        return
    except Exception:
        return

def ck_del(key: str):
    if not cookie_ready():
        return
    mgr = cookies()
    try:
        # __contains__도 내부적으로 cookies를 읽을 수 있어 안전하게 try 안에서 처리
        if key in mgr:
            del mgr[key]
            mgr.save()
    except CookiesNotReady:
        return
    except Exception:
        return


# -------------------------
# UI / CSS
# -------------------------
def inject_css():
    st.markdown(
        f"""
<style>
.block-container {{
  max-width: 1120px;
  padding-top: 1.0rem;
  padding-bottom: 2.2rem;
}}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 420px at 30% 0%, rgba(160,196,242,0.28), rgba(255,255,255,0) 60%),
              linear-gradient(180deg, rgba(160,196,242,0.18) 0%, rgba(255,255,255,1) 55%);
}}
.small {{
  color: rgba(31,36,48,0.65);
  font-size: 0.92rem;
}}
.card {{
  border: 1px solid rgba(160,196,242,0.58);
  border-radius: 18px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.94);
  box-shadow: 0 10px 26px rgba(160,196,242,0.14);
}}
.task {{
  border: 1px solid rgba(160,196,242,0.46);
  border-radius: 16px;
  padding: 10px 10px;
  background: rgba(255,255,255,0.95);
}}
.task + .task {{ margin-top: 8px; }}

.pill {{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:4px 10px;
  border-radius:999px;
  border:1px solid rgba(160,196,242,0.60);
  font-size:0.82rem;
  background: rgba(255,255,255,0.80);
  color: rgba(31,36,48,0.78);
}}
.pill-strong {{
  background: rgba(160,196,242,0.28);
  border-color: rgba(160,196,242,0.88);
  color: rgba(31,36,48,0.90);
}}
div[data-testid="stButton"] > button {{
  border-radius: 14px !important;
  white-space: nowrap !important;
}}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
  border-radius: 14px !important;
  border: 1px solid rgba(160,196,242,0.55) !important;
}}
hr {{
  margin: 1.1rem 0;
  border: none;
  border-top: 1px solid rgba(160,196,242,0.35);
}}
</style>
""",
        unsafe_allow_html=True,
    )


# -------------------------
# Stable device user_id (COOKIE, READY-SAFE)
# -------------------------
def get_or_create_user_id() -> str:
    # 1) 쿠키가 ready이면: cookie uid를 source of truth로 고정
    if cookie_ready():
        uid = ck_get("uid", "").strip()
        if uid:
            st.session_state["user_id"] = uid
            return uid

        # 쿠키 ready인데 uid가 없으면 생성 후 저장
        new_uid = str(uuid.uuid4())
        st.session_state["user_id"] = new_uid
        ck_set("uid", new_uid)
        st.rerun()

    # 2) 쿠키 not-ready이면: 임시 session uid로 앱을 끝까지 렌더 (ready 뜰 기회를 줌)
    if not st.session_state.get("user_id"):
        st.session_state["user_id"] = str(uuid.uuid4())

    # 안내만(중단 금지)
    st.info("쿠키 초기화 중… 잠시 후 자동으로 고정 user_id로 전환돼요.")
    return st.session_state["user_id"]


# -------------------------
# DB
# -------------------------
def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA foreign_keys = ON;")
    return c

def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")

def init_db():
    c = conn()
    cur = c.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS habits (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          title TEXT NOT NULL,
          dow_mask TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          task_date TEXT NOT NULL,
          text TEXT NOT NULL,
          source TEXT NOT NULL CHECK(source IN ('plan','habit')),
          habit_id INTEGER,
          status TEXT NOT NULL CHECK(status IN ('todo','success','fail')) DEFAULT 'todo',
          fail_reason TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(user_id, task_date, source, habit_id, text)
        );
        """
    )
    c.commit()
    c.close()


# -------------------------
# Date helpers (Mon-Sun)
# -------------------------
def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())

def week_days(ws: date) -> List[date]:
    return [ws + timedelta(days=i) for i in range(7)]

def korean_dow(i: int) -> str:
    return ["월", "화", "수", "목", "금", "토", "일"][i]

def month_grid(year: int, month: int) -> List[List[Optional[date]]]:
    first = date(year, month, 1)
    first_wd = first.weekday()
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)

    grid: List[List[Optional[date]]] = []
    row: List[Optional[date]] = [None] * 7
    day = 1
    idx = first_wd

    while day <= last.day:
        row[idx] = date(year, month, day)
        day += 1
        idx += 1
        if idx == 7:
            grid.append(row)
            row = [None] * 7
            idx = 0

    if any(x is not None for x in row):
        grid.append(row)
    return grid


# -------------------------
# Habits / Tasks
# -------------------------
def list_habits(user_id: str, active_only: bool = True) -> pd.DataFrame:
    c = conn()
    q = "SELECT id, title, dow_mask, active FROM habits WHERE user_id=?"
    params = [user_id]
    if active_only:
        q += " AND active=1"
    q += " ORDER BY id DESC"
    df = pd.read_sql_query(q, c, params=params)
    c.close()
    return df

def add_habit(user_id: str, title: str, dows: List[int]):
    title = (title or "").strip()
    if not title:
        return
    mask = ["0"] * 7
    for i in dows:
        if 0 <= i <= 6:
            mask[i] = "1"
    dow_mask = "".join(mask)

    c = conn()
    c.execute(
        """
        INSERT INTO habits(user_id, title, dow_mask, active, created_at, updated_at)
        VALUES (?,?,?,1,?,?)
        """,
        (user_id, title, dow_mask, now_iso(), now_iso()),
    )
    c.commit()
    c.close()

def set_habit_active(user_id: str, habit_id: int, active: bool):
    c = conn()
    c.execute(
        "UPDATE habits SET active=?, updated_at=? WHERE user_id=? AND id=?",
        (1 if active else 0, now_iso(), user_id, habit_id),
    )
    c.commit()
    c.close()

def delete_habit(user_id: str, habit_id: int):
    today = date.today().isoformat()
    c = conn()
    cur = c.cursor()
    cur.execute(
        """
        DELETE FROM tasks
        WHERE user_id=? AND source='habit' AND habit_id=? AND task_date>=? AND status='todo'
        """,
        (user_id, habit_id, today),
    )
    cur.execute("DELETE FROM habits WHERE user_id=? AND id=?", (user_id, habit_id))
    c.commit()
    c.close()

def ensure_week_habit_tasks(user_id: str, ws: date):
    habits = list_habits(user_id, active_only=True)
    if habits.empty:
        return

    days = week_days(ws)
    c = conn()
    cur = c.cursor()
    for _, h in habits.iterrows():
        hid = int(h["id"])
        title = str(h["title"])
        mask = str(h["dow_mask"] or "0000000")
        for d in days:
            if mask[d.weekday()] == "1":
                cur.execute(
                    """
                    INSERT OR IGNORE INTO tasks
                      (user_id, task_date, text, source, habit_id, status, fail_reason, created_at, updated_at)
                    VALUES (?,?,?,?,?,'todo',NULL,?,?)
                    """,
                    (user_id, d.isoformat(), title, "habit", hid, now_iso(), now_iso()),
                )
    c.commit()
    c.close()

def add_plan_task(user_id: str, d: date, text: str):
    text = (text or "").strip()
    if not text:
        return
    c = conn()
    c.execute(
        """
        INSERT INTO tasks
          (user_id, task_date, text, source, habit_id, status, fail_reason, created_at, updated_at)
        VALUES (?,?,?,?,?,'todo',NULL,?,?)
        """,
        (user_id, d.isoformat(), text, "plan", None, now_iso(), now_iso()),
    )
    c.commit()
    c.close()

def delete_task(user_id: str, task_id: int):
    c = conn()
    c.execute("DELETE FROM tasks WHERE user_id=? AND id=?", (user_id, task_id))
    c.commit()
    c.close()

def list_tasks_for_date(user_id: str, d: date) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(
        """
        SELECT id, task_date, text, source, habit_id, status, fail_reason
        FROM tasks
        WHERE user_id=? AND task_date=?
        ORDER BY source DESC, id DESC
        """,
        c,
        params=(user_id, d.isoformat()),
    )
    c.close()
    return df

def update_task_status(user_id: str, task_id: int, status: str):
    c = conn()
    c.execute(
        "UPDATE tasks SET status=?, updated_at=? WHERE user_id=? AND id=?",
        (status, now_iso(), user_id, task_id),
    )
    if status != "fail":
        c.execute(
            "UPDATE tasks SET fail_reason=NULL, updated_at=? WHERE user_id=? AND id=?",
            (now_iso(), user_id, task_id),
        )
    c.commit()
    c.close()

def update_task_fail(user_id: str, task_id: int, reason: str):
    reason = (reason or "").strip() or "이유 미기록"
    c = conn()
    c.execute(
        "UPDATE tasks SET status='fail', fail_reason=?, updated_at=? WHERE user_id=? AND id=?",
        (reason, now_iso(), user_id, task_id),
    )
    c.commit()
    c.close()

def get_tasks_range(user_id: str, start_d: date, end_d: date) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(
        """
        SELECT id, task_date, text, source, habit_id, status, fail_reason
        FROM tasks
        WHERE user_id=? AND task_date BETWEEN ? AND ?
        ORDER BY task_date ASC, id DESC
        """,
        c,
        params=(user_id, start_d.isoformat(), end_d.isoformat()),
    )
    c.close()
    return df

def get_all_failures(user_id: str, limit: int = 350) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(
        """
        SELECT task_date, text, source, habit_id, fail_reason
        FROM tasks
        WHERE user_id=? AND status='fail'
        ORDER BY task_date DESC
        LIMIT ?
        """,
        c,
        params=(user_id, limit),
    )
    c.close()
    return df

def count_today_todos(user_id: str) -> int:
    today = date.today().isoformat()
    c = conn()
    row = c.execute(
        "SELECT COUNT(*) FROM tasks WHERE user_id=? AND task_date=? AND status='todo'",
        (user_id, today),
    ).fetchone()
    c.close()
    return int(row[0] if row else 0)


# -------------------------
# Reminder (COOKIE)
# -------------------------
def parse_hhmm(s: str) -> time:
    s = (s or "").strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return time(21, 30)
    hh, mm = int(m.group(1)), int(m.group(2))
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    return time(hh, mm)

def should_remind(now_dt: datetime, remind_t: time, window_min: int) -> bool:
    target = datetime.combine(now_dt.date(), remind_t, tzinfo=KST)
    delta_min = abs((now_dt - target).total_seconds()) / 60.0
    return delta_min <= float(window_min)


# -------------------------
# OpenAI (COOKIE, READY-SAFE)
# -------------------------
def openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되지 않았어요. pip install openai")
    if not api_key.strip():
        raise RuntimeError("OpenAI API Key가 비어 있어요.")
    return OpenAI(api_key=api_key.strip())

def ck_openai_key() -> str:
    return ck_get("openai_key", "").strip()

def ck_openai_model() -> str:
    m = ck_get("openai_model", "gpt-4o-mini").strip()
    return m if m else "gpt-4o-mini"

def effective_openai_key() -> str:
    sk = st.session_state.get("openai_api_key", "")
    return sk.strip() if sk and sk.strip() else ck_openai_key()

def effective_openai_model() -> str:
    sm = st.session_state.get("openai_model", "")
    return sm.strip() if sm and sm.strip() else ck_openai_model()

def set_ck_openai(api_key: str, model: str):
    ck_set("openai_key", (api_key or "").strip())
    ck_set("openai_model", (model or "gpt-4o-mini").strip())


# -------------------------
# Coaching prompt
# -------------------------
BASE_COACH_PROMPT = (
    "사용자의 계획 실패 이유 목록을 분석해 공통 원인을 3가지 이내로 분류하고, "
    "각 원인에 대해 실행 가능하고 현실적인 개선 조언을 제시해줘. "
    "앞에서 했던 실패가 2주 이상 반복된다면 창의적인 다른 조언을 제시해. "
    "톤은 비난 없이 코칭 중심으로 작성해."
)

COACH_SCHEMA = """
반드시 JSON만 출력해. (설명/마크다운 금지)
형식:
{
  "top_causes":[
    {
      "cause":"원인 카테고리(짧게)",
      "summary":"사용자 데이터(항목명/요일/패턴/원문 표현)를 반영한 2~4문장",
      "actionable_advice":[
        "이번 주에 바로 가능한 아주 구체적인 조언1",
        "조언2",
        "조언3"
      ],
      "creative_advice_when_repeated_2w":[
        "(2주+ 반복이면) 완전히 다른 접근의 창의적 대안1",
        "대안2"
      ]
    }
  ]
}
규칙:
- top_causes 최대 3개
- summary/advice는 반드시 '사용자 데이터'의 구체 요소를 최소 2개 이상 언급
- actionable_advice는 '작고 구체적'
- 비난/자책 유도 금지
- repeated_2w=true 항목이 하나라도 있으면 해당 원인에는 creative_advice_when_repeated_2w를 반드시 채워라
"""

def normalize_reason(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s가-힣]", "", t)
    return t

def repeated_reason_flags(df_fail: pd.DataFrame) -> Dict[str, bool]:
    if df_fail.empty:
        return {}
    x = df_fail.copy()
    x["task_date"] = pd.to_datetime(x["task_date"]).dt.date
    x["rnorm"] = x["fail_reason"].fillna("").map(normalize_reason)
    flags: Dict[str, bool] = {}
    for rnorm, g in x.groupby("rnorm"):
        if not rnorm:
            continue
        dates = sorted(g["task_date"].tolist())
        if len(dates) >= 2 and (dates[-1] - dates[0]).days >= 14:
            flags[rnorm] = True
    return flags

def compute_user_signals(user_id: str, days: int = 28) -> Dict[str, Any]:
    end = date.today()
    start = end - timedelta(days=days - 1)
    df = get_tasks_range(user_id, start, end)
    if df.empty:
        return {"has_data": False, "window_days": days, "window_start": start.isoformat(), "window_end": end.isoformat()}

    df = df.copy()
    df["task_date"] = pd.to_datetime(df["task_date"]).dt.date
    df["dow"] = df["task_date"].map(lambda d: d.weekday())
    df["is_fail"] = df["status"].eq("fail")
    df["is_success"] = df["status"].eq("success")

    fail_by_dow = (
        df[df["is_fail"]]
        .groupby("dow")["is_fail"]
        .sum()
        .reindex(range(7), fill_value=0)
        .to_dict()
    )
    fail_by_dow = {korean_dow(int(k)): int(v) for k, v in fail_by_dow.items()}

    top_failed = (
        df[df["is_fail"]]
        .groupby(["text", "source"])["is_fail"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_failed_items = [
        {"item": r["text"], "type": r["source"], "fail_count": int(r["is_fail"])}
        for _, r in top_failed.iterrows()
    ]

    reasons = df[df["is_fail"]]["fail_reason"].fillna("").map(lambda s: s.strip())
    top_reasons = reasons[reasons != ""].value_counts().head(10).to_dict()

    return {
        "has_data": True,
        "window_days": days,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "counts": {
            "total": int(len(df)),
            "success": int(df["is_success"].sum()),
            "fail": int(df["is_fail"].sum()),
            "todo": int((df["status"] == "todo").sum()),
        },
        "fail_by_dow": fail_by_dow,
        "top_failed_items": top_failed_items,
        "top_reasons": top_reasons,
    }

def llm_weekly_reason_analysis(api_key: str, model: str, reasons: List[str]) -> Dict[str, Any]:
    client = openai_client(api_key)
    prompt = f"""
너는 사용자의 실패 이유를 읽고, '이번 주' 관점에서 공통 원인을 최대 3개로 묶어 요약해.

실패 이유 목록:
{json.dumps(reasons, ensure_ascii=False)}

출력은 JSON만.
형식:
{{
  "groups":[
    {{"cause":"원인","description":"요약 1~2문장","examples":["예시1","예시2"],"estimated_count": 0}}
  ]
}}
규칙:
- groups 최대 3개
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": prompt}],
        temperature=0.35,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {"groups": []}

def llm_overall_coaching(api_key: str, model: str, fail_items: List[Dict[str, Any]], signals: Dict[str, Any]) -> Dict[str, Any]:
    client = openai_client(api_key)
    prompt = f"""
{BASE_COACH_PROMPT}

사용자 패턴 요약:
{json.dumps(signals, ensure_ascii=False, indent=2)}

실패 기록 샘플:
{json.dumps(fail_items, ensure_ascii=False, indent=2)}

{COACH_SCHEMA}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a supportive coaching assistant. Output must be valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.75,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {"top_causes": []}

def llm_chat(api_key: str, model: str, system_context: str, msgs: List[Dict[str, str]]) -> str:
    client = openai_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_context}] + msgs,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


# ============================================================
# A) Open-Meteo Weather (no key)
# ============================================================
WEATHER_CODE_KO = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "서리 안개",
    51: "이슬비(약)",
    53: "이슬비(중)",
    55: "이슬비(강)",
    56: "어는 이슬비(약)",
    57: "어는 이슬비(강)",
    61: "비(약)",
    63: "비(중)",
    65: "비(강)",
    66: "어는 비(약)",
    67: "어는 비(강)",
    71: "눈(약)",
    73: "눈(중)",
    75: "눈(강)",
    77: "싸라기눈",
    80: "소나기(약)",
    81: "소나기(중)",
    82: "소나기(강)",
    85: "눈 소나기(약)",
    86: "눈 소나기(강)",
    95: "뇌우",
    96: "뇌우(우박 약)",
    99: "뇌우(우박 강)",
}

@st.cache_data(ttl=60 * 60, show_spinner=False)
def geocode_city(city_name: str) -> Optional[Dict[str, Any]]:
    city_name = (city_name or "").strip()
    if not city_name:
        return None
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "language": "ko", "format": "json"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    results = js.get("results") or []
    return results[0] if results else None

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_daily_weather(lat: float, lon: float, d: date, tz: str = "Asia/Seoul") -> Optional[Dict[str, Any]]:
    if d <= date.today():
        base = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "timezone": tz,
        "start_date": d.isoformat(),
        "end_date": d.isoformat(),
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
    }
    r = requests.get(base, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()

    daily = js.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return None

    idx = 0
    code = (daily.get("weathercode") or [None])[idx]
    tmax = (daily.get("temperature_2m_max") or [None])[idx]
    tmin = (daily.get("temperature_2m_min") or [None])[idx]
    psum = (daily.get("precipitation_sum") or [None])[idx]
    pprob = (daily.get("precipitation_probability_max") or [None])[idx]

    return {
        "date": d.isoformat(),
        "weathercode": code,
        "desc": WEATHER_CODE_KO.get(int(code), f"code {code}") if code is not None else "—",
        "tmax": tmax,
        "tmin": tmin,
        "precip_sum": psum,
        "precip_prob": pprob,
    }

def weather_card(selected: date):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🌤️ Weather (Open-Meteo)")

    default_city = ck_get("city", "Seoul")
    city = st.text_input("도시/지역", value=default_city, key="weather_city_input", help="예: Seoul, Busan, Tokyo")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("도시 저장", use_container_width=True, key="weather_save_city"):
            ck_set("city", (city or "Seoul").strip())
            st.success("저장됐어요.")
            st.rerun()
    with colB:
        show = st.toggle("표시", value=(ck_get("weather_show", "true") == "true"), key="weather_show_toggle")
        ck_set("weather_show", "true" if show else "false")

    if (ck_get("weather_show", "true") != "true"):
        st.markdown("<div class='small'>날씨 표시가 꺼져 있어요.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        geo = geocode_city(city)
        if not geo:
            st.warning("도시를 찾지 못했어요. 다른 이름으로 시도해보세요.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        lat, lon = geo["latitude"], geo["longitude"]
        label = f"{geo.get('name','')} · {geo.get('country','')}"
        w = fetch_daily_weather(lat, lon, selected, tz="Asia/Seoul")
        if not w:
            st.info("해당 날짜의 날씨 데이터가 없어요.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.markdown(
            f"<span class='pill pill-strong'>{label}</span> "
            f"<span class='pill'>{selected.isoformat()} ({korean_dow(selected.weekday())})</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("상태", w["desc"])
        tmax = w["tmax"]
        tmin = w["tmin"]
        c2.metric("기온", f"{tmin:.0f}° ~ {tmax:.0f}°" if tmin is not None and tmax is not None else "—")
        pp = w.get("precip_prob")
        ps = w.get("precip_sum")
        c3.metric("강수", f"{pp}% / {ps}mm" if pp is not None and ps is not None else "—")

        st.caption("데이터 출처: Open-Meteo (Forecast/Archive)")
    except Exception as e:
        st.error(f"날씨 로딩 실패: {type(e).__name__}")
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# B) Weekly PDF report + Dashboard
# ============================================================
def try_register_korean_font() -> str:
    if st.session_state.get("__pdf_font_name"):
        return st.session_state["__pdf_font_name"]

    candidates = []
    candidates += glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/NotoSansKR*.ttf", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/NanumGothic*.ttf", recursive=True)

    for path in candidates:
        try:
            font_name = os.path.splitext(os.path.basename(path))[0]
            pdfmetrics.registerFont(TTFont(font_name, path))
            st.session_state["__pdf_font_name"] = font_name
            return font_name
        except Exception:
            continue

    st.session_state["__pdf_font_name"] = "Helvetica"
    return "Helvetica"

def failures_by_dow(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"dow": ["월", "화", "수", "목", "금", "토", "일"], "fail_count": [0] * 7})
    x = df.copy()
    x["task_date"] = pd.to_datetime(x["task_date"]).dt.date
    x = x[x["status"] == "fail"]
    rows = []
    for i in range(7):
        dname = korean_dow(i)
        rows.append({"dow": dname, "fail_count": int((x["task_date"].map(lambda d: d.weekday()) == i).sum())})
    return pd.DataFrame(rows)

def top_reasons(df: pd.DataFrame, topk: int = 8) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["reason", "count"])
    x = df[df["status"] == "fail"].copy()
    s = x["fail_reason"].fillna("").map(lambda v: str(v).strip())
    s = s[s != ""]
    vc = s.value_counts().head(topk)
    return pd.DataFrame({"reason": vc.index.tolist(), "count": vc.values.tolist()})

def weekly_reason_trend(user_id: str, weeks: int = 12, topk: int = 6) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=7 * weeks - 1)
    df = get_tasks_range(user_id, start, end)
    if df.empty:
        return pd.DataFrame(columns=["week", "reason", "count"])

    df = df.copy()
    df["task_date"] = pd.to_datetime(df["task_date"]).dt.date
    df = df[df["status"] == "fail"].copy()
    df["reason"] = df["fail_reason"].fillna("").map(lambda v: str(v).strip())
    df = df[df["reason"] != ""]
    if df.empty:
        return pd.DataFrame(columns=["week", "reason", "count"])

    top = df["reason"].value_counts().head(topk).index.tolist()
    df = df[df["reason"].isin(top)].copy()

    df["week"] = df["task_date"].map(lambda d: week_start(d).isoformat())
    out = (
        df.groupby(["week", "reason"]).size().reset_index(name="count")
        .sort_values(["week", "count"], ascending=[True, False])
    )
    return out

def make_matplotlib_bar_png(data: pd.DataFrame, xcol: str, ycol: str, title: str) -> bytes:
    fig = plt.figure(figsize=(6.2, 2.4), dpi=160)
    ax = fig.add_subplot(111)
    ax.bar(data[xcol].tolist(), data[ycol].tolist())
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def build_weekly_pdf_bytes(user_id: str, ws: date, city_label: str = "") -> bytes:
    we = ws + timedelta(days=6)
    df = get_tasks_range(user_id, ws, we)

    counts = {
        "total": int(len(df)),
        "success": int((df["status"] == "success").sum()) if not df.empty else 0,
        "fail": int((df["status"] == "fail").sum()) if not df.empty else 0,
        "todo": int((df["status"] == "todo").sum()) if not df.empty else 0,
    }

    font_name = try_register_korean_font()
    styles = getSampleStyleSheet()
    base = ParagraphStyle(name="Base", parent=styles["Normal"], fontName=font_name, fontSize=10.5, leading=14)
    h1 = ParagraphStyle(name="H1", parent=styles["Heading1"], fontName=font_name, fontSize=16, leading=20, spaceAfter=8)
    h2 = ParagraphStyle(name="H2", parent=styles["Heading2"], fontName=font_name, fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=6)
    small = ParagraphStyle(name="Small", parent=styles["Normal"], fontName=font_name, fontSize=9.5, leading=12, textColor=colors.HexColor("#444444"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="FAILOG Weekly Report",
    )

    story: List[Any] = []
    story.append(Paragraph("FAILOG · Weekly Report", h1))
    story.append(Paragraph(f"기간: {ws.isoformat()} ~ {we.isoformat()} (KST)", base))
    if city_label.strip():
        story.append(Paragraph(f"날씨 기준 도시: {city_label}", small))
    story.append(Paragraph(f"생성 시각: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')} (KST)", small))
    story.append(Spacer(1, 10))

    story.append(Paragraph("요약", h2))
    tdata = [["Total", "Success", "Fail", "Todo"], [str(counts["total"]), str(counts["success"]), str(counts["fail"]), str(counts["todo"])]]
    table = Table(tdata, colWidths=[35 * mm, 35 * mm, 35 * mm, 35 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3FF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2430")),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#BBD7F6")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("실패 분포(요일)", h2))
    dow_df = failures_by_dow(df)
    png1 = make_matplotlib_bar_png(dow_df, "dow", "fail_count", "Failures by Day of Week")
    story.append(RLImage(io.BytesIO(png1), width=170 * mm, height=58 * mm))
    story.append(Spacer(1, 8))

    story.append(Paragraph("실패 원인 TOP", h2))
    tr = top_reasons(df, topk=8)
    if tr.empty:
        story.append(Paragraph("이번 주에는 실패 원인 텍스트가 없어요.", base))
    else:
        rdata = [["원인", "횟수"]] + [[row["reason"], str(int(row["count"]))] for _, row in tr.iterrows()]
        rtable = Table(rdata, colWidths=[140 * mm, 25 * mm])
        rtable.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3FF")),
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.8),
                    ("ALIGN", (1, 1), (1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBD7F6")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(rtable)

    story.append(Spacer(1, 10))
    story.append(Paragraph("실패 목록", h2))
    if df.empty:
        story.append(Paragraph("이번 주에는 기록이 없어요.", base))
    else:
        f = df[df["status"] == "fail"].copy()
        if f.empty:
            story.append(Paragraph("이번 주에는 실패가 없어요. 🎉", base))
        else:
            f["task_date"] = pd.to_datetime(f["task_date"]).dt.date
            f = f.sort_values(["task_date", "id"], ascending=[True, True]).head(80)
            for _, row in f.iterrows():
                d0 = row["task_date"]
                dtxt = f"{d0.isoformat()} ({korean_dow(d0.weekday())})"
                task = str(row["text"])
                src = "Habit" if row["source"] == "habit" else "Plan"
                reason = str(row["fail_reason"] or "").strip()
                story.append(Paragraph(f"• {dtxt} · [{src}] {task}", base))
                if reason:
                    story.append(Paragraph(f"&nbsp;&nbsp;↳ 이유: {reason}", small))
                story.append(Spacer(1, 2))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# -------------------------
# Screens
# -------------------------
def screen_planner(user_id: str):
    st.markdown("## Planner")

    if st_autorefresh is not None:
        st_autorefresh(interval=60_000, key="auto_refresh_planner")

    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = date.today()

    selected = st.session_state["selected_date"]
    ws = week_start(selected)
    ensure_week_habit_tasks(user_id, ws)

    # Reminder settings in COOKIE (ready-safe via ck_get)
    en = (ck_get("rem_enabled", "true").lower() == "true")
    rt_str = ck_get("rem_time", "21:30")
    win_str = ck_get("rem_win", "15")
    remind_t = parse_hhmm(rt_str)
    try:
        win = int(win_str)
    except Exception:
        win = 15

    if en and should_remind(datetime.now(KST), remind_t, win):
        todos = count_today_todos(user_id)
        if todos > 0:
            st.toast(f"⏰ 아직 체크하지 않은 항목이 {todos}개 있어요", icon="⏰")

    left, right = st.columns([1.05, 1.95], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Month")

        y, m = selected.year, selected.month
        nav = st.columns([1, 2, 1])
        with nav[0]:
            if st.button("◀", use_container_width=True, key="m_prev"):
                if m == 1:
                    y -= 1
                    m = 12
                else:
                    m -= 1
                st.session_state["selected_date"] = date(y, m, 1)
                st.rerun()
        with nav[1]:
            st.markdown(
                f"<div style='text-align:center; font-weight:700; font-size:1.05rem;'>{y}.{m:02d}</div>",
                unsafe_allow_html=True,
            )
        with nav[2]:
            if st.button("▶", use_container_width=True, key="m_next"):
                if m == 12:
                    y += 1
                    m = 1
                else:
                    m += 1
                st.session_state["selected_date"] = date(y, m, 1)
                st.rerun()

        st.markdown(
            "<div style='display:grid; grid-template-columns: repeat(7, 1fr); gap:6px; font-size:0.78rem; opacity:0.75; margin-top:8px;'>"
            + "".join([f"<div style='text-align:center;'>{k}</div>" for k in ["월", "화", "수", "목", "금", "토", "일"]])
            + "</div>",
            unsafe_allow_html=True,
        )

        grid = month_grid(y, m)
        today = date.today()
        for row in grid:
            cols = st.columns(7, gap="small")
            for i, d in enumerate(row):
                if d is None:
                    cols[i].markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
                    continue
                label = f"{d.day}"
                if d == today:
                    label = f"•{d.day}"
                if cols[i].button(label, key=f"cal_{d.isoformat()}", use_container_width=True):
                    st.session_state["selected_date"] = d
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("알림 설정", expanded=False):
            en_ui = st.toggle("리마인더 켜기", value=en, key="rem_en_ui")
            t_ui = st.text_input("시간(HH:MM)", value=rt_str, key="rem_t_ui")
            w_ui = st.number_input("허용 오차(분)", min_value=1, max_value=120, value=win, key="rem_w_ui")
            if st.button("저장", use_container_width=True, key="rem_save"):
                ck_set("rem_enabled", "true" if en_ui else "false")
                ck_set("rem_time", (t_ui or "21:30"))
                ck_set("rem_win", str(int(w_ui)))
                st.success("저장됐어요.")

        weather_card(selected)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Current Week")
        st.markdown(
            f"<span class='pill pill-strong'>Week</span> "
            f"<span class='pill'>{ws.isoformat()} ~ {(ws+timedelta(days=6)).isoformat()}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        wcols = st.columns(7, gap="small")
        days = week_days(ws)
        for i, d in enumerate(days):
            label = f"{korean_dow(i)}\n{d.day}"
            if wcols[i].button(label, key=f"w_{d.isoformat()}", use_container_width=True):
                st.session_state["selected_date"] = d
                st.rerun()
            if d == selected:
                wcols[i].caption("선택")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"#### {selected.isoformat()} ({korean_dow(selected.weekday())})")

        with st.form("plan_add_form", clear_on_submit=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                plan_text = st.text_input("계획 추가(1회성)", placeholder="예: 독서 10분 / 이메일 정리", key="plan_text_input")
            with c2:
                submitted = st.form_submit_button("추가", use_container_width=True)
            if submitted:
                add_plan_task(user_id, selected, plan_text)
                st.rerun()

        with st.expander("습관(반복) 관리", expanded=False):
            with st.form("habit_add_form", clear_on_submit=True):
                hc1, hc2 = st.columns([3, 2])
                with hc1:
                    habit_title = st.text_input("습관 이름", placeholder="예: 운동 10분", key="habit_title_input")
                with hc2:
                    dow_labels = [korean_dow(i) for i in range(7)]
                    picked = st.multiselect(
                        "반복 요일",
                        options=list(range(7)),
                        format_func=lambda x: dow_labels[x],
                        default=[0, 1, 2, 3, 4],
                        key="habit_dow_input",
                    )
                habit_submit = st.form_submit_button("습관 저장", use_container_width=True)

                if habit_submit:
                    add_habit(user_id, habit_title, picked)
                    ensure_week_habit_tasks(user_id, ws)
                    st.success("습관을 저장했어요.")
                    st.rerun()

            hdf = list_habits(user_id, active_only=False)
            if hdf.empty:
                st.markdown("<div class='small'>아직 습관이 없어요.</div>", unsafe_allow_html=True)
            else:
                for _, h in hdf.iterrows():
                    hid = int(h["id"])
                    mask = str(h["dow_mask"] or "0000000")
                    days_txt = " ".join([korean_dow(i) for i in range(7) if mask[i] == "1"]) or "—"
                    active = int(h["active"]) == 1

                    a, b, c = st.columns([6, 1, 1], gap="small")
                    with a:
                        st.write(f"• {h['title']}  ·  {days_txt}")
                    with b:
                        if st.button("ON" if active else "OFF", key=f"hab_toggle_{hid}", use_container_width=True):
                            set_habit_active(user_id, hid, not active)
                            ensure_week_habit_tasks(user_id, ws)
                            st.rerun()
                    with c:
                        if st.button("삭제", key=f"hab_del_{hid}", use_container_width=True):
                            delete_habit(user_id, hid)
                            st.success("습관을 삭제했어요.")
                            st.rerun()

        df = list_tasks_for_date(user_id, selected)
        if df.empty:
            st.markdown("<div class='small'>아직 항목이 없어요.</div>", unsafe_allow_html=True)
        else:
            for _, r in df.iterrows():
                tid = int(r["id"])
                src = r["source"]
                status = r["status"]
                text = r["text"]
                reason = r["fail_reason"] or ""

                status_icon = {"todo": "⏳", "success": "✅", "fail": "❌"}.get(status, "⏳")
                badge = "Habit" if src == "habit" else "Plan"

                st.markdown("<div class='task'>", unsafe_allow_html=True)
                top = st.columns([6, 1.2, 1.2, 1.0], gap="small")

                with top[0]:
                    st.markdown(f"**{status_icon} {text}**  <span class='pill'>{badge}</span>", unsafe_allow_html=True)
                    if status == "fail":
                        st.caption(f"실패 원인: {reason}")

                with top[1]:
                    if st.button("성공", key=f"s_{tid}", use_container_width=True, type="primary"):
                        update_task_status(user_id, tid, "success")
                        st.session_state.pop(f"show_fail_{tid}", None)
                        st.rerun()

                with top[2]:
                    if st.button("실패", key=f"f_{tid}", use_container_width=True):
                        st.session_state[f"show_fail_{tid}"] = True

                with top[3]:
                    if st.button("삭제", key=f"del_{tid}", use_container_width=True):
                        delete_task(user_id, tid)
                        st.session_state.pop(f"show_fail_{tid}", None)
                        st.rerun()

                if st.session_state.get(f"show_fail_{tid}", False):
                    reason_in = st.text_input("실패 원인(한 문장)", value=reason, key=f"r_{tid}")
                    a, b = st.columns([1, 4], gap="small")
                    with a:
                        if st.button("저장", key=f"save_fail_{tid}", use_container_width=True, type="primary"):
                            update_task_fail(user_id, tid, reason_in)
                            st.session_state[f"show_fail_{tid}"] = False
                            st.rerun()
                    with b:
                        st.caption("짧아도 좋아요. ‘무슨 조건 때문에’가 핵심이에요.")

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def screen_failures(user_id: str):
    st.markdown("## Failure Report")

    if "fail_week_offset" not in st.session_state:
        st.session_state["fail_week_offset"] = 0

    offset = int(st.session_state["fail_week_offset"])
    base = date.today() - timedelta(days=7 * offset)
    ws = week_start(base)
    we = ws + timedelta(days=6)

    nav = st.columns([1, 3, 1])
    with nav[0]:
        if st.button("〈", use_container_width=True, key="fw_prev"):
            st.session_state["fail_week_offset"] += 1
            st.rerun()
    with nav[1]:
        st.markdown(f"<div style='text-align:center; font-weight:700;'>{ws.isoformat()} ~ {we.isoformat()}</div>", unsafe_allow_html=True)
    with nav[2]:
        if st.button("〉", use_container_width=True, key="fw_next", disabled=(offset == 0)):
            st.session_state["fail_week_offset"] = max(0, offset - 1)
            st.rerun()

    df = get_tasks_range(user_id, ws, we)
    if df.empty:
        st.info("이 주에는 기록이 없어요.")
        return

    df = df.copy()
    df["task_date"] = pd.to_datetime(df["task_date"]).dt.date
    fails = df[df["status"] == "fail"].copy()

    tab1, tab2, tab3 = st.tabs(["대시보드", "주간 분석/코칭", "PDF 리포트"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Dashboard (최근 트렌드)")

        colA, colB = st.columns([1.2, 1.0])
        with colA:
            weeks = st.slider("기간(주)", min_value=4, max_value=24, value=12, step=1, key="dash_weeks")
        with colB:
            topk = st.slider("원인 TOP", min_value=3, max_value=10, value=6, step=1, key="dash_topk")

        dow_df = failures_by_dow(df)
        c = (
            alt.Chart(dow_df)
            .mark_bar()
            .encode(
                x=alt.X("dow:N", sort=["월", "화", "수", "목", "금", "토", "일"], title=None),
                y=alt.Y("fail_count:Q", title=None),
                tooltip=["dow", "fail_count"],
            )
            .properties(height=160)
        )
        st.markdown("**이번 주 실패(요일 분포)**")
        st.altair_chart(c, use_container_width=True)

        trend = weekly_reason_trend(user_id, weeks=weeks, topk=topk)
        if trend.empty:
            st.info("최근 기간에 ‘실패 원인 텍스트’가 부족해서 트렌드를 만들 수 없어요.")
        else:
            st.markdown("**실패 원인 트렌드(주별)**")
            c2 = (
                alt.Chart(trend)
                .mark_line(point=True)
                .encode(
                    x=alt.X("week:N", title=None, sort=sorted(trend["week"].unique().tolist())),
                    y=alt.Y("count:Q", title=None),
                    color=alt.Color("reason:N", title="원인"),
                    tooltip=["week", "reason", "count"],
                )
                .properties(height=240)
            )
            st.altair_chart(c2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 주간 실패 차트")
        days = week_days(ws)
        chart_rows = [{"dow": korean_dow(d.weekday()), "fail_count": int((fails["task_date"] == d).sum())} for d in days]
        chart_df = pd.DataFrame(chart_rows)

        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("dow:N", sort=["월", "화", "수", "목", "금", "토", "일"], title=None),
                y=alt.Y("fail_count:Q", title=None),
                tooltip=["dow", "fail_count"],
            )
            .properties(height=155)
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        api_key = effective_openai_key()
        model = effective_openai_model()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 원인 주간 분석")

        weekly_reasons = [r for r in fails["fail_reason"].fillna("").tolist() if str(r).strip()]
        if not api_key:
            st.info("OpenAI 키가 설정되면 분석이 표시돼요. (하단에서 키 입력)")
        elif len(weekly_reasons) == 0:
            st.write("이번 주에는 실패 원인 입력이 아직 없어요.")
        else:
            if st.button("분석 생성/갱신", use_container_width=True, key="weekly_analyze", type="primary"):
                try:
                    st.session_state["weekly_analysis"] = llm_weekly_reason_analysis(api_key, model, weekly_reasons)
                except Exception as e:
                    st.error(f"분석 생성 실패: {type(e).__name__}")

            analysis = st.session_state.get("weekly_analysis")
            if analysis and isinstance(analysis, dict):
                groups = analysis.get("groups", []) or []
                for g in groups[:3]:
                    with st.container(border=True):
                        st.markdown(f"**{g.get('cause','원인')}**  ·  ~{g.get('estimated_count',0)}회")
                        st.write(g.get("description", ""))
                        for s in (g.get("examples") or [])[:3]:
                            st.write(f"- {s}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 맞춤형 AI코칭")

        if not api_key:
            st.info("OpenAI 키가 설정되면 코칭/챗봇이 표시돼요. (하단에서 키 입력)")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        all_fail = get_all_failures(user_id, limit=350)
        if all_fail.empty:
            st.write("아직 실패 데이터가 없어요.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        flags = repeated_reason_flags(all_fail)
        items: List[Dict[str, Any]] = []
        for _, r in all_fail.head(90).iterrows():
            reason = str(r["fail_reason"] or "")
            rnorm = normalize_reason(reason)
            items.append(
                {
                    "date": str(r["task_date"]),
                    "task": str(r["text"]),
                    "type": str(r["source"]),
                    "reason": reason,
                    "repeated_2w": bool(flags.get(rnorm, False)),
                }
            )

        signals = compute_user_signals(user_id, days=28)

        if st.button("코칭 생성/갱신", use_container_width=True, key="overall_coach_btn", type="primary"):
            try:
                st.session_state["overall_coach"] = llm_overall_coaching(api_key, model, items, signals)
            except Exception as e:
                st.error(f"코칭 생성 실패: {type(e).__name__}")

        coach = st.session_state.get("overall_coach")
        if coach and isinstance(coach, dict):
            top = coach.get("top_causes", []) or []
            if not top:
                st.caption("코칭 결과가 비어 있어요. 다시 생성해보세요.")
            for i, c in enumerate(top[:3], start=1):
                with st.container(border=True):
                    st.markdown(f"**{i}) {c.get('cause','원인')}**")
                    st.write(c.get("summary", ""))
                    st.markdown("**실행 조언**")
                    for tip in (c.get("actionable_advice") or [])[:3]:
                        st.write(f"- {tip}")
                    creative = c.get("creative_advice_when_repeated_2w") or []
                    if creative:
                        st.markdown("**2주+ 반복이면: 창의적 대안**")
                        for tip in creative[:3]:
                            st.write(f"- {tip}")

        st.markdown("<hr/>", unsafe_allow_html=True)

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        for m in st.session_state["chat_messages"]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user_msg = st.chat_input("메시지를 입력하세요")
        if user_msg:
            st.session_state["chat_messages"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            end = date.today()
            start = end - timedelta(days=13)
            last14 = get_tasks_range(user_id, start, end)
            last14_fail = last14[last14["status"] == "fail"]
            top_reasons_14 = (
                last14_fail["fail_reason"].fillna("").map(lambda s: s.strip()).value_counts().head(6).to_dict()
                if not last14_fail.empty
                else {}
            )

            system_context = f"""
너는 FAILOG의 코칭 챗봇이야.
원칙:
- 비난/자책 유도 금지, 코칭 톤
- 실행 가능하고 현실적인 조언(작게, 구체적으로)
- 사용자의 패턴(요일/항목/plan-habit 특성/연속성)을 근거로 개인화
- 반복 실패(2주+)가 보이면, 다른 각도의 창의적 대안을 최소 1개 포함

사용자 요약:
- 최근 14일 실패 이유 상위: {json.dumps(top_reasons_14, ensure_ascii=False)}
- 최근 28일 패턴 요약: {json.dumps(signals, ensure_ascii=False)}
- 누적 실패 샘플(최근 8개): {json.dumps(items[:8], ensure_ascii=False)}
""".strip()

            try:
                assistant_text = llm_chat(api_key, model, system_context, st.session_state["chat_messages"][-14:])
            except Exception as e:
                assistant_text = f"(OpenAI 호출 오류: {type(e).__name__}) 키/모델을 확인해 주세요."

            st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.write(assistant_text)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Weekly PDF 리포트")
        st.caption("이 탭은 OpenAI 없이도 동작해요.")

        city = ck_get("city", "").strip()
        city_label = ""
        try:
            if city:
                g = geocode_city(city)
                if g:
                    city_label = f"{g.get('name','')} · {g.get('country','')}"
        except Exception:
            city_label = city

        c1, c2, c3 = st.columns([1.1, 1.1, 2.2])
        with c1:
            target_ws = st.date_input("주 시작(월)", value=ws, key="pdf_ws")
            target_ws = week_start(target_ws)
        with c2:
            filename = st.text_input("파일명", value=f"failog_week_{target_ws.isoformat()}.pdf", key="pdf_name")
        with c3:
            st.write("")
            st.write("")
            gen = st.button("PDF 생성", use_container_width=True, type="primary", key="pdf_gen")

        if gen:
            with st.spinner("PDF 생성 중..."):
                try:
                    pdf_bytes = build_weekly_pdf_bytes(user_id, target_ws, city_label=city_label)
                    st.session_state["__latest_pdf__"] = (filename, pdf_bytes)
                    st.success("PDF가 생성됐어요.")
                except Exception as e:
                    st.error(f"PDF 생성 실패: {type(e).__name__}")

        if st.session_state.get("__latest_pdf__"):
            fn, bts = st.session_state["__latest_pdf__"]
            st.download_button("PDF 다운로드", data=bts, file_name=fn, mime="application/pdf", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Bottom OpenAI panel (COOKIE, READY-SAFE)
# -------------------------
def render_openai_bottom_panel():
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 🔑 OpenAI 설정 (쿠키 저장 옵션)")

    default_key = ck_openai_key()
    default_model = ck_openai_model()

    col1, col2, col3 = st.columns([3.0, 1.6, 1.4])
    with col1:
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", "") or default_key,
            type="password",
            placeholder="sk-...",
            key="bottom_openai_key",
        )
    with col2:
        model = st.text_input(
            "모델",
            value=st.session_state.get("openai_model", "") or default_model,
            key="bottom_openai_model",
        )
    with col3:
        save_default = (default_key.strip() != "")
        save = st.toggle("쿠키 저장", value=save_default, help="같은 브라우저에서 유지돼요.", key="bottom_openai_save")

    a, b, c = st.columns([1, 1, 3])
    with a:
        if st.button("적용", use_container_width=True, key="bottom_apply", type="primary"):
            st.session_state["openai_api_key"] = (api_key or "").strip()
            st.session_state["openai_model"] = (model or "gpt-4o-mini").strip()

            if save:
                set_ck_openai(api_key or "", model or "gpt-4o-mini")
            else:
                ck_del("openai_key")
                ck_set("openai_model", (model or "gpt-4o-mini").strip())

            st.success("적용됐어요.")
    with b:
        if st.button("저장값 삭제", use_container_width=True, key="bottom_clear"):
            ck_del("openai_key")
            ck_del("openai_model")
            st.success("쿠키 저장값을 삭제했어요.")
            st.rerun()
    with c:
        st.caption("쿠키 저장을 켜면 같은 브라우저에서는 새로고침/재접속해도 유지돼요. (공유 PC에서는 끄세요)")


# -------------------------
# Top nav
# -------------------------
def top_nav():
    if "screen" not in st.session_state:
        st.session_state["screen"] = "planner"

    c1, c2, _ = st.columns([1.2, 1.8, 6])
    with c1:
        if st.button(" Planner", use_container_width=True, key="nav_plan",
                     type="primary" if st.session_state["screen"] == "planner" else "secondary"):
            st.session_state["screen"] = "planner"
            st.rerun()
    with c2:
        if st.button("Failure Report", use_container_width=True, key="nav_fail",
                     type="primary" if st.session_state["screen"] == "fail" else "secondary"):
            st.session_state["screen"] = "fail"
            st.rerun()

    st.write("")
    return st.session_state["screen"]


# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(page_title="FAILOG", page_icon="🧊", layout="wide")
    inject_css()
    init_db()

    # ❌ cookies()를 여기서 강제 호출하지 않음 (not-ready 문제를 앞당길 수 있음)

    user_id = get_or_create_user_id()

    st.markdown("# FAILOG")
    st.markdown(
        "<div class='small'>실패를 성공으로! 계획과 습관의 실패를 기록하고 맞춤형 코칭을 받아보자</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    screen = top_nav()
    if screen == "planner":
        screen_planner(user_id)
    else:
        screen_failures(user_id)

    render_openai_bottom_panel()


if __name__ == "__main__":
    main()
