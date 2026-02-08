import io
import json
import os
import re
import sqlite3
import uuid
from datetime import date, datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from zoneinfo import ZoneInfo

# Optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Optional cookie manager (prefs only; NOT used for user_id)
try:
    import extra_streamlit_components as stx
except Exception:
    stx = None

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# PDF (ReportLab)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet


# -------------------------
# Constants
# -------------------------
KST = ZoneInfo("Asia/Seoul")
DB_PATH = "planner.db"

# Theme / colors
ACCENT_BLUE = "#A0C4F2"
TEXT_DARK = "#1f2430"

# Dashboard fixed params (per your request)
DASH_TREND_WEEKS = 8
DASH_TOPK = 6
CATEGORY_MAX = 7
CATEGORY_MAP_WINDOW_WEEKS = 12

# PDF font
FONTS_DIR = "fonts"
KOREAN_FONT_PATH = os.path.join(FONTS_DIR, "NanumGothic-Regular.ttf")
KOREAN_FONT_NAME = "NanumGothicRegular"
NANUM_TTF_URL = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"

# Consent (privacy/AI usage)
CONSENT_COOKIE_KEY = "failog_ai_consent"  # "true"/"false"


# ============================================================
# UI / CSS
# ============================================================
def inject_css():
    st.markdown(
        f"""
<style>
/* Layout */
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
hr {{
  margin: 1.1rem 0;
  border: none;
  border-top: 1px solid rgba(160,196,242,0.35);
}}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
  border-radius: 14px !important;
  border: 1px solid rgba(160,196,242,0.55) !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
  outline: none !important;
  box-shadow: 0 0 0 4px rgba(160,196,242,0.35) !important;
  border-color: rgba(160,196,242,0.95) !important;
}}

/* Hero title */
.failog-hero {{
  border: 1px solid rgba(160,196,242,0.60);
  border-radius: 22px;
  padding: 18px 18px;
  background: rgba(255,255,255,0.92);
  box-shadow: 0 12px 34px rgba(160,196,242,0.14);
}}
.failog-title {{
  font-size: 2.55rem;
  font-weight: 900;
  letter-spacing: -0.02em;
  margin: 0;
  line-height: 1.08;
  color: {TEXT_DARK};
}}
.failog-sub {{
  margin-top: 6px;
  color: rgba(31,36,48,0.66);
  font-size: 1.02rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
<div class="failog-hero">
  <div class="failog-title">FAILOG</div>
  <div class="failog-sub">실패를 성공으로 — 계획과 습관의 실패를 기록하고, 패턴을 이해하고, 다음 주를 설계해요.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")


# ============================================================
# URL-fixed user_id
# ============================================================
def get_or_create_user_id() -> str:
    qp = st.query_params
    uid = (qp.get("uid", "") or "").strip()
    if uid:
        st.session_state["user_id"] = uid
        return uid

    new_uid = str(uuid.uuid4())
    st.query_params["uid"] = new_uid
    st.session_state["user_id"] = new_uid
    st.rerun()


# ============================================================
# Cookies (prefs only; best-effort)
# ============================================================
def cookie_mgr():
    if stx is None:
        return None
    if "x_cookie_mgr" not in st.session_state:
        st.session_state["x_cookie_mgr"] = stx.CookieManager()
    return st.session_state["x_cookie_mgr"]


def ck_get(key: str, default: str = "") -> str:
    cm = cookie_mgr()
    if cm is None:
        return default
    try:
        v = cm.get(key)
        return default if v is None else str(v)
    except Exception:
        return default


def ck_set(key: str, value: str, expires_days: int = 3650):
    cm = cookie_mgr()
    if cm is None:
        return
    v = "" if value is None else str(value)
    try:
        # Some versions support expires_at_days
        if hasattr(cm, "set") and "expires_at_days" in cm.set.__code__.co_varnames:
            cm.set(key, v, expires_at_days=int(expires_days))
        else:
            cm.set(key, v)
    except Exception:
        try:
            cm.set(key, v)
        except Exception:
            pass


def ck_del(key: str):
    cm = cookie_mgr()
    if cm is None:
        return
    for fn in ("delete", "remove", "delete_cookie"):
        if hasattr(cm, fn):
            try:
                getattr(cm, fn)(key)
                return
            except Exception:
                pass
    try:
        cm.set(key, "")
    except Exception:
        pass


# ============================================================
# Consent helpers
# ============================================================
def consent_value() -> bool:
    # 1) session_state first
    if "ai_consent" in st.session_state:
        return bool(st.session_state["ai_consent"])
    # 2) cookie best-effort
    v = ck_get(CONSENT_COOKIE_KEY, "").strip().lower()
    if v in ("true", "1", "yes", "y"):
        st.session_state["ai_consent"] = True
        return True
    if v in ("false", "0", "no", "n"):
        st.session_state["ai_consent"] = False
        return False
    # default: not consented
    st.session_state["ai_consent"] = False
    return False


def set_consent(v: bool):
    st.session_state["ai_consent"] = bool(v)
    ck_set(CONSENT_COOKIE_KEY, "true" if v else "false")


# ============================================================
# DB
# ============================================================
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

    # Category map cache (per user)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS category_maps (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          created_at TEXT NOT NULL,
          window_weeks INTEGER NOT NULL,
          max_categories INTEGER NOT NULL,
          payload_json TEXT NOT NULL
        );
        """
    )

    c.commit()
    c.close()


# ============================================================
# Date helpers
# ============================================================
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


# ============================================================
# Habits / Tasks
# ============================================================
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


# ============================================================
# Reminder (prefs)
# ============================================================
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


# ============================================================
# OpenAI (for coaching + categorization)
# ============================================================
def openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되지 않았어요. pip install openai")
    if not api_key.strip():
        raise RuntimeError("OpenAI API Key가 비어 있어요.")
    return OpenAI(api_key=api_key.strip())


def prefs_openai_key() -> str:
    return ck_get("failog_openai_key", "").strip()


def prefs_openai_model() -> str:
    m = ck_get("failog_openai_model", "gpt-4o-mini").strip()
    return m if m else "gpt-4o-mini"


def effective_openai_key() -> str:
    sk = st.session_state.get("openai_api_key", "")
    return sk.strip() if sk and sk.strip() else prefs_openai_key()


def effective_openai_model() -> str:
    sm = st.session_state.get("openai_model", "")
    return sm.strip() if sm and sm.strip() else prefs_openai_model()


def set_prefs_openai(api_key: str, model: str):
    ck_set("failog_openai_key", (api_key or "").strip())
    ck_set("failog_openai_model", (model or "gpt-4o-mini").strip())


# ============================================================
# Coaching prompts
# ============================================================
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
        return {
            "has_data": False,
            "window_days": days,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
        }

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
# OpenAI Categorization (Dashboard)
# ============================================================
CATEGORY_SCHEMA = """
반드시 JSON만 출력.
형식:
{
  "categories": [
    {
      "name": "카테고리명(짧게)",
      "definition": "이 카테고리에 포함되는 실패 원인의 특징(1문장)",
      "examples": ["원문 예시1","원문 예시2"]
    }
  ],
  "mapping": {
    "원문 실패원인": "카테고리명",
    "또다른 원문": "카테고리명"
  }
}
규칙:
- categories 최대 __MAX_CATEGORIES__개
- mapping의 키는 반드시 입력 원문 목록에 존재하는 문자열 그대로
- mapping 값은 categories[].name 중 하나
- 애매하면 '기타' 카테고리를 하나 포함해도 됨 (그 경우 name='기타')
""".strip()


def list_recent_failure_reasons(user_id: str, weeks: int) -> List[str]:
    end = date.today()
    start = end - timedelta(days=7 * weeks - 1)
    df = get_tasks_range(user_id, start, end)
    if df.empty:
        return []
    f = df[df["status"] == "fail"].copy()
    if f.empty:
        return []
    reasons = f["fail_reason"].fillna("").map(lambda v: str(v).strip())
    reasons = reasons[reasons != ""]
    if reasons.empty:
        return []
    vc = reasons.value_counts()
    return vc.index.tolist()


def llm_build_category_map(api_key: str, model: str, reasons: List[str], max_categories: int) -> Dict[str, Any]:
    client = openai_client(api_key)

    reasons_limited = reasons[:120]
    schema = CATEGORY_SCHEMA.replace("__MAX_CATEGORIES__", str(max_categories))

    prompt = f"""
너는 사용자의 '실패 원인' 텍스트들을 비슷한 것끼리 묶어 카테고리로 분류해.
목표:
- 사용자 표현이 다양해도 의미가 비슷하면 같은 카테고리로 묶기
- 카테고리명은 짧고 직관적으로
- 전체 카테고리는 최대 {max_categories}개
- 가능한 한 '기타'는 최소화하되, 정말 애매하면 '기타'를 포함해도 됨

실패 원인 원문 목록:
{json.dumps(reasons_limited, ensure_ascii=False, indent=2)}

출력 스키마:
{schema}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {"categories": [], "mapping": {}}


def db_get_latest_category_map(user_id: str) -> Optional[Dict[str, Any]]:
    c = conn()
    row = c.execute(
        """
        SELECT payload_json
        FROM category_maps
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    c.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def db_save_category_map(user_id: str, payload: Dict[str, Any], window_weeks: int, max_categories: int):
    c = conn()
    c.execute(
        """
        INSERT INTO category_maps(user_id, created_at, window_weeks, max_categories, payload_json)
        VALUES (?,?,?,?,?)
        """,
        (user_id, now_iso(), int(window_weeks), int(max_categories), json.dumps(payload, ensure_ascii=False)),
    )
    c.commit()
    c.close()


def get_or_build_category_map(
    user_id: str, api_key: str, model: str, force_refresh: bool = False
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not force_refresh:
        cached = db_get_latest_category_map(user_id)
        if cached and isinstance(cached, dict) and isinstance(cached.get("mapping", None), dict) and cached.get("mapping"):
            return cached, "캐시된 카테고리 맵을 사용 중"

    reasons = list_recent_failure_reasons(user_id, weeks=CATEGORY_MAP_WINDOW_WEEKS)
    if len(reasons) < 4:
        return None, "최근 12주 실패 원인 텍스트가 부족해요(최소 4개 필요)."

    payload = llm_build_category_map(api_key, model, reasons, max_categories=CATEGORY_MAX)

    mapping = payload.get("mapping", {}) if isinstance(payload, dict) else {}
    if not isinstance(mapping, dict) or len(mapping) == 0:
        return None, "카테고리 맵 생성 결과가 비어 있어요. 다시 시도해 주세요."

    # save and return
    db_save_category_map(user_id, payload, window_weeks=CATEGORY_MAP_WINDOW_WEEKS, max_categories=CATEGORY_MAX)
    return payload, "카테고리 맵을 새로 만들었어요"


def apply_category_mapping(df_fail: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    x = df_fail.copy()
    x["reason_raw"] = x["fail_reason"].fillna("").map(lambda v: str(v).strip())
    x["category"] = x["reason_raw"].map(lambda r: mapping.get(r, "기타"))
    x.loc[x["reason_raw"] == "", "category"] = "기타"
    return x


def weekly_category_trend(user_id: str, weeks: int, topk: int, mapping: Dict[str, str]) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=7 * weeks - 1)
    df = get_tasks_range(user_id, start, end)
    if df.empty:
        return pd.DataFrame(columns=["week", "category", "count"])

    df = df.copy()
    df["task_date"] = pd.to_datetime(df["task_date"]).dt.date
    df = df[df["status"] == "fail"].copy()
    if df.empty:
        return pd.DataFrame(columns=["week", "category", "count"])

    df = apply_category_mapping(df, mapping)
    df["week"] = df["task_date"].map(lambda d: week_start(d).isoformat())

    totals = df.groupby("category").size().sort_values(ascending=False)
    top_categories = totals.head(topk).index.tolist()

    df = df[df["category"].isin(top_categories)].copy()
    out = df.groupby(["week", "category"]).size().reset_index(name="count")
    out["count"] = out["count"].astype(int)

    weeks_sorted = sorted(df["week"].unique().tolist())
    all_rows = []
    for w in weeks_sorted:
        for cat in top_categories:
            sub = out[(out["week"] == w) & (out["category"] == cat)]
            cnt = int(sub["count"].iloc[0]) if not sub.empty else 0
            all_rows.append({"week": w, "category": cat, "count": cnt})
    return pd.DataFrame(all_rows)


# ============================================================
# Open-Meteo Weather (no key)
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
    base = "https://archive-api.open-meteo.com/v1/archive" if d <= date.today() else "https://api.open-meteo.com/v1/forecast"
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

    default_city = ck_get("failog_city", "Seoul")
    city = st.text_input("도시/지역", value=default_city, key="weather_city_input", help="예: Seoul, Busan, Tokyo")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("도시 저장", use_container_width=True, key="weather_save_city"):
            ck_set("failog_city", (city or "Seoul").strip())
            st.success("저장됐어요.")
            st.rerun()
    with colB:
        show = st.toggle("표시", value=(ck_get("failog_weather_show", "true") == "true"), key="weather_show_toggle")
        ck_set("failog_weather_show", "true" if show else "false")

    if ck_get("failog_weather_show", "true") != "true":
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
        st.caption("데이터 출처: Open-Meteo")
    except Exception as e:
        st.error(f"날씨 로딩 실패: {type(e).__name__}")
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PDF: Korean font embedding
# ============================================================
def ensure_korean_font_downloaded() -> bool:
    try:
        os.makedirs(FONTS_DIR, exist_ok=True)
        if os.path.exists(KOREAN_FONT_PATH) and os.path.getsize(KOREAN_FONT_PATH) > 50_000:
            return True

        r = requests.get(NANUM_TTF_URL, timeout=20)
        r.raise_for_status()
        with open(KOREAN_FONT_PATH, "wb") as f:
            f.write(r.content)
        return os.path.exists(KOREAN_FONT_PATH) and os.path.getsize(KOREAN_FONT_PATH) > 50_000
    except Exception:
        return False


def register_korean_font() -> str:
    if st.session_state.get("__pdf_font_registered__", False):
        return st.session_state.get("__pdf_font_name__", "Helvetica")

    ok = ensure_korean_font_downloaded()
    if ok:
        try:
            pdfmetrics.registerFont(TTFont(KOREAN_FONT_NAME, KOREAN_FONT_PATH))
            st.session_state["__pdf_font_registered__"] = True
            st.session_state["__pdf_font_name__"] = KOREAN_FONT_NAME
            return KOREAN_FONT_NAME
        except Exception:
            pass

    st.session_state["__pdf_font_registered__"] = True
    st.session_state["__pdf_font_name__"] = "Helvetica"
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

    font_name = register_korean_font()
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
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(TEXT_DARK)),
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


# ============================================================
# Screens
# ============================================================
def screen_planner(user_id: str):
    st.markdown("## Planner")

    if st_autorefresh is not None:
        st_autorefresh(interval=60_000, key="auto_refresh_planner")

    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = date.today()

    selected = st.session_state["selected_date"]
    ws = week_start(selected)
    ensure_week_habit_tasks(user_id, ws)

    # Reminder settings (prefs)
    en = (ck_get("failog_rem_enabled", "true").lower() == "true")
    rt_str = ck_get("failog_rem_time", "21:30")
    win_str = ck_get("failog_rem_win", "15")
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
                f"<div style='text-align:center; font-weight:800; font-size:1.05rem;'>{y}.{m:02d}</div>",
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
                ck_set("failog_rem_enabled", "true" if en_ui else "false")
                ck_set("failog_rem_time", (t_ui or "21:30"))
                ck_set("failog_rem_win", str(int(w_ui)))
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
                    if st.button("성공", key=f"s_{tid}", use_container_width=True):
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
                        if st.button("저장", key=f"save_fail_{tid}", use_container_width=True):
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
        st.markdown(
            f"<div style='text-align:center; font-weight:800;'>{ws.isoformat()} ~ {we.isoformat()}</div>",
            unsafe_allow_html=True,
        )
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

    # -------------------------
    # Dashboard
    # -------------------------
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Dashboard")

        st.caption(
            f"트렌드: 최근 {DASH_TREND_WEEKS}주 · 표시: TOP {DASH_TOPK} 카테고리 · "
            f"카테고리 맵: 최근 {CATEGORY_MAP_WINDOW_WEEKS}주 기반 (최대 {CATEGORY_MAX}개)"
        )

        # Fail by DOW (this week)
        st.markdown("**이번 주 실패(요일 분포)**")
        dow_df = failures_by_dow(df)
        c_dow = (
            alt.Chart(dow_df)
            .mark_bar()
            .encode(
                x=alt.X("dow:N", sort=["월", "화", "수", "목", "금", "토", "일"], title=None),
                y=alt.Y("fail_count:Q", title=None),
                tooltip=["dow", "fail_count"],
            )
            .properties(height=160)
        )
        st.altair_chart(c_dow, use_container_width=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("**실패 원인 트렌드(주별, 카테고리)**")

        # Consent gate for AI features
        if not consent_value():
            st.info("AI 기능 사용 동의가 필요해요. (하단 ‘데이터/AI 안내 및 동의’에서 체크)")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        api_key = effective_openai_key()
        model = effective_openai_model()
        if not api_key:
            st.info("OpenAI 키가 설정되면 ‘카테고리 트렌드’가 표시돼요. (하단 OpenAI 설정)")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        colA, colB = st.columns([1.2, 2.8])
        with colA:
            refresh = st.button("카테고리 맵 갱신", use_container_width=True, key="cat_map_refresh")
        with colB:
            st.caption("갱신을 누르면 최근 12주 실패 원인을 다시 묶어(최대 7개) 카테고리 맵을 업데이트해요.")

        try:
            with st.spinner("카테고리 맵 확인 중..."):
                cat_map, msg = get_or_build_category_map(user_id, api_key, model, force_refresh=bool(refresh))
        except Exception as e:
            st.error(f"카테고리 맵 처리 실패: {type(e).__name__}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.caption(msg)

        if not cat_map:
            st.info("카테고리 맵이 아직 없어요. 실패 원인 텍스트가 더 쌓이면 자동으로 만들 수 있어요.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        mapping = cat_map.get("mapping", {}) if isinstance(cat_map, dict) else {}
        categories = cat_map.get("categories", []) if isinstance(cat_map, dict) else []

        if isinstance(categories, list) and categories:
            with st.expander("카테고리 정의 보기", expanded=False):
                for cdef in categories[:CATEGORY_MAX]:
                    name = str(cdef.get("name", "카테고리"))
                    definition = str(cdef.get("definition", ""))
                    examples = cdef.get("examples", []) or []
                    st.markdown(f"**• {name}**")
                    if definition:
                        st.write(definition)
                    if examples:
                        st.write("- 예시:", ", ".join([str(x) for x in examples[:3]]))

        trend = weekly_category_trend(user_id, weeks=DASH_TREND_WEEKS, topk=DASH_TOPK, mapping=mapping)
        if trend.empty:
            st.info("최근 기간에 실패 원인 데이터가 부족해서 트렌드를 만들 수 없어요.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        y_axis = alt.Axis(title="실패 횟수", tickMinStep=1)
        c_trend = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:N", title="주 시작일(월)", sort=sorted(trend["week"].unique().tolist())),
                y=alt.Y("count:Q", title="실패 횟수", axis=y_axis),
                color=alt.Color("category:N", title="카테고리"),
                tooltip=["week", "category", "count"],
            )
            .properties(height=260)
        )
        st.altair_chart(c_trend, use_container_width=True)
        st.caption("X축: 주 시작일(월요일) · Y축: 그 주에 해당 카테고리로 기록된 실패 원인 횟수(실제 횟수)")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Weekly analysis / coaching
    # (변경점 #1: 주간 실패 차트 제거)
    # -------------------------
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 주간 분석 / 코칭")

        # Consent gate for AI features
        if not consent_value():
            st.info("AI 기능 사용 동의가 필요해요. (하단 ‘데이터/AI 안내 및 동의’에서 체크)")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        api_key = effective_openai_key()
        model = effective_openai_model()

        st.markdown("#### 원인 주간 분석")

        weekly_reasons = [r for r in fails["fail_reason"].fillna("").tolist() if str(r).strip()]
        if not api_key:
            st.info("OpenAI 키가 설정되면 분석이 표시돼요. (하단에서 키 입력)")
        elif len(weekly_reasons) == 0:
            st.write("이번 주에는 실패 원인 입력이 아직 없어요.")
        else:
            if st.button("분석 생성/갱신", use_container_width=True, key="weekly_analyze"):
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

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("#### 맞춤형 AI코칭")

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

        if st.button("코칭 생성/갱신", use_container_width=True, key="overall_coach_btn"):
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
        else:
            st.caption("‘코칭 생성/갱신’을 눌러 코칭을 받아보세요.")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("#### 코칭 챗봇")

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

    # -------------------------
    # PDF report
    # -------------------------
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Weekly PDF 리포트 (한글 폰트 포함)")

        st.caption("가장 확실한 방법: fonts/NanumGothic-Regular.ttf 를 레포에 포함하면(커밋) 네모(■) 깨짐이 100% 사라져요.")

        city = ck_get("failog_city", "").strip()
        city_label = ""
        try:
            if city:
                g = geocode_city(city)
                if g:
                    city_label = f"{g.get('name','')} · {g.get('country','')}"
        except Exception:
            city_label = city

        font_ready = ensure_korean_font_downloaded()
        if not font_ready:
            st.warning("폰트 다운로드가 막힌 환경이면 PDF 한글이 깨질 수 있어요. (레포에 폰트 파일 포함 권장)")
        else:
            st.success("PDF 한글 폰트 준비 완료")

        c1, c2, c3 = st.columns([1.1, 1.1, 2.2])
        with c1:
            target_ws = st.date_input("주 시작(월)", value=ws, key="pdf_ws")
            target_ws = week_start(target_ws)
        with c2:
            filename = st.text_input("파일명", value=f"failog_week_{target_ws.isoformat()}.pdf", key="pdf_name")
        with c3:
            st.write("")
            st.write("")
            gen = st.button("PDF 생성", use_container_width=True, key="pdf_gen")

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


# ============================================================
# Bottom OpenAI panel (prefs)
# ============================================================
def render_openai_bottom_panel():
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 🔑 OpenAI 설정")

    default_key = prefs_openai_key()
    default_model = prefs_openai_model()

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
        save = st.toggle(
            "쿠키 저장",
            value=save_default,
            help="같은 브라우저에서 유지돼요. (쿠키가 막히면 저장 안 될 수 있어요)",
            key="bottom_openai_save",
        )

    a, b, c = st.columns([1, 1, 3])
    with a:
        if st.button("적용", use_container_width=True, key="bottom_apply"):
            st.session_state["openai_api_key"] = (api_key or "").strip()
            st.session_state["openai_model"] = (model or "gpt-4o-mini").strip()

            if save:
                set_prefs_openai(api_key or "", model or "gpt-4o-mini")
            else:
                ck_del("failog_openai_key")
                ck_set("failog_openai_model", (model or "gpt-4o-mini").strip())

            st.success("적용됐어요.")
    with b:
        if st.button("저장값 삭제", use_container_width=True, key="bottom_clear"):
            ck_del("failog_openai_key")
            ck_del("failog_openai_model")
            st.success("저장값을 삭제했어요.")
            st.rerun()
    with c:
        st.caption("user_id는 URL(uid)로 고정되어 있고, OpenAI 키는 선택적으로 쿠키에 저장됩니다.")


# ============================================================
# Privacy / AI consent panel (변경점 #2)
# ============================================================
def render_privacy_ai_consent_panel():
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 🔒 데이터/AI 안내 및 동의")

    current = consent_value()

    with st.container():
        st.caption(
            "실패 이유·생활 패턴은 개인에게 민감한 데이터일 수 있어요. "
            "FAILOG는 아래 원칙으로 데이터를 다룹니다."
        )

        with st.expander("자세히 보기", expanded=False):
            st.markdown(
                """
- **저장**: 계획/습관/체크/실패원인은 서버의 **SQLite(planner.db)**에 저장됩니다.  
- **식별자**: user_id는 로그인 대신 **URL의 uid 파라미터**로 구분됩니다. (링크를 공유하면 동일 데이터가 보일 수 있어요)  
- **쿠키**: OpenAI 키/모델, 알림/날씨 등 일부 설정은 **쿠키**에 저장될 수 있습니다. (브라우저 정책에 따라 제한 가능)  
- **AI(OpenAI) 사용**:  
  - *버튼을 눌러 요청한 경우에만* 실패 원인을 분석/카테고리화/코칭을 위해 OpenAI API가 호출됩니다.  
  - 호출 시, 분석에 필요한 범위의 텍스트(실패 원인/요약된 패턴 등)가 전송될 수 있습니다.  
  - 동의하지 않으면 AI 기능은 작동하지 않습니다.
                """.strip()
            )

        checked = st.checkbox(
            "위 내용을 이해했으며, OpenAI 기반 분석/코칭 기능 사용에 동의합니다.",
            value=current,
            key="ai_consent_checkbox",
        )
        if checked != current:
            set_consent(bool(checked))
            st.success("동의 설정이 저장됐어요.")


# ============================================================
# Top nav
# ============================================================
def top_nav():
    if "screen" not in st.session_state:
        st.session_state["screen"] = "planner"

    c1, c2, _ = st.columns([1.2, 1.8, 6])
    with c1:
        if st.button(" Planner", use_container_width=True, key="nav_plan"):
            st.session_state["screen"] = "planner"
            st.rerun()
    with c2:
        if st.button(" Failure Report", use_container_width=True, key="nav_fail"):
            st.session_state["screen"] = "fail"
            st.rerun()

    st.write("")
    return st.session_state["screen"]


# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(page_title="FAILOG", page_icon="🧊", layout="wide")
    inject_css()
    init_db()

    user_id = get_or_create_user_id()

    render_hero()

    screen = top_nav()
    if screen == "planner":
        screen_planner(user_id)
    else:
        screen_failures(user_id)

    render_openai_bottom_panel()
    render_privacy_ai_consent_panel()


if __name__ == "__main__":
    main()
