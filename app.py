# app.py
# ------------------------------------------------------------
# FAILOG
# 실패를 성공으로! 계획과 습관의 실패를 기록하고 맞춤형 코칭을 받아보자
#
# Key upgrades:
# - device/browser-level user isolation (no login) via cookie-based user_id
# - per-user SQLite partitioning + auto migration from old shared tables
# - improved UI theme (#A0C4F2 + white), consistent spacing, button hierarchy
# - weekly fail chart Mon..Sun order + smaller height
# - deeper personalized coaching prompt using 28-day signals + repeated>=14d creative alternatives forced
# - reminder uses Asia/Seoul time; optional auto-refresh if streamlit-autorefresh installed
#
# Run:
#   pip install streamlit pandas openai altair extra-streamlit-components
#   (optional) pip install streamlit-autorefresh
#   streamlit run app.py
# ------------------------------------------------------------

import json
import re
import sqlite3
import uuid
from datetime import date, datetime, timedelta, time
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import streamlit as st
import altair as alt

# Cookie manager
import extra_streamlit_components as stx

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

# Timezone (Python 3.9+)
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
ACCENT = "#A0C4F2"
DB_PATH = "planner.db"


# =========================
# THEME / CSS (서비스 느낌)
# =========================
def inject_css():
    st.markdown(
        f"""
<style>
/* Page canvas */
.block-container {{
  max-width: 1120px;
  padding-top: 1.0rem;
  padding-bottom: 2.2rem;
}}

/* Soft background */
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 420px at 30% 0%, rgba(160,196,242,0.28), rgba(255,255,255,0) 60%),
              linear-gradient(180deg, rgba(160,196,242,0.18) 0%, rgba(255,255,255,1) 55%);
}}

/* Typography */
h1,h2,h3 {{
  letter-spacing: -0.02em;
}}
.small {{
  color: rgba(31,36,48,0.65);
  font-size: 0.92rem;
}}

/* Cards */
.card {{
  border: 1px solid rgba(160,196,242,0.58);
  border-radius: 18px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.94);
  box-shadow: 0 10px 26px rgba(160,196,242,0.14);
}}
.card-tight {{
  border: 1px solid rgba(160,196,242,0.45);
  border-radius: 16px;
  padding: 12px 12px;
  background: rgba(255,255,255,0.93);
}}

/* Pills / badges */
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
.pill-muted {{
  background: rgba(255,255,255,0.88);
  border-color: rgba(160,196,242,0.42);
}}

/* Task item */
.task {{
  border: 1px solid rgba(160,196,242,0.46);
  border-radius: 16px;
  padding: 10px 10px;
  background: rgba(255,255,255,0.95);
}}
.task + .task {{
  margin-top: 8px;
}}

/* Reduce default widget heaviness */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
  border-radius: 14px !important;
  border: 1px solid rgba(160,196,242,0.55) !important;
}}
[data-testid="stMultiSelect"] div {{
  border-radius: 14px !important;
}}

/* Buttons */
div[data-testid="stButton"] > button {{
  border-radius: 14px !important;
  white-space: nowrap !important;
}}

/* Month calendar button compact + prevent two-digit wrapping */
.month-grid div[data-testid="stButton"] > button {{
  font-size: 0.82rem !important;
  padding: 0.10rem 0.18rem !important;
  line-height: 1.05 !important;
  min-height: 30px !important;
  white-space: nowrap !important;
}}

/* Subtle section dividers */
hr {{
  margin: 1.1rem 0;
  border: none;
  border-top: 1px solid rgba(160,196,242,0.35);
}}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================
# USER ID (cookie-based)
# =========================
def get_or_create_user_id() -> str:
    """
    기기(브라우저)별로 고정되는 user_id.
    - same browser -> same cookie -> same user_id
    - different device -> different cookie -> different user_id
    """
    cookie = stx.CookieManager()
    uid = cookie.get("failog_uid")
    if not uid:
        uid = str(uuid.uuid4())
        # expires_at=None: 브라우저가 유지하는 한 장기 쿠키(환경에 따라 다를 수 있음)
        cookie.set("failog_uid", uid, expires_at=None)
    return uid


# =========================
# DB helpers
# =========================
def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA foreign_keys = ON;")
    return c

def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")

def table_has_column(c: sqlite3.Connection, table: str, col: str) -> bool:
    rows = c.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r[1] == col for r in rows)

def table_exists(c: sqlite3.Connection, table: str) -> bool:
    row = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)).fetchone()
    return bool(row)

def init_db_and_migrate_if_needed():
    """
    목표 스키마:
      tasks(user_id,..., UNIQUE(user_id, task_date, source, habit_id, text))
      habits(user_id,...)
      settings(user_id,key,value, UNIQUE(user_id,key))
    기존 스키마(공유 DB)가 있으면 자동으로 v2로 마이그레이션하고 old table은 backup으로 rename.
    """
    c = conn()
    cur = c.cursor()

    # If new tables already exist, done.
    if table_exists(c, "tasks") and table_has_column(c, "tasks", "user_id") and \
       table_exists(c, "habits") and table_has_column(c, "habits", "user_id") and \
       table_exists(c, "settings") and table_has_column(c, "settings", "user_id"):
        c.close()
        return

    # If old tables exist without user_id, migrate.
    # We'll create new tables tasks_v2/habits_v2/settings_v2, copy data with user_id='shared',
    # then rename.
    shared_uid = "shared"

    # Create v2 tables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks_v2 (
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS habits_v2 (
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
        CREATE TABLE IF NOT EXISTS settings_v2 (
          user_id TEXT NOT NULL,
          key TEXT NOT NULL,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (user_id, key)
        );
        """
    )

    # Copy from old tasks if exist
    if table_exists(c, "tasks"):
        if table_has_column(c, "tasks", "user_id"):
            # Already v2-like, copy to tasks_v2 just in case
            cur.execute(
                """
                INSERT OR IGNORE INTO tasks_v2
                (id, user_id, task_date, text, source, habit_id, status, fail_reason, created_at, updated_at)
                SELECT id, user_id, task_date, text, source, habit_id, status, fail_reason, created_at, updated_at
                FROM tasks;
                """
            )
        else:
            cur.execute(
                """
                INSERT OR IGNORE INTO tasks_v2
                (user_id, task_date, text, source, habit_id, status, fail_reason, created_at, updated_at)
                SELECT ?, task_date, text, source, habit_id, status, fail_reason, created_at, updated_at
                FROM tasks;
                """,
                (shared_uid,),
            )

    # Copy from old habits if exist
    if table_exists(c, "habits"):
        if table_has_column(c, "habits", "user_id"):
            cur.execute(
                """
                INSERT OR IGNORE INTO habits_v2
                (id, user_id, title, dow_mask, active, created_at, updated_at)
                SELECT id, user_id, title, dow_mask, active, created_at, updated_at
                FROM habits;
                """
            )
        else:
            cur.execute(
                """
                INSERT OR IGNORE INTO habits_v2
                (user_id, title, dow_mask, active, created_at, updated_at)
                SELECT ?, title, dow_mask, active, created_at, updated_at
                FROM habits;
                """,
                (shared_uid,),
            )

    # Copy from old settings if exist
    if table_exists(c, "settings"):
        if table_has_column(c, "settings", "user_id"):
            cur.execute(
                """
                INSERT OR IGNORE INTO settings_v2
                (user_id, key, value, updated_at)
                SELECT user_id, key, value, updated_at
                FROM settings;
                """
            )
        else:
            cur.execute(
                """
                INSERT OR IGNORE INTO settings_v2
                (user_id, key, value, updated_at)
                SELECT ?, key, value, updated_at
                FROM settings;
                """,
                (shared_uid,),
            )

    c.commit()

    # Backup old tables then promote v2
    def safe_rename(old: str, new: str):
        if table_exists(c, old):
            cur.execute(f"ALTER TABLE {old} RENAME TO {new};")

    safe_rename("tasks", "tasks_backup")
    safe_rename("habits", "habits_backup")
    safe_rename("settings", "settings_backup")

    cur.execute("ALTER TABLE tasks_v2 RENAME TO tasks;")
    cur.execute("ALTER TABLE habits_v2 RENAME TO habits;")
    cur.execute("ALTER TABLE settings_v2 RENAME TO settings;")

    c.commit()
    c.close()


# =========================
# Settings (per user)
# =========================
DEFAULT_SETTINGS = {
    "openai_api_key": "",
    "openai_model": "gpt-4o-mini",
    "reminder_enabled": "true",
    "reminder_time": "21:30",
    "reminder_window_min": "15",
    "reminder_autorefresh": "true",  # optional
    "reminder_autorefresh_sec": "60",
}

def ensure_user_settings(user_id: str):
    c = conn()
    cur = c.cursor()
    for k, v in DEFAULT_SETTINGS.items():
        cur.execute(
            """
            INSERT OR IGNORE INTO settings(user_id, key, value, updated_at)
            VALUES (?,?,?,?)
            """,
            (user_id, k, v, now_iso()),
        )
    c.commit()
    c.close()

def get_setting(user_id: str, key: str, default: str = "") -> str:
    c = conn()
    row = c.execute(
        "SELECT value FROM settings WHERE user_id=? AND key=?",
        (user_id, key),
    ).fetchone()
    c.close()
    return row[0] if row else default

def set_setting(user_id: str, key: str, value: str):
    c = conn()
    c.execute(
        """
        INSERT INTO settings(user_id, key, value, updated_at)
        VALUES (?,?,?,?)
        ON CONFLICT(user_id, key) DO UPDATE SET
          value=excluded.value,
          updated_at=excluded.updated_at
        """,
        (user_id, key, value, now_iso()),
    )
    c.commit()
    c.close()


# =========================
# Date helpers (Mon-Sun)
# =========================
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


# =========================
# Habits / Tasks (per user)
# =========================
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
    # keep past success/fail for coaching; clean future/ongoing todo from this habit
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
    reason = (reason or "").strip()
    c = conn()
    c.execute(
        "UPDATE tasks SET status='fail', fail_reason=?, updated_at=? WHERE user_id=? AND id=?",
        (reason if reason else "이유 미기록", now_iso(), user_id, task_id),
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


# =========================
# Reminder
# =========================
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


# =========================
# OpenAI
# =========================
def effective_openai_key(user_id: str) -> str:
    # session override first
    sk = st.session_state.get("openai_api_key", "")
    if sk.strip():
        return sk.strip()
    return get_setting(user_id, "openai_api_key", "").strip()

def openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되지 않았어요. pip install openai")
    if not api_key.strip():
        raise RuntimeError("OpenAI API Key가 비어 있어요.")
    return OpenAI(api_key=api_key.strip())


# =========================
# Repeated failure detection (>=14 days)
# =========================
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


# =========================
# Coaching prompts (deeper personalization)
# =========================
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
  (예: 특정 습관/계획 이름, 실패가 몰리는 요일, 사용자 실패 이유 표현 일부, 연속 실패 구간 등)
- actionable_advice는 '작고 구체적' (환경/시간/트리거/대체행동/장애물 대비 포함)
- 비난/자책 유도 금지, 코칭 톤
- repeated_2w=true 항목이 하나라도 있으면 해당 원인에는 creative_advice_when_repeated_2w를 반드시 채워라(빈 배열 금지)
"""

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

    total = len(df)
    fail = int(df["is_fail"].sum())
    succ = int(df["is_success"].sum())
    todo = int((df["status"] == "todo").sum())

    by_source = df.groupby("source")["status"].value_counts().unstack(fill_value=0).to_dict()

    dow_fail = df[df["is_fail"]].groupby("dow")["is_fail"].sum().reindex(range(7), fill_value=0).to_dict()
    fail_by_dow = {korean_dow(int(k)): int(v) for k, v in dow_fail.items()}

    top_failed_items_df = (
        df[df["is_fail"]]
        .groupby(["text", "source"])["is_fail"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_failed_items = [
        {"item": r["text"], "type": r["source"], "fail_count": int(r["is_fail"])}
        for _, r in top_failed_items_df.iterrows()
    ]

    reasons = df[df["is_fail"]]["fail_reason"].fillna("").map(lambda s: s.strip())
    top_reasons = reasons[reasons != ""].value_counts().head(10).to_dict()

    # Longest streak of consecutive days with >=1 failure
    fails_by_day = df[df["is_fail"]].groupby("task_date")["is_fail"].sum()
    fail_days = sorted(fails_by_day.index.tolist())
    longest = 0
    current = 0
    prev = None
    for d in fail_days:
        if prev is None or (d - prev).days == 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        prev = d
    longest = max(longest, current) if fail_days else 0

    return {
        "has_data": True,
        "window_days": days,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "counts": {"total": total, "success": succ, "fail": fail, "todo": todo},
        "fail_by_dow": fail_by_dow,
        "by_source": by_source,
        "top_failed_items": top_failed_items,
        "top_reasons": top_reasons,
        "longest_fail_streak_days": int(longest),
    }


def llm_weekly_reason_analysis(api_key: str, model: str, reasons: List[str]) -> Dict[str, Any]:
    client = openai_client(api_key)
    prompt = f"""
너는 사용자의 실패 이유를 읽고, '이번 주' 관점에서 공통 원인을 최대 3개로 묶어 요약해.
가능하면 사용자가 쓴 표현을 유지하고, 추상적인 말 대신 실제 표현을 묶는 방식으로 분류해.

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
- examples는 원문을 짧게(각 1줄)
- estimated_count는 정수
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

아래 '사용자 패턴 요약'과 '실패 기록 샘플'을 함께 참고해서,
누구에게나 해당되는 말이 아니라 이 사용자에게 맞춘 날카로운 코칭을 만들어줘.

특히:
- 실패가 몰리는 요일/상황이 보이면 그 패턴에 맞춘 조언을 해줘.
- plan(일회성)과 habit(반복) 중 어디에서 더 흔들리는지에 따라 접근을 달리해줘.
- 항목명이 구체적일수록 행동 설계를 더 구체화해줘(트리거/장소/시간/대체행동/장애물 대비).
- repeated_2w=true 항목이 하나라도 있으면, 그 원인에는 반드시 창의 대안을 포함해(빈 배열 금지).

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


# =========================
# UI: Bottom OpenAI panel (per user)
# =========================
def render_openai_bottom_panel(user_id: str):
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 🔑 OpenAI 설정")

    col1, col2, col3 = st.columns([3.0, 1.6, 1.4])
    with col1:
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            placeholder="sk-...",
            key="bottom_openai_key",
        )
    with col2:
        model = st.text_input("모델", value=get_setting(user_id, "openai_model", "gpt-4o-mini"), key="bottom_openai_model")
    with col3:
        save = st.toggle("로컬 저장", value=False, help="공용 PC면 끄는 걸 추천", key="bottom_openai_save")

    a, b = st.columns([1, 4])
    with a:
        if st.button("적용", use_container_width=True, key="bottom_apply"):
            st.session_state["openai_api_key"] = api_key.strip()
            set_setting(user_id, "openai_model", (model.strip() or "gpt-4o-mini"))
            if save:
                set_setting(user_id, "openai_api_key", api_key.strip())
            st.success("적용됐어요.")
    with b:
        st.caption("키가 없으면 원인 분석/코칭/챗봇이 동작하지 않아요.")


# =========================
# Screen: Planner
# =========================
def screen_planner(user_id: str):
    st.markdown("## Planner")

    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = date.today()

    selected = st.session_state["selected_date"]
    ws = week_start(selected)

    ensure_week_habit_tasks(user_id, ws)

    # Optional autorefresh to make reminder reliable
    if st_autorefresh is not None:
        if get_setting(user_id, "reminder_autorefresh", "true").lower() == "true":
            sec = int(get_setting(user_id, "reminder_autorefresh_sec", "60"))
            st_autorefresh(interval=sec * 1000, key="planner_refresh")

    # Reminder (KST)
    if get_setting(user_id, "reminder_enabled", "true").lower() == "true":
        rt = parse_hhmm(get_setting(user_id, "reminder_time", "21:30"))
        win = int(get_setting(user_id, "reminder_window_min", "15"))
        now_dt = datetime.now(KST)
        if should_remind(now_dt, rt, win):
            todos = count_today_todos(user_id)
            if todos > 0:
                st.toast(f"⏰ 아직 체크하지 않은 항목이 {todos}개 있어요", icon="⏰")

    left, right = st.columns([1.05, 1.95], gap="large")

    # Month card
    with left:
        st.markdown("<div class='card month-grid'>", unsafe_allow_html=True)
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

                # make selected date a bit highlighted by using emoji dot kept minimal
                if cols[i].button(label, key=f"cal_{d.isoformat()}", use_container_width=True):
                    st.session_state["selected_date"] = d
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("알림 설정", expanded=False):
            en = st.toggle("리마인더 켜기", value=get_setting(user_id, "reminder_enabled", "true").lower() == "true", key="rem_en")
            t = st.text_input("시간(HH:MM)", value=get_setting(user_id, "reminder_time", "21:30"), key="rem_time")
            w = st.number_input("허용 오차(분)", min_value=1, max_value=120, value=int(get_setting(user_id, "reminder_window_min", "15")), key="rem_win")
            if st_autorefresh is not None:
                ar = st.toggle("자동 갱신(권장)", value=get_setting(user_id, "reminder_autorefresh", "true").lower() == "true", key="rem_ar")
                sec = st.number_input("갱신 주기(초)", min_value=15, max_value=600, value=int(get_setting(user_id, "reminder_autorefresh_sec", "60")), key="rem_ar_sec")
            else:
                ar = None
                sec = None
                st.caption("자동 갱신을 쓰려면 streamlit-autorefresh 설치가 필요해요.")

            if st.button("저장", use_container_width=True, key="rem_save"):
                set_setting(user_id, "reminder_enabled", "true" if en else "false")
                set_setting(user_id, "reminder_time", (t or "21:30"))
                set_setting(user_id, "reminder_window_min", str(int(w)))
                if ar is not None:
                    set_setting(user_id, "reminder_autorefresh", "true" if ar else "false")
                    set_setting(user_id, "reminder_autorefresh_sec", str(int(sec)))
                st.success("저장됐어요.")

    # Right: week + tasks
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Current Week")
        st.markdown(
            f"<span class='pill pill-strong'>Week</span> "
            f"<span class='pill pill-muted'>{ws.isoformat()} ~ {(ws+timedelta(days=6)).isoformat()}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        # week buttons
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

        # Add plan (form)
        with st.form("plan_add_form", clear_on_submit=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                plan_text = st.text_input("계획 추가(1회성)", placeholder="예: 독서 10분 / 이메일 정리", key="plan_text_input")
            with c2:
                submitted = st.form_submit_button("추가", use_container_width=True)
            if submitted:
                add_plan_task(user_id, selected, plan_text)
                st.rerun()

        # Habit manage (min)
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

        # Tasks list
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
                    st.markdown(
                        f"**{status_icon} {text}**  "
                        f"<span class='pill pill-muted'>{badge}</span>",
                        unsafe_allow_html=True,
                    )
                    if status == "fail":
                        st.caption(f"실패 원인: {reason}")

                # Button hierarchy: success primary, others secondary-like
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


# =========================
# Screen: Failure Report
# =========================
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

    # --- Weekly fail chart (Mon..Sun, small height)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 주간 실패 차트")

    days = week_days(ws)  # Mon..Sun order
    chart_rows = []
    for d in days:
        chart_rows.append({"dow": korean_dow(d.weekday()), "fail_count": int((fails["task_date"] == d).sum())})
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

    api_key = effective_openai_key(user_id)
    model = get_setting(user_id, "openai_model", "gpt-4o-mini")

    # --- Weekly reason analysis
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
                    ex = g.get("examples", []) or []
                    for s in ex[:3]:
                        st.write(f"- {s}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # --- Personalized coaching + chat
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
                "type": str(r["source"]),  # plan/habit
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
            st.write("코칭 결과가 비어 있어요.")
        else:
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

    # Chat (kept at bottom, no extra header box)
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

        # Personal context: last 14 days reasons + signals + recent fail samples
        end = date.today()
        start = end - timedelta(days=13)
        last14 = get_tasks_range(user_id, start, end)
        last14_fail = last14[last14["status"] == "fail"]
        top_reasons = (
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
- 최근 14일 실패 이유 상위: {json.dumps(top_reasons, ensure_ascii=False)}
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


# =========================
# Top nav
# =========================
def top_nav():
    if "screen" not in st.session_state:
        st.session_state["screen"] = "planner"

    c1, c2, _ = st.columns([1.2, 1.5, 6])
    with c1:
        if st.button(" Planner", use_container_width=True, key="nav_plan", type="primary" if st.session_state["screen"]=="planner" else "secondary"):
            st.session_state["screen"] = "planner"
            st.rerun()
    with c2:
        if st.button("Failure Report", use_container_width=True, key="nav_fail", type="primary" if st.session_state["screen"]=="fail" else "secondary"):
            st.session_state["screen"] = "fail"
            st.rerun()

    st.write("")
    return st.session_state["screen"]


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="FAILOG", page_icon="🧊", layout="wide")
    inject_css()

    # init / migrate schema once
    init_db_and_migrate_if_needed()

    # user id
    user_id = get_or_create_user_id()
    st.session_state["user_id"] = user_id

    # per-user default settings
    ensure_user_settings(user_id)

    # header
    st.markdown("# FAILOG")
    st.markdown(
        "<div class='small'>실패를 성공으로! 계획과 습관의 실패를 기록하고 맞춤형 코칭을 받아보자</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # nav
    screen = top_nav()

    if screen == "planner":
        screen_planner(user_id)
    else:
        screen_failures(user_id)

    render_openai_bottom_panel(user_id)


if __name__ == "__main__":
    main()
