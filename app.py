# app.py
# ------------------------------------------------------------
# FAILOG: 실패를 성공으로! 계획과 습관의 실패를 기록하고 맞춤형 코칭을 받아보자
#
# 2-screen Streamlit app (깔끔/단순 + 삭제 + 오류 해결 + 개인화 코칭 강화)
# - Main: Planner (Month + Current Week, 계획/습관 추가/삭제, 성공/실패, 실패 원인 입력)
# - Sub : Failure Report (주간 실패 차트, 원인 주간 분석, 맞춤형 AI코칭, 챗봇)
#
# OpenAI 키: 하단 입력 + 로컬 저장 토글(DB)
#
# Run:
#   pip install streamlit pandas openai altair
#   streamlit run app.py
# ------------------------------------------------------------

import json
import re
import sqlite3
from datetime import date, datetime, timedelta, time
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import streamlit as st
import altair as alt

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# THEME / CSS  (#A0C4F2 + white)
# =========================
ACCENT = "#A0C4F2"

def inject_css():
    st.markdown(
        f"""
<style>
/* Page */
.block-container {{
  max-width: 1100px;
  padding-top: 1.1rem;
  padding-bottom: 2rem;
}}

/* Soft blue background */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, rgba(160,196,242,0.18) 0%, rgba(255,255,255,1) 55%);
}}

/* Typography */
h1,h2,h3 {{ letter-spacing: -0.02em; }}
.small {{ color: rgba(49,51,63,0.65); font-size: 0.92rem; }}

/* Cards */
.card {{
  border: 1px solid rgba(160,196,242,0.55);
  border-radius: 18px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.92);
  box-shadow: 0 6px 18px rgba(160,196,242,0.12);
}}

/* Pills */
.pill {{
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  border:1px solid rgba(160,196,242,0.60);
  font-size:0.85rem;
  margin-right:6px;
  background: rgba(255,255,255,0.75);
}}
.pill-strong {{
  background: rgba(160,196,242,0.26);
  border-color: rgba(160,196,242,0.85);
}}
.pill-weak {{
  background: rgba(255,255,255,0.85);
  border-color: rgba(160,196,242,0.45);
}}

/* Tasks */
.task {{
  border: 1px solid rgba(160,196,242,0.45);
  border-radius: 16px;
  padding: 10px 10px;
  background: rgba(255,255,255,0.92);
}}
.task + .task {{ margin-top: 8px; }}

hr {{ margin: 1.2rem 0; }}

/* Buttons: prevent wrap and keep compact (fix Month 2-digit wrapping) */
button {{
  white-space: nowrap !important;
}}
div[data-testid="stButton"] > button {{
  border-radius: 14px !important;
}}
/* Month calendar buttons: slightly smaller */
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {{
  font-size: 0.85rem;
  padding: 0.15rem 0.25rem;
  line-height: 1.1;
}}
</style>
""",
        unsafe_allow_html=True,
    )


# =========================
# DB (SQLite)
# =========================
DB_PATH = "planner.db"

def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA foreign_keys = ON;")
    return c

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def init_db():
    c = conn()
    cur = c.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          task_date TEXT NOT NULL,
          text TEXT NOT NULL,
          source TEXT NOT NULL CHECK(source IN ('plan','habit')),
          habit_id INTEGER,
          status TEXT NOT NULL CHECK(status IN ('todo','success','fail')) DEFAULT 'todo',
          fail_reason TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE(task_date, source, habit_id, text)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS habits (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        CREATE TABLE IF NOT EXISTS settings (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )

    defaults = {
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        "reminder_enabled": "true",
        "reminder_time": "21:30",
        "reminder_window_min": "15",
    }
    for k, v in defaults.items():
        cur.execute(
            "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?,?,?)",
            (k, v, now_iso()),
        )

    c.commit()
    c.close()

def get_setting(key: str, default: str = "") -> str:
    c = conn()
    cur = c.cursor()
    row = cur.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    c.close()
    return row[0] if row else default

def set_setting(key: str, value: str):
    c = conn()
    c.execute(
        """
        INSERT INTO settings (key, value, updated_at)
        VALUES (?,?,?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value, now_iso()),
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
# Habits / Tasks CRUD
# =========================
def list_habits(active_only: bool = True) -> pd.DataFrame:
    c = conn()
    q = "SELECT id, title, dow_mask, active FROM habits"
    if active_only:
        q += " WHERE active=1"
    q += " ORDER BY id DESC"
    df = pd.read_sql_query(q, c)
    c.close()
    return df

def add_habit(title: str, dows: List[int]):
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
        INSERT INTO habits (title, dow_mask, active, created_at, updated_at)
        VALUES (?,?,1,?,?)
        """,
        (title, dow_mask, now_iso(), now_iso()),
    )
    c.commit()
    c.close()

def set_habit_active(habit_id: int, active: bool):
    c = conn()
    c.execute(
        "UPDATE habits SET active=?, updated_at=? WHERE id=?",
        (1 if active else 0, now_iso(), habit_id),
    )
    c.commit()
    c.close()

def delete_habit(habit_id: int):
    # 오늘/미래의 todo 습관 항목은 정리, 과거 성공/실패 기록은 유지(코칭/분석 품질↑)
    today = date.today().isoformat()
    c = conn()
    cur = c.cursor()
    cur.execute(
        "DELETE FROM tasks WHERE source='habit' AND habit_id=? AND task_date>=? AND status='todo'",
        (habit_id, today),
    )
    cur.execute("DELETE FROM habits WHERE id=?", (habit_id,))
    c.commit()
    c.close()

def ensure_week_habit_tasks(ws: date):
    habits = list_habits(active_only=True)
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
                      (task_date, text, source, habit_id, status, fail_reason, created_at, updated_at)
                    VALUES (?,?,?,?, 'todo', NULL, ?, ?)
                    """,
                    (d.isoformat(), title, "habit", hid, now_iso(), now_iso()),
                )
    c.commit()
    c.close()

def add_plan_task(d: date, text: str):
    text = (text or "").strip()
    if not text:
        return
    c = conn()
    c.execute(
        """
        INSERT INTO tasks
          (task_date, text, source, habit_id, status, fail_reason, created_at, updated_at)
        VALUES (?,?,?,?, 'todo', NULL, ?, ?)
        """,
        (d.isoformat(), text, "plan", None, now_iso(), now_iso()),
    )
    c.commit()
    c.close()

def delete_task(task_id: int):
    c = conn()
    c.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    c.commit()
    c.close()

def list_tasks_for_date(d: date) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(
        """
        SELECT id, task_date, text, source, habit_id, status, fail_reason
        FROM tasks
        WHERE task_date=?
        ORDER BY source DESC, id DESC
        """,
        c,
        params=(d.isoformat(),),
    )
    c.close()
    return df

def update_task_status(task_id: int, status: str):
    c = conn()
    c.execute("UPDATE tasks SET status=?, updated_at=? WHERE id=?", (status, now_iso(), task_id))
    if status != "fail":
        c.execute("UPDATE tasks SET fail_reason=NULL, updated_at=? WHERE id=?", (now_iso(), task_id))
    c.commit()
    c.close()

def update_task_fail(task_id: int, reason: str):
    reason = (reason or "").strip()
    c = conn()
    c.execute(
        "UPDATE tasks SET status='fail', fail_reason=?, updated_at=? WHERE id=?",
        (reason if reason else "이유 미기록", now_iso(), task_id),
    )
    c.commit()
    c.close()

def get_tasks_range(start_d: date, end_d: date) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(
        """
        SELECT id, task_date, text, source, habit_id, status, fail_reason
        FROM tasks
        WHERE task_date BETWEEN ? AND ?
        ORDER BY task_date ASC, id DESC
        """,
        c,
        params=(start_d.isoformat(), end_d.isoformat()),
    )
    c.close()
    return df

def get_all_failures(limit: int = 300) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(
        """
        SELECT task_date, text, source, habit_id, fail_reason
        FROM tasks
        WHERE status='fail'
        ORDER BY task_date DESC
        LIMIT ?
        """,
        c,
        params=(limit,),
    )
    c.close()
    return df


# =========================
# In-app reminder
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
    target = datetime.combine(now_dt.date(), remind_t)
    delta_min = abs((now_dt - target).total_seconds()) / 60.0
    return delta_min <= float(window_min)

def count_today_todos() -> int:
    today = date.today().isoformat()
    c = conn()
    row = c.execute("SELECT COUNT(*) FROM tasks WHERE task_date=? AND status='todo'", (today,)).fetchone()
    c.close()
    return int(row[0] if row else 0)


# =========================
# OpenAI
# =========================
def effective_openai_key() -> str:
    sk = st.session_state.get("openai_api_key", "")
    if sk.strip():
        return sk.strip()
    return get_setting("openai_api_key", "").strip()

def openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되지 않았어요. pip install openai")
    if not api_key.strip():
        raise RuntimeError("OpenAI API Key가 비어 있어요.")
    return OpenAI(api_key=api_key.strip())


# =========================
# Repeated failure detection (>=14 days) by normalized reason
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
# LLM prompts (more personalized)
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
      "summary":"사용자 상황에 맞춘 2~4문장 (구체적)",
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
- summary/advice는 '사용자 데이터'에 등장한 구체 요소(습관/계획 이름, 요일 패턴, 연속성, 실패 이유 표현)를 반드시 반영
- 비난/자책 유도 금지, 코칭 톤
- repeated_2w=true 항목이 하나라도 있으면, 그에 대응하는 원인에는 creative_advice_when_repeated_2w를 반드시 채워라
"""


def compute_user_signals(days: int = 28) -> Dict[str, Any]:
    """
    코칭 개인화를 위한 신호 추출:
    - 최근 N일: 요일별 실패 분포, plan vs habit 실패 비율, 실패가 잦은 항목 top, 연속 실패 구간, 대표 실패 이유 top
    """
    end = date.today()
    start = end - timedelta(days=days - 1)
    df = get_tasks_range(start, end)
    if df.empty:
        return {"window_days": days, "has_data": False}

    df = df.copy()
    df["task_date"] = pd.to_datetime(df["task_date"]).dt.date
    df["dow"] = df["task_date"].map(lambda d: d.weekday())  # 0..6
    df["is_fail"] = df["status"].eq("fail")
    df["is_success"] = df["status"].eq("success")

    # Overall rates
    total = len(df)
    fail = int(df["is_fail"].sum())
    succ = int(df["is_success"].sum())
    todo = int((df["status"] == "todo").sum())

    # Source split
    by_source = df.groupby("source")["status"].value_counts().unstack(fill_value=0).to_dict()

    # Day-of-week fail counts (Mon..Sun)
    dow_fail = df[df["is_fail"]].groupby("dow")["is_fail"].sum().reindex(range(7), fill_value=0).to_dict()

    # Top failed items
    top_failed_items = (
        df[df["is_fail"]].groupby(["text", "source"])["is_fail"].sum().sort_values(ascending=False).head(8).reset_index()
    )
    top_failed_items_list = [
        {"item": r["text"], "type": r["source"], "fail_count": int(r["is_fail"])} for _, r in top_failed_items.iterrows()
    ]

    # Top reasons
    reasons = df[df["is_fail"]]["fail_reason"].fillna("").map(lambda s: s.strip())
    top_reasons = reasons[reasons != ""].value_counts().head(8).to_dict()

    # Find simple streaks (consecutive days with at least one fail)
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
        "window_days": days,
        "has_data": True,
        "counts": {"total": total, "success": succ, "fail": fail, "todo": todo},
        "fail_by_dow": {korean_dow(int(k)): int(v) for k, v in dow_fail.items()},
        "by_source": by_source,  # nested dict
        "top_failed_items": top_failed_items_list,
        "top_reasons": top_reasons,
        "longest_fail_streak_days": int(longest),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }


def llm_weekly_reason_analysis(api_key: str, model: str, reasons: List[str]) -> Dict[str, Any]:
    client = openai_client(api_key)
    prompt = f"""
너는 사용자의 실패 이유를 읽고, '이번 주' 관점에서 공통 원인을 최대 3개로 묶어 요약해.
입력은 사용자가 직접 쓴 실패 이유 목록이야. 가능한 한 사용자가 쓴 표현을 존중해서 묶어줘.

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
- estimated_count는 대략적인 개수(정수)
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
    """
    개인화 강화:
    - fail_items: 최근 실패 샘플(원인 + repeated_2w 플래그)
    - signals: 최근 4주 패턴(요일, 항목, plan/habit 비율, 연속 실패 등)
    """
    client = openai_client(api_key)
    prompt = f"""
{BASE_COACH_PROMPT}

아래 '사용자 패턴 요약'과 '실패 기록 샘플'을 함께 참고해서,
누구에게나 해당되는 말이 아니라, 이 사용자에게 맞춘 날카로운 코칭을 만들어줘.
특히:
- 실패가 몰리는 요일/상황이 보이면 그 패턴에 맞춘 조언을 해줘.
- plan(일회성)과 habit(반복) 중 어디에서 더 흔들리는지에 따라 접근을 달리해줘.
- 항목명이 구체적일수록(예: 운동 10분) 행동 설계를 더 구체화해줘.
- repeated_2w=true가 하나라도 있으면, 그 원인에는 반드시 '창의 버전' 대안을 포함해.

사용자 패턴 요약(최근 {signals.get("window_days")}일):
{json.dumps(signals, ensure_ascii=False, indent=2)}

실패 기록 샘플(최근 실패 일부):
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
# Bottom OpenAI panel
# =========================
def render_openai_bottom_panel():
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 🔑 OpenAI 설정")

    col1, col2, col3 = st.columns([3.2, 1.6, 1.4])
    with col1:
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            placeholder="sk-...",
            key="bottom_openai_key",
        )
    with col2:
        model = st.text_input("모델", value=get_setting("openai_model", "gpt-4o-mini"), key="bottom_openai_model")
    with col3:
        save = st.toggle("로컬 저장", value=False, help="공용 PC면 끄는 걸 추천", key="bottom_openai_save")

    b1, b2 = st.columns([1, 4])
    with b1:
        if st.button("적용", use_container_width=True, key="bottom_apply"):
            st.session_state["openai_api_key"] = api_key.strip()
            set_setting("openai_model", (model.strip() or "gpt-4o-mini"))
            if save:
                set_setting("openai_api_key", api_key.strip())
            st.success("적용됐어요.")
    with b2:
        st.caption("키가 없으면 원인 분석/코칭/챗봇이 동작하지 않아요.")


# =========================
# Screen: Planner
# =========================
def screen_planner():
    st.markdown("## Planner")

    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = date.today()

    selected = st.session_state["selected_date"]
    ws = week_start(selected)

    ensure_week_habit_tasks(ws)

    # Reminder popup
    if get_setting("reminder_enabled", "true").lower() == "true":
        rt = parse_hhmm(get_setting("reminder_time", "21:30"))
        win = int(get_setting("reminder_window_min", "15"))
        if should_remind(datetime.now(), rt, win):
            todos = count_today_todos()
            if todos > 0:
                st.toast(f"⏰ 아직 체크하지 않은 항목이 {todos}개 있어요", icon="⏰")

    left, right = st.columns([1.05, 1.95], gap="large")

    # Month (compact)
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
                f"<div style='text-align:center; font-weight:650; font-size:1.05rem;'>{y}.{m:02d}</div>",
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
            "<div style='display:grid; grid-template-columns: repeat(7, 1fr); gap:6px; font-size:0.80rem; opacity:0.75; margin-top:8px;'>"
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
            en = st.toggle("리마인더 켜기", value=get_setting("reminder_enabled", "true").lower() == "true", key="rem_en")
            t = st.text_input("시간(HH:MM)", value=get_setting("reminder_time", "21:30"), key="rem_time")
            w = st.number_input("허용 오차(분)", min_value=1, max_value=120, value=int(get_setting("reminder_window_min", "15")), key="rem_win")
            if st.button("저장", use_container_width=True, key="rem_save"):
                set_setting("reminder_enabled", "true" if en else "false")
                set_setting("reminder_time", (t or "21:30"))
                set_setting("reminder_window_min", str(int(w)))
                st.success("저장됐어요.")

    # Current Week (main)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Current Week")
        st.markdown(
            f"<span class='pill pill-strong'>Week</span><span class='pill pill-weak'>{ws.isoformat()} ~ {(ws+timedelta(days=6)).isoformat()}</span>",
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

        # Plan add (form)
        with st.form("plan_add_form", clear_on_submit=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                plan_text = st.text_input("계획 추가(1회성)", placeholder="예: 독서 10분 / 이메일 정리", key="plan_text_input")
            with c2:
                submitted = st.form_submit_button("추가", use_container_width=True)
            if submitted:
                add_plan_task(selected, plan_text)
                st.rerun()

        # Habit manage (minimal)
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
                        key="habit_dow_input"
                    )
                habit_submit = st.form_submit_button("습관 저장", use_container_width=True)

                if habit_submit:
                    add_habit(habit_title, picked)
                    ensure_week_habit_tasks(ws)
                    st.success("습관을 저장했어요.")
                    st.rerun()

            hdf = list_habits(active_only=False)
            if hdf.empty:
                st.markdown("<div class='small'>아직 습관이 없어요.</div>", unsafe_allow_html=True)
            else:
                for _, h in hdf.iterrows():
                    hid = int(h["id"])
                    mask = str(h["dow_mask"] or "0000000")
                    days_txt = " ".join([korean_dow(i) for i in range(7) if mask[i] == "1"]) or "—"
                    active = int(h["active"]) == 1

                    a, b, c = st.columns([6, 1, 1])
                    with a:
                        st.write(f"• {h['title']}  ·  {days_txt}")
                    with b:
                        if st.button("ON" if active else "OFF", key=f"hab_toggle_{hid}", use_container_width=True):
                            set_habit_active(hid, not active)
                            ensure_week_habit_tasks(ws)
                            st.rerun()
                    with c:
                        if st.button("삭제", key=f"hab_del_{hid}", use_container_width=True):
                            delete_habit(hid)
                            st.success("습관을 삭제했어요.")
                            st.rerun()

        # Tasks list (with delete)
        df = list_tasks_for_date(selected)
        if df.empty:
            st.markdown("<div class='small'>아직 항목이 없어요.</div>", unsafe_allow_html=True)
        else:
            for _, r in df.iterrows():
                tid = int(r["id"])
                src = r["source"]
                status = r["status"]
                text = r["text"]
                reason = r["fail_reason"] or ""

                icon_src = "🔁" if src == "habit" else "📝"
                icon_status = {"todo": "⏳", "success": "✅", "fail": "❌"}.get(status, "⏳")

                st.markdown("<div class='task'>", unsafe_allow_html=True)
                top = st.columns([6, 1.2, 1.2, 1.0], gap="small")

                with top[0]:
                    st.markdown(
                        f"**{icon_status} {text}**  <span class='pill pill-weak'>{icon_src}</span>",
                        unsafe_allow_html=True,
                    )
                    if status == "fail":
                        st.caption(f"실패 원인: {reason}")

                with top[1]:
                    if st.button("성공", key=f"s_{tid}", use_container_width=True):
                        update_task_status(tid, "success")
                        st.session_state.pop(f"show_fail_{tid}", None)
                        st.rerun()

                with top[2]:
                    if st.button("실패", key=f"f_{tid}", use_container_width=True):
                        st.session_state[f"show_fail_{tid}"] = True

                with top[3]:
                    if st.button("삭제", key=f"del_{tid}", use_container_width=True):
                        delete_task(tid)
                        st.session_state.pop(f"show_fail_{tid}", None)
                        st.rerun()

                if st.session_state.get(f"show_fail_{tid}", False):
                    reason_in = st.text_input("실패 원인(한 문장)", value=reason, key=f"r_{tid}")
                    a, b = st.columns([1, 4])
                    with a:
                        if st.button("저장", key=f"save_fail_{tid}", use_container_width=True):
                            update_task_fail(tid, reason_in)
                            st.session_state[f"show_fail_{tid}"] = False
                            st.rerun()
                    with b:
                        st.caption("짧아도 좋아요.")
                st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Screen: Failure Report
# =========================
def screen_failures():
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
        st.markdown(f"<div style='text-align:center; font-weight:650;'>{ws.isoformat()} ~ {we.isoformat()}</div>", unsafe_allow_html=True)
    with nav[2]:
        if st.button("〉", use_container_width=True, key="fw_next", disabled=(offset == 0)):
            st.session_state["fail_week_offset"] = max(0, offset - 1)
            st.rerun()

    df = get_tasks_range(ws, we)
    if df.empty:
        st.info("이 주에는 기록이 없어요.")
        return

    df = df.copy()
    df["task_date"] = pd.to_datetime(df["task_date"]).dt.date

    # --- Weekly fail chart (Mon..Sun order fixed + smaller height)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 주간 실패 차트")

    fails = df[df["status"] == "fail"].copy()

    days = week_days(ws)  # Mon..Sun in order
    day_counts = []
    for d in days:
        day_counts.append(
            {"dow": korean_dow(d.weekday()), "order": d.weekday(), "fail_count": int((fails["task_date"] == d).sum())}
        )
    chart_df = pd.DataFrame(day_counts)

    # Altair: keep order + smaller height
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("dow:N", sort=["월", "화", "수", "목", "금", "토", "일"], title=None),
            y=alt.Y("fail_count:Q", title=None),
            tooltip=["dow", "fail_count"],
        )
        .properties(height=170)
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    api_key = effective_openai_key()
    model = get_setting("openai_model", "gpt-4o-mini")

    # --- Weekly reason analysis
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 원인 주간 분석")

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
                    ex = g.get("examples", []) or []
                    if ex:
                        for s in ex[:3]:
                            st.write(f"- {s}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # --- Personalized AI coaching + chatbot (no extra header box)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 맞춤형 AI코칭")

    if not api_key:
        st.info("OpenAI 키가 설정되면 코칭/챗봇이 표시돼요. (하단에서 키 입력)")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    all_fail = get_all_failures(limit=350)
    if all_fail.empty:
        st.write("아직 실패 데이터가 없어요.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # repeated flags (2주 이상) — 반드시 creative advice로 이어지게 prompt에 강제
    flags = repeated_reason_flags(all_fail.rename(columns={"fail_reason": "fail_reason", "task_date": "task_date"}))

    # build coaching payload (recent sample with plan/habit)
    items: List[Dict[str, Any]] = []
    for _, r in all_fail.head(80).iterrows():
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

    signals = compute_user_signals(days=28)

    if st.button("코칭 생성/갱신", use_container_width=True, key="overall_coach_btn"):
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
                    # (중요) 반복이면 creative를 반드시 보여주기
                    if creative:
                        st.markdown("**2주+ 반복이면: 창의적 대안**")
                        for tip in creative[:3]:
                            st.write(f"- {tip}")
    else:
        st.caption("‘코칭 생성/갱신’을 눌러 코칭을 받아보세요.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ---- Chatbot (keep, but remove the extra heading block as requested)
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

        # compact context for personalization
        # (최근 14일 top reasons + 최근 실패 샘플 + signals 일부)
        end = date.today()
        start = end - timedelta(days=13)
        last14 = get_tasks_range(start, end)
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
# Top nav (2 screens)
# =========================
def top_nav():
    if "screen" not in st.session_state:
        st.session_state["screen"] = "planner"

    c1, c2, _ = st.columns([1.2, 1.5, 6])
    with c1:
        if st.button(" Planner", use_container_width=True, key="nav_plan"):
            st.session_state["screen"] = "planner"
    with c2:
        if st.button("Failure Report", use_container_width=True, key="nav_fail"):
            st.session_state["screen"] = "fail"

    st.write("")
    return st.session_state["screen"]


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="FAILOG", page_icon="🧊", layout="wide")
    inject_css()
    init_db()

    st.markdown("# FAILOG")
    st.markdown("<div class='small'>실패를 성공으로! 계획과 습관의 실패를 기록하고 맞춤형 코칭을 받아보자</div>", unsafe_allow_html=True)
    st.write("")

    screen = top_nav()
    if screen == "planner":
        screen_planner()
    else:
        screen_failures()

    render_openai_bottom_panel()


if __name__ == "__main__":
    main()

