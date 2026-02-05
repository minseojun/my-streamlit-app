# app.py
# ------------------------------------------------------------
# 2-screen Streamlit app (깔끔/단순 버전)
# 1) 메인: 달력형 플래너
#    - Month(작게) + Current Week(크게)
#    - 날짜 선택 → 체크리스트(계획 + 습관)
#    - 계획 추가(해당 날짜 1회성)
#    - 습관 추가(반복): 월~일 선택 → 해당 주/날짜에 자동 생성
#    - 각 항목: [성공] / [실패] 버튼
#      - 실패 누르면 해당 항목 아래에 실패 원인 입력칸 노출 → 저장
#    - 앱 내부 리마인더(팝업/배너): 설정 시간대에 오늘 todo가 남아있으면 toast + info
#
# 2) 서브: 실패 화면
#    - 주간 실패 차트(이번 주 기본, < 버튼으로 이전 주 이동)
#    - 주간 실패 원인 분석(LLM): 이번 주 실패 이유를 3개 이내로 묶어 요약
#    - 전체(누적) AI 코칭(LLM): 공통 원인 3개 이내 + 실행 조언 + 2주 이상 반복이면 창의 조언
#    - 챗봇: 사용자가 질문/대화 가능(코칭 톤 유지)
#
# OpenAI 키
#  - 하단 입력칸(사이드바 X)
#  - "로컬 저장" 스위치로 DB 저장 여부 선택(장기 사용)
#
# Run:
#   pip install streamlit pandas openai
#   streamlit run app.py
# ------------------------------------------------------------

import json
import re
import sqlite3
from datetime import date, datetime, timedelta, time
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# 스타일(최소/깔끔)
# =========================
def inject_css():
    st.markdown(
        """
<style>
.block-container { max-width: 1100px; padding-top: 1.1rem; padding-bottom: 2rem; }
h1,h2,h3 { letter-spacing: -0.02em; }
.small { color: rgba(49,51,63,0.65); font-size: 0.92rem; }
.card { border: 1px solid rgba(49,51,63,0.12); border-radius: 16px; padding: 14px 14px; background: rgba(255,255,255,0.9); }
.pill { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.14); font-size:0.85rem; margin-right:6px; }
.pill-strong { background: rgba(0,120,212,0.08); border-color: rgba(0,120,212,0.25); }
.pill-weak { background: rgba(0,0,0,0.02); }
.task { border: 1px solid rgba(49,51,63,0.12); border-radius: 14px; padding: 10px 10px; }
.task + .task { margin-top: 8px; }
hr { margin: 1.2rem 0; }
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

    # 1회성 계획 + 습관으로 생성된 항목 모두 tasks에 저장
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
          dow_mask TEXT NOT NULL,  -- 7 chars '0'/'1' for Mon..Sun
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

    # defaults
    defaults = {
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        "reminder_enabled": "true",
        "reminder_time": "21:30",       # HH:MM
        "reminder_window_min": "15",    # minutes
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
# 날짜/달력 헬퍼 (월~일)
# =========================
def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_days(ws: date) -> List[date]:
    return [ws + timedelta(days=i) for i in range(7)]


def korean_dow(i: int) -> str:
    return ["월", "화", "수", "목", "금", "토", "일"][i]


def month_grid(year: int, month: int) -> List[List[Optional[date]]]:
    first = date(year, month, 1)
    first_wd = first.weekday()  # Mon=0
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
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
# 습관/계획 CRUD + 자동 생성
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


def ensure_week_habit_tasks(ws: date):
    """앱 열었을 때: 해당 주에 필요한 습관 항목을 자동으로 tasks에 생성(중복 방지 UNIQUE)."""
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
        INSERT OR IGNORE INTO tasks
          (task_date, text, source, habit_id, status, fail_reason, created_at, updated_at)
        VALUES (?,?,?,?, 'todo', NULL, ?, ?)
        """,
        (d.isoformat(), text, "plan", None, now_iso(), now_iso()),
    )
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
    c.execute(
        "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
        (status, now_iso(), task_id),
    )
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
        SELECT id, task_date, text, source, status, fail_reason
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
        SELECT task_date, text, fail_reason
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
# 앱 내부 리마인더
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
    cur = c.cursor()
    row = cur.execute(
        "SELECT COUNT(*) FROM tasks WHERE task_date=? AND status='todo'",
        (today,),
    ).fetchone()
    c.close()
    return int(row[0] if row else 0)


# =========================
# OpenAI 키(하단 입력)
# =========================
def effective_openai_key() -> str:
    # 세션 우선, 없으면 DB 저장 키 사용
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
# 반복(2주+) 감지: 실패 원인 텍스트 기준
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
# LLM: 주간 분석 / 전체 코칭 / 챗봇
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
      "summary":"2~3문장 설명",
      "actionable_advice":["실행 조언1","실행 조언2","실행 조언3"],
      "creative_advice_when_repeated_2w":["(반복이면)창의 조언1","..."]
    }
  ]
}
규칙:
- top_causes 최대 3개
- actionable_advice는 작고 구체적으로
- 비난/자책 유도 금지
- repeated_2w=true 항목이 있으면 해당 원인에 creative_advice_when_repeated_2w 포함
- 반복 없으면 creative_advice_when_repeated_2w는 []
"""


def llm_weekly_reason_analysis(api_key: str, model: str, reasons: List[str]) -> Dict[str, Any]:
    client = openai_client(api_key)
    prompt = f"""
너는 사용자의 실패 이유를 읽고, 주간 기준으로 공통 원인을 최대 3개로 묶어 요약해.
입력은 사용자가 직접 쓴 실패 이유 목록이야.

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
        temperature=0.4,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {"groups": []}


def llm_overall_coaching(api_key: str, model: str, fail_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = openai_client(api_key)
    prompt = f"""
{BASE_COACH_PROMPT}

입력 데이터(최근 실패 기록):
{json.dumps(fail_items, ensure_ascii=False, indent=2)}

{COACH_SCHEMA}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a supportive coaching assistant. Output must be valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
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
# 하단 OpenAI 설정 UI
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
        )
    with col2:
        model = st.text_input("모델", value=get_setting("openai_model", "gpt-4o-mini"))
    with col3:
        save = st.toggle("로컬 저장", value=False, help="공용 PC면 끄는 걸 추천")

    b1, b2 = st.columns([1, 4])
    with b1:
        if st.button("적용", use_container_width=True):
            st.session_state["openai_api_key"] = api_key.strip()
            set_setting("openai_model", (model.strip() or "gpt-4o-mini"))
            if save:
                set_setting("openai_api_key", api_key.strip())
            st.success("적용됐어요.")
    with b2:
        st.caption("키가 없으면 실패 분석/코칭/챗봇이 동작하지 않아요.")


# =========================
# 화면 1: 플래너
# =========================
def screen_planner():
    st.markdown("## 📅 플래너")
    st.markdown("<div class='small'>Month는 전체 흐름, 아래는 <b>현재 주</b>를 크게 보여줘요.</div>", unsafe_allow_html=True)

    # state
    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = date.today()

    selected = st.session_state["selected_date"]
    ws = week_start(selected)

    # 습관 자동 생성
    ensure_week_habit_tasks(ws)

    # reminder
    if get_setting("reminder_enabled", "true").lower() == "true":
        rt = parse_hhmm(get_setting("reminder_time", "21:30"))
        win = int(get_setting("reminder_window_min", "15"))
        if should_remind(datetime.now(), rt, win):
            todos = count_today_todos()
            if todos > 0:
                st.toast(f"⏰ 아직 체크하지 않은 항목이 {todos}개 있어요", icon="⏰")
                st.info("오늘은 ‘완벽’ 말고 ‘체크’만 해도 충분해요. 실패여도 한 문장 남기면 내일이 쉬워져요.")

    left, right = st.columns([1.05, 1.95], gap="large")

    # ---- Month (compact)
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Month")

        y, m = selected.year, selected.month
        nav = st.columns([1, 2, 1])
        with nav[0]:
            if st.button("◀", use_container_width=True):
                if m == 1:
                    y -= 1
                    m = 12
                else:
                    m -= 1
                st.session_state["selected_date"] = date(y, m, 1)
                st.rerun()
        with nav[1]:
            st.markdown(f"<div style='text-align:center; font-weight:650; font-size:1.05rem;'>{y}.{m:02d}</div>", unsafe_allow_html=True)
        with nav[2]:
            if st.button("▶", use_container_width=True):
                if m == 12:
                    y += 1
                    m = 1
                else:
                    m += 1
                st.session_state["selected_date"] = date(y, m, 1)
                st.rerun()

        st.markdown(
            "<div style='display:grid; grid-template-columns: repeat(7, 1fr); gap:6px; font-size:0.82rem; opacity:0.7; margin-top:8px;'>"
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
                    cols[i].markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
                    continue

                label = f"{d.day}"
                if d == today:
                    label = f"•{d.day}"

                if cols[i].button(label, key=f"cal_{d.isoformat()}", use_container_width=True):
                    st.session_state["selected_date"] = d
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Simple reminder settings (kept minimal; still 2 screens)
        with st.expander("알림 설정", expanded=False):
            en = st.toggle("리마인더 켜기", value=get_setting("reminder_enabled", "true").lower() == "true")
            t = st.text_input("시간(HH:MM)", value=get_setting("reminder_time", "21:30"))
            w = st.number_input("허용 오차(분)", min_value=1, max_value=120, value=int(get_setting("reminder_window_min", "15")))
            if st.button("저장", use_container_width=True):
                set_setting("reminder_enabled", "true" if en else "false")
                set_setting("reminder_time", (t or "21:30"))
                set_setting("reminder_window_min", str(int(w)))
                st.success("저장됐어요.")

    # ---- Current Week (main)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Current Week")
        st.markdown(
            f"<span class='pill pill-strong'>주간</span><span class='pill pill-weak'>{ws.isoformat()} ~ {(ws+timedelta(days=6)).isoformat()}</span>",
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

        # ---- Add plan (1-time)
        c1, c2 = st.columns([4, 1])
        with c1:
            new_plan = st.text_input("계획 추가(1회성)", placeholder="예: 독서 10분 / 이메일 정리", key="new_plan_text")
        with c2:
            if st.button("추가", use_container_width=True):
                add_plan_task(selected, new_plan)
                st.session_state["new_plan_text"] = ""
                st.rerun()

        # ---- Add habit (minimal)
        with st.expander("습관 추가(반복)", expanded=False):
            hc1, hc2 = st.columns([3, 2])
            with hc1:
                habit_title = st.text_input("습관 이름", placeholder="예: 운동 10분", key="habit_title")
            with hc2:
                dow_labels = [f"{korean_dow(i)}" for i in range(7)]
                picked = st.multiselect("반복 요일", options=list(range(7)), format_func=lambda x: dow_labels[x], default=[0, 1, 2, 3, 4])
            if st.button("습관 저장", use_container_width=True):
                add_habit(habit_title, picked)
                st.session_state["habit_title"] = ""
                ensure_week_habit_tasks(ws)
                st.success("습관을 저장했어요. 이번 주부터 자동으로 체크리스트에 떠요.")
                st.rerun()

            # show active habits compact
            hdf = list_habits(active_only=False)
            if not hdf.empty:
                st.markdown("<div class='small'>현재 습관</div>", unsafe_allow_html=True)
                for _, h in hdf.iterrows():
                    mask = str(h["dow_mask"])
                    days_txt = "".join([korean_dow(i) if mask[i] == "1" else "" for i in range(7)])
                    a, b = st.columns([5, 1])
                    with a:
                        st.write(f"• {h['title']}  ·  {days_txt if days_txt else '—'}")
                    with b:
                        active = int(h["active"]) == 1
                        if st.button("ON" if active else "OFF", key=f"hab_{h['id']}", use_container_width=True):
                            set_habit_active(int(h["id"]), not active)
                            ensure_week_habit_tasks(ws)
                            st.rerun()

        # ---- Task list
        df = list_tasks_for_date(selected)
        if df.empty:
            st.markdown("<div class='small'>아직 항목이 없어요. 계획을 하나 추가하거나 습관을 만들어보세요.</div>", unsafe_allow_html=True)
        else:
            for _, r in df.iterrows():
                tid = int(r["id"])
                src = r["source"]  # plan/habit
                status = r["status"]
                text = r["text"]
                reason = r["fail_reason"] or ""

                icon_src = "🔁" if src == "habit" else "📝"
                icon_status = {"todo": "⏳", "success": "✅", "fail": "❌"}.get(status, "⏳")

                st.markdown("<div class='task'>", unsafe_allow_html=True)
                top = st.columns([6, 1.2, 1.2], gap="small")

                with top[0]:
                    st.markdown(f"**{icon_status} {text}**  <span class='pill pill-weak'>{icon_src}</span>", unsafe_allow_html=True)
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

                # fail editor
                if st.session_state.get(f"show_fail_{tid}", False):
                    reason_in = st.text_input("실패 원인(한 문장)", value=reason, key=f"r_{tid}")
                    a, b = st.columns([1, 4])
                    with a:
                        if st.button("저장", key=f"save_fail_{tid}", use_container_width=True):
                            update_task_fail(tid, reason_in)
                            st.session_state[f"show_fail_{tid}"] = False
                            st.rerun()
                    with b:
                        st.caption("짧아도 좋아요. ‘무슨 조건 때문에’가 핵심이에요.")
                st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 화면 2: 실패 화면
# =========================
def screen_failures():
    st.markdown("## ⚠️ 실패")
    st.markdown("<div class='small'>이번 주를 중심으로, <b>&lt;</b> 버튼으로 이전 주 기록을 볼 수 있어요.</div>", unsafe_allow_html=True)

    if "fail_week_offset" not in st.session_state:
        st.session_state["fail_week_offset"] = 0

    offset = int(st.session_state["fail_week_offset"])
    base = date.today() - timedelta(days=7 * offset)
    ws = week_start(base)
    we = ws + timedelta(days=6)

    nav = st.columns([1, 3, 1])
    with nav[0]:
        if st.button("〈", use_container_width=True):
            st.session_state["fail_week_offset"] += 1
            st.rerun()
    with nav[1]:
        st.markdown(f"<div style='text-align:center; font-weight:650;'>{ws.isoformat()} ~ {we.isoformat()}</div>", unsafe_allow_html=True)
    with nav[2]:
        if st.button("〉", use_container_width=True, disabled=(offset == 0)):
            st.session_state["fail_week_offset"] = max(0, offset - 1)
            st.rerun()

    df = get_tasks_range(ws, we)
    if df.empty:
        st.info("이 주에는 기록이 없어요.")
        return

    df = df.copy()
    df["task_date"] = pd.to_datetime(df["task_date"]).dt.date

    # --- Weekly fail chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 주간 실패 차트")
    fails = df[df["status"] == "fail"].copy()
    # counts per day
    day_counts = {d: 0 for d in week_days(ws)}
    for d, g in fails.groupby("task_date"):
        day_counts[d] = len(g)
    chart_df = pd.DataFrame({"date": list(day_counts.keys()), "fail_count": list(day_counts.values())})
    chart_df["label"] = chart_df["date"].apply(lambda d: f"{korean_dow(d.weekday())}\n{d.day}")
    st.bar_chart(chart_df.set_index("label")["fail_count"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # --- Weekly reason analysis (LLM)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 실패 원인 분석(주간)")

    api_key = effective_openai_key()
    model = get_setting("openai_model", "gpt-4o-mini")

    weekly_reasons = [r for r in fails["fail_reason"].fillna("").tolist() if str(r).strip()]

    if not api_key:
        st.info("OpenAI 키가 설정되면 주간 원인 분석이 표시돼요. (하단에서 키 입력)")
    elif len(weekly_reasons) == 0:
        st.write("이번 주에는 실패 원인 입력이 아직 없어요. 실패 시 한 문장만 남겨도 분석이 좋아져요.")
    else:
        if st.button("주간 분석 생성/갱신", use_container_width=True):
            try:
                st.session_state["weekly_analysis"] = llm_weekly_reason_analysis(api_key, model, weekly_reasons)
            except Exception as e:
                st.error(f"분석 생성 실패: {type(e).__name__}")

        analysis = st.session_state.get("weekly_analysis")
        if analysis and isinstance(analysis, dict):
            groups = analysis.get("groups", [])
            if not groups:
                st.write("분석 결과가 비어 있어요. 이유를 조금 더 모은 뒤 다시 시도해보세요.")
            else:
                for g in groups[:3]:
                    with st.container(border=True):
                        st.markdown(f"**{g.get('cause','원인')}**  ·  ~{g.get('estimated_count',0)}회")
                        st.write(g.get("description", ""))
                        ex = g.get("examples", []) or []
                        if ex:
                            st.caption("예시")
                            for s in ex[:3]:
                                st.write(f"- {s}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # --- Overall coaching + Chatbot
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### AI 코칭(누적)")

    if not api_key:
        st.info("OpenAI 키가 설정되면 코칭/챗봇이 표시돼요. (하단에서 키 입력)")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    all_fail = get_all_failures(limit=250)
    if all_fail.empty:
        st.write("아직 실패 데이터가 없어요. 👍")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # repeated flags across all failures
    flags = repeated_reason_flags(all_fail.rename(columns={"fail_reason": "fail_reason", "task_date": "task_date"}))

    # build coaching payload (recent sample)
    items: List[Dict[str, Any]] = []
    for _, r in all_fail.head(60).iterrows():
        reason = str(r["fail_reason"] or "")
        rnorm = normalize_reason(reason)
        items.append(
            {
                "date": str(r["task_date"]),
                "task": str(r["text"]),
                "reason": reason,
                "repeated_2w": bool(flags.get(rnorm, False)),
            }
        )

    colA, colB = st.columns([1.2, 2.8])
    with colA:
        if st.button("코칭 생성/갱신", use_container_width=True):
            try:
                st.session_state["overall_coach"] = llm_overall_coaching(api_key, model, items)
            except Exception as e:
                st.error(f"코칭 생성 실패: {type(e).__name__}")

    coach = st.session_state.get("overall_coach")
    if coach and isinstance(coach, dict):
        top = coach.get("top_causes", []) or []
        if not top:
            st.write("코칭 결과가 비어 있어요. 실패 이유를 더 모은 뒤 다시 시도해보세요.")
        else:
            for i, c in enumerate(top[:3], start=1):
                with st.container(border=True):
                    st.markdown(f"**{i}) {c.get('cause','원인')}**")
                    st.write(c.get("summary", ""))
                    st.markdown("**실행 조언(현실 버전)**")
                    for tip in (c.get("actionable_advice") or [])[:3]:
                        st.write(f"- {tip}")
                    creative = c.get("creative_advice_when_repeated_2w") or []
                    if creative:
                        st.markdown("**2주+ 반복이면: 다른 각도의 대안(창의 버전)**")
                        for tip in creative[:2]:
                            st.write(f"- {tip}")
    else:
        st.caption("‘코칭 생성/갱신’을 눌러 누적 코칭을 받아보세요.")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 챗봇")
    st.markdown("<div class='small'>코칭 톤(비난 없이, 실행 가능/현실적인 조언)으로 답해요.</div>", unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # chat history render
    for m in st.session_state["chat_messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("무엇이든 물어보세요 (예: 이번 주 실패를 줄이는 한 가지 실험은?)")
    if user_msg:
        st.session_state["chat_messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        # compact context for system
        # quick stats: last 14 days fail reasons top
        today = date.today()
        last14 = get_tasks_range(today - timedelta(days=13), today)
        last14_fail = last14[last14["status"] == "fail"]
        top_reasons = (
            last14_fail["fail_reason"].fillna("").map(lambda s: s.strip()).value_counts().head(5).to_dict()
            if not last14_fail.empty
            else {}
        )

        system_context = f"""
너는 실패 기록 기반 코칭 챗봇이야.
원칙:
- 비난/자책 유도 금지
- 실행 가능하고 현실적인 조언(작게, 구체적으로)
- 사용자의 상황을 '조건' 관점에서 다뤄
- 반복 실패(2주+)가 보이면 다른 각도의 창의적 대안을 제시

사용자 데이터 요약:
- 최근 14일 실패 이유 상위: {json.dumps(top_reasons, ensure_ascii=False)}
- 누적 실패 샘플(최근 10개): {json.dumps(items[:10], ensure_ascii=False)}

대화에서는 질문에 답하면서, 필요하면 '다음에 시도할 1가지 실험'을 제안해.
""".strip()

        try:
            assistant_text = llm_chat(api_key, model, system_context, st.session_state["chat_messages"][-12:])
        except Exception as e:
            assistant_text = f"(OpenAI 호출 오류: {type(e).__name__}) 키/모델을 확인해 주세요."

        st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.write(assistant_text)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 상단 네비(2화면)
# =========================
def top_nav():
    if "screen" not in st.session_state:
        st.session_state["screen"] = "planner"

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("📅 플래너", use_container_width=True):
            st.session_state["screen"] = "planner"
    with c2:
        if st.button("⚠️ 실패", use_container_width=True):
            st.session_state["screen"] = "fail"

    st.write("")
    return st.session_state["screen"]


# =========================
# Main
# =========================
def main():
    st.set_page_config(page_title="Planner + Fail Coach", page_icon="🧭", layout="wide")
    inject_css()
    init_db()

    st.markdown("# 🧭 Planner")
    st.markdown("<div class='small'>달력형 플래너 + 실패 분석 + 코칭(비난 없이)</div>", unsafe_allow_html=True)
    st.write("")

    screen = top_nav()

    if screen == "planner":
        screen_planner()
    else:
        screen_failures()

    # 하단 OpenAI 설정(요청사항)
    render_openai_bottom_panel()


if __name__ == "__main__":
    main()
