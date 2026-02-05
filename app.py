# failog_app.py
# Streamlit app: failog (계획 실패 기록 → 원인 분류(저장) → 주간 리포트/트렌드/리마인더/코칭)
#
# 포함 기능
# 1) 데일리 체크(성공/실패 + 실패 이유 + 원인 카테고리 저장)
# 2) 원인 카테고리별 파이차트/트렌드(주차별/일자별)
# 3) 습관/목표별 주간 리포트(성공률, 실패 Top 원인, 반복 실패 감지)
# 4) 알림(리마인더)
#    - 앱이 열려 있을 때: 설정한 시간대에 "미체크/대기" 항목이 있으면 화면 토스트/배너
#    - OS/캘린더용: 매일 리마인더 .ics 파일 다운로드(가장 현실적인 크로스플랫폼)
# 5) 코칭 생성(공통 원인 3개 이내 + 실행가능 조언 + 2주 이상 반복 원인에 창의 대안)
#    - OpenAI 키가 있으면 LLM으로 더 섬세하게
#    - 없으면 로컬 규칙 기반으로 동작
#
# 실행:
#   pip install streamlit pandas openai streamlit-autorefresh
#   export OPENAI_API_KEY="..."
#   streamlit run failog_app.py
#
# NOTE: Streamlit 리마인더는 "앱이 켜져 있을 때"만 동작합니다.
#       지속 푸시 알림은 별도 백엔드/모바일/브라우저 푸시가 필요하므로, 여기서는 .ics 제공이 가장 실용적입니다.

import os
import re
import json
import sqlite3
from datetime import datetime, date, timedelta, time
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import streamlit as st

# Optional: autorefresh for reminder polling
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Optional: OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "failog — 실패를 실행 전략으로 바꿔주는 코칭"
DB_PATH = os.environ.get("FAILOG_DB_PATH", "failog.db")
DEFAULT_TZ = "Asia/Seoul"  # UI 참고용(서버 시간은 환경에 의존)


# -----------------------------
# Utilities
# -----------------------------
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def today_local() -> date:
    # Streamlit 서버가 로컬 타임존이 아닐 수도 있지만, 사용자 기준 단순 사용.
    return date.today()


def normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s가-힣]", "", t)
    return t


def week_start(d: date) -> date:
    # Monday start
    return d - timedelta(days=d.weekday())


def to_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


# -----------------------------
# DB Layer (SQLite) + Migrations
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def table_columns(conn, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]


def ensure_column(conn, table: str, col: str, ddl_type: str, default_sql: Optional[str] = None):
    cols = table_columns(conn, table)
    if col in cols:
        return
    dflt = f" DEFAULT {default_sql}" if default_sql is not None else ""
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl_type}{dflt};")


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # plans
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        """
    )

    # daily logs (base)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date TEXT NOT NULL,
            plan_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','success','fail')),
            reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(log_date, plan_id),
            FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE
        );
        """
    )

    # taxonomy table for causes (editable)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cause_taxonomy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            keywords TEXT, -- JSON array string
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    # settings
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    conn.commit()

    # migrations for daily_logs: store cause classification
    ensure_column(conn, "daily_logs", "cause_name", "TEXT", None)
    ensure_column(conn, "daily_logs", "cause_source", "TEXT", "'none'")  # user|ai|rule|none
    ensure_column(conn, "daily_logs", "cause_confidence", "REAL", "0.0")  # 0~1
    ensure_column(conn, "daily_logs", "cause_updated_at", "TEXT", None)

    conn.commit()

    # seed taxonomy if empty
    n = cur.execute("SELECT COUNT(*) FROM cause_taxonomy;").fetchone()[0]
    if n == 0:
        seed = [
            ("시간/일정", "회의/야근/이동/마감 등으로 시간이 밀리거나 계획 타이밍이 깨진 경우", ["시간", "야근", "회의", "일정", "약속", "마감", "이동", "출근", "늦"]),
            ("에너지/컨디션", "피로/수면/컨디션 저하로 실행 에너지가 부족한 경우", ["피곤", "졸림", "잠", "컨디션", "지침", "아파", "두통"]),
            ("환경/방해요인", "폰/SNS/유튜브/소음/침대 등 방해자극이 강했던 경우", ["폰", "휴대폰", "유튜브", "sns", "방해", "소음", "침대", "게임", "넷플"]),
            ("계획/설계", "목표가 과도하거나 구체성이 부족해서 시작/유지가 어려웠던 경우", ["너무", "과하게", "무리", "계획", "목표", "분량", "우선순위", "정리"]),
            ("동기/의미", "의욕 저하/귀찮음/미루기/의미 부족으로 실행이 끊긴 경우", ["의욕", "동기", "귀찮", "하기싫", "의미", "미룸", "미루"]),
            ("기타(명확화 필요)", "분류가 애매하거나 이유가 불명확한 경우(다음 기록 때 한 문장 더 구체화)", []),
        ]
        for name, desc, kws in seed:
            cur.execute(
                """
                INSERT INTO cause_taxonomy (name, description, keywords, active, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (name, desc, json.dumps(kws, ensure_ascii=False), now_iso(), now_iso()),
            )
        conn.commit()

    # seed settings defaults
    def set_default(key: str, value: str):
        cur.execute(
            "INSERT OR IGNORE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now_iso()),
        )

    set_default("reminder_enabled", "true")
    set_default("reminder_time", "21:30")  # HH:MM
    set_default("reminder_window_min", "15")  # minutes
    set_default("reminder_poll_sec", "60")  # seconds
    conn.commit()

    conn.close()


# -----------------------------
# CRUD
# -----------------------------
def add_plan(title: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO plans (title, active, created_at) VALUES (?, 1, ?)",
        (title.strip(), now_iso()),
    )
    conn.commit()
    conn.close()


def set_plan_active(plan_id: int, active: bool):
    conn = get_conn()
    conn.execute("UPDATE plans SET active=? WHERE id=?", (1 if active else 0, plan_id))
    conn.commit()
    conn.close()


def list_plans(active_only: bool = False) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT id, title, active, created_at FROM plans"
    if active_only:
        q += " WHERE active=1"
    q += " ORDER BY id DESC"
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df


def ensure_daily_rows(log_date: date):
    conn = get_conn()
    cur = conn.cursor()
    plans = cur.execute("SELECT id FROM plans WHERE active=1").fetchall()
    for (pid,) in plans:
        cur.execute(
            """
            INSERT OR IGNORE INTO daily_logs
              (log_date, plan_id, status, reason, created_at, updated_at, cause_name, cause_source, cause_confidence, cause_updated_at)
            VALUES (?, ?, 'pending', NULL, ?, ?, NULL, 'none', 0.0, NULL)
            """,
            (log_date.isoformat(), pid, now_iso(), now_iso()),
        )
    conn.commit()
    conn.close()


def get_daily_logs(log_date: date) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT dl.id, dl.log_date, dl.plan_id, p.title AS plan_title,
               dl.status, dl.reason,
               dl.cause_name, dl.cause_source, dl.cause_confidence,
               dl.updated_at
        FROM daily_logs dl
        JOIN plans p ON p.id = dl.plan_id
        WHERE dl.log_date = ?
        ORDER BY p.id DESC
        """,
        conn,
        params=(log_date.isoformat(),),
    )
    conn.close()
    return df


def update_log_success(log_id: int):
    conn = get_conn()
    conn.execute(
        """
        UPDATE daily_logs
        SET status='success', reason=NULL,
            cause_name=NULL, cause_source='none', cause_confidence=0.0, cause_updated_at=NULL,
            updated_at=?
        WHERE id=?
        """,
        (now_iso(), log_id),
    )
    conn.commit()
    conn.close()


def update_log_fail(
    log_id: int,
    reason: str,
    cause_name: Optional[str],
    cause_source: str,
    cause_confidence: float,
):
    conn = get_conn()
    conn.execute(
        """
        UPDATE daily_logs
        SET status='fail', reason=?,
            cause_name=?, cause_source=?, cause_confidence=?, cause_updated_at=?,
            updated_at=?
        WHERE id=?
        """,
        (
            reason.strip() if reason else "이유 미기록",
            cause_name,
            cause_source,
            float(cause_confidence),
            now_iso(),
            now_iso(),
            log_id,
        ),
    )
    conn.commit()
    conn.close()


def update_log_pending(log_id: int):
    conn = get_conn()
    conn.execute(
        """
        UPDATE daily_logs
        SET status='pending', reason=NULL,
            cause_name=NULL, cause_source='none', cause_confidence=0.0, cause_updated_at=NULL,
            updated_at=?
        WHERE id=?
        """,
        (now_iso(), log_id),
    )
    conn.commit()
    conn.close()


def get_failures(start_date: date, end_date: date) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT dl.id, dl.log_date, dl.plan_id, p.title AS plan_title,
               dl.reason, dl.cause_name, dl.cause_source, dl.cause_confidence
        FROM daily_logs dl
        JOIN plans p ON p.id = dl.plan_id
        WHERE dl.status='fail'
          AND dl.log_date BETWEEN ? AND ?
        ORDER BY dl.log_date ASC
        """,
        conn,
        params=(start_date.isoformat(), end_date.isoformat()),
    )
    conn.close()
    return df


def get_logs_range(start_date: date, end_date: date, active_only: bool = False) -> pd.DataFrame:
    conn = get_conn()
    where_active = "AND p.active=1" if active_only else ""
    df = pd.read_sql_query(
        f"""
        SELECT dl.id, dl.log_date, dl.plan_id, p.title AS plan_title, p.active,
               dl.status, dl.reason,
               dl.cause_name, dl.cause_source, dl.cause_confidence
        FROM daily_logs dl
        JOIN plans p ON p.id = dl.plan_id
        WHERE dl.log_date BETWEEN ? AND ?
        {where_active}
        ORDER BY dl.log_date ASC, p.id DESC
        """,
        conn,
        params=(start_date.isoformat(), end_date.isoformat()),
    )
    conn.close()
    return df


# -----------------------------
# Taxonomy + Settings
# -----------------------------
def list_causes(active_only: bool = True) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT id, name, description, keywords, active, updated_at FROM cause_taxonomy"
    if active_only:
        q += " WHERE active=1"
    q += " ORDER BY id ASC"
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df


def upsert_setting(key: str, value: str):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value, now_iso()),
    )
    conn.commit()
    conn.close()


def get_setting(key: str, default: str) -> str:
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else default


def add_cause(name: str, description: str, keywords_list: List[str]):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO cause_taxonomy (name, description, keywords, active, created_at, updated_at)
        VALUES (?, ?, ?, 1, ?, ?)
        """,
        (name.strip(), description.strip(), json.dumps(keywords_list, ensure_ascii=False), now_iso(), now_iso()),
    )
    conn.commit()
    conn.close()


def set_cause_active(cause_id: int, active: bool):
    conn = get_conn()
    conn.execute(
        "UPDATE cause_taxonomy SET active=?, updated_at=? WHERE id=?",
        (1 if active else 0, now_iso(), cause_id),
    )
    conn.commit()
    conn.close()


# -----------------------------
# Classification (OpenAI / Fallback keyword)
# -----------------------------
def fallback_classify_reason(reason: str, causes_df: pd.DataFrame) -> Tuple[str, float, str]:
    """Return (cause_name, confidence, source)."""
    r = (reason or "").lower()
    best = ("기타(명확화 필요)", 0.35)
    for _, row in causes_df.iterrows():
        name = row["name"]
        try:
            kws = json.loads(row["keywords"] or "[]")
        except Exception:
            kws = []
        if not kws:
            continue
        hits = sum(1 for kw in kws if kw and kw.lower() in r)
        if hits > 0:
            conf = min(0.55 + 0.1 * hits, 0.9)
            if conf > best[1]:
                best = (name, conf)
    return best[0], best[1], "rule"


def openai_classify_reason(reason: str, cause_names: List[str]) -> Tuple[str, float, str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI not configured.")
    client = OpenAI(api_key=api_key)
    model = os.environ.get("FAILOG_OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
너는 사용자의 '계획 실패 이유'를 아래 원인 카테고리 중 하나로 분류해.
카테고리 목록: {json.dumps(cause_names, ensure_ascii=False)}

입력 실패 이유:
{reason}

규칙:
- 반드시 목록 중 하나만 선택
- 출력은 JSON만
형식:
{{"cause":"...", "confidence":0.0}}
confidence는 0~1 (확신이 낮으면 0.4~0.6)
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))

    cause = obj.get("cause", "").strip()
    conf = float(obj.get("confidence", 0.5))
    if cause not in cause_names:
        # safety: snap to 기타
        cause = "기타(명확화 필요)" if "기타(명확화 필요)" in cause_names else cause_names[-1]
        conf = min(conf, 0.55)
    conf = max(0.0, min(1.0, conf))
    return cause, conf, "ai"


def classify_reason(reason: str, prefer_openai: bool = True) -> Tuple[str, float, str]:
    causes_df = list_causes(active_only=True)
    cause_names = causes_df["name"].tolist()
    if not reason.strip():
        # 빈 이유는 기타로
        return ("기타(명확화 필요)" if "기타(명확화 필요)" in cause_names else cause_names[-1], 0.35, "rule")

    if prefer_openai:
        try:
            return openai_classify_reason(reason, cause_names)
        except Exception:
            pass
    return fallback_classify_reason(reason, causes_df)


# -----------------------------
# Repeated detection (>=14 days) by CAUSE (plan_id + cause_name)
# -----------------------------
def detect_repeated_causes_2w(failures_df: pd.DataFrame) -> Dict[Tuple[int, str], bool]:
    """
    Returns flags for (plan_id, cause_name) if failures span >= 14 days within the analysis window.
    """
    if failures_df.empty:
        return {}
    df = failures_df.copy()
    df["log_date"] = pd.to_datetime(df["log_date"]).dt.date
    df["cause_name"] = df["cause_name"].fillna("기타(명확화 필요)")
    flags: Dict[Tuple[int, str], bool] = {}
    for (pid, cause), g in df.groupby(["plan_id", "cause_name"]):
        dates = sorted(g["log_date"].tolist())
        if len(dates) >= 2 and (dates[-1] - dates[0]).days >= 14:
            flags[(int(pid), str(cause))] = True
    return flags


# -----------------------------
# Coaching Engine
# -----------------------------
COACH_SCHEMA_HINT = """
반드시 JSON만 출력해. (설명/마크다운 금지)
형식:
{
  "top_causes": [
    {
      "cause": "원인 카테고리 이름(짧게)",
      "summary": "왜 이게 공통 원인인지 2~3문장",
      "actionable_advice": ["현실적 조언 1", "현실적 조언 2", "현실적 조언 3"],
      "creative_advice_when_repeated_2w": ["(해당 원인이 2주 이상 반복된 항목이 있을 때만) 창의적 조언 1", "..."]
    }
  ],
  "tone_note": "전체 톤이 비난 없이 코칭 중심인지 점검하는 한 문장"
}
규칙:
- top_causes는 최대 3개
- actionable_advice는 '지금 당장 실행' 가능한 수준(작고 구체적)으로
- '비난/자책 유도' 표현 금지
- 2주 이상 반복(repeated_2w=true)된 원인이 있으면, 해당 원인에 creative_advice_when_repeated_2w를 반드시 포함
- 반복 원인이 없으면 creative_advice_when_repeated_2w는 빈 배열([])로
"""


def build_coach_prompt(items: List[Dict[str, Any]]) -> str:
    return f"""
너는 '실패 기록을 실행 전략으로 바꾸는' 코칭 AI야.
사용자가 적은 "실패 이유"와 "원인 카테고리" 데이터를 보고 공통 원인을 최대 3가지로 묶고,
각 원인에 대해 실행 가능하고 현실적인 개선 조언을 제시해.
추가 규칙: 만약 repeated_2w=true 인 원인(같은 원인이 2주 이상 반복)이 있다면,
그 원인에 대해 기존 조언과 결이 다른 "창의적인 대안 조언"도 제시해.
톤은 절대 비난하지 말고, 코칭/격려 중심으로.

입력 데이터:
{json.dumps(items, ensure_ascii=False, indent=2)}

{COACH_SCHEMA_HINT}
""".strip()


def openai_coach(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI not configured.")
    client = OpenAI(api_key=api_key)
    model = os.environ.get("FAILOG_OPENAI_MODEL", "gpt-4o-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a supportive coaching assistant. Output must be valid JSON only."},
            {"role": "user", "content": build_coach_prompt(items)},
        ],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def fallback_coach(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"top_causes": [], "tone_note": "기록이 비어 있어 분석 대신 다음 기록을 기다리고 있어요."}

    df = pd.DataFrame(items)
    # count by cause
    counts = df["cause"].value_counts().head(3)

    top_causes = []
    for cause, _cnt in counts.items():
        sub = df[df["cause"] == cause]
        repeated = bool(sub["repeated_2w"].fillna(False).any())

        actionable = [
            "실패가 난 날의 '첫 장애물'만 한 문장으로 적고, 내일은 그 장애물을 피하는 장치를 딱 1개만 추가해요(예: 회의 후 10분 휴식 고정).",
            "계획을 '시작 2분 버전'으로 축소해서 진입장벽을 낮춰요(예: 운동 20분 → 스트레칭 2분만).",
            "실패가 많이 나는 시간대를 파악해서, 그 시간엔 방해요인을 미리 치우는 루틴(알림 끄기/장소 이동/도구 미리 준비)을 만들어요.",
        ]
        creative = []
        if repeated:
            creative = [
                "2주 이상 반복이면, 목표를 '성과'가 아니라 '조건 실험'으로 바꿔요. 예: '운동 성공' → '운동이 되는 조건 찾기'를 1주일만 실험.",
                "트리거를 완전히 바꿔봐요. 시간(저녁→아침), 장소(집→카페/헬스장), 방식(혼자→동료/클래스) 중 하나만 교체해요.",
            ]

        top_causes.append(
            {
                "cause": cause,
                "summary": f"최근 기록에서 '{cause}' 유형이 자주 등장해요. 이건 의지 문제가 아니라 '조건/설계' 조정으로 개선될 가능성이 큰 신호예요.",
                "actionable_advice": actionable,
                "creative_advice_when_repeated_2w": creative if repeated else [],
            }
        )

    return {"top_causes": top_causes, "tone_note": "실패를 탓이 아니라 '조정 가능한 조건 데이터'로 다루는 톤을 유지했어요."}


def run_coaching(items: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    try:
        return openai_coach(items), "OpenAI"
    except Exception:
        return fallback_coach(items), "Local"


# -----------------------------
# Reminder: in-app + .ics
# -----------------------------
def parse_hhmm(s: str) -> time:
    s = s.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return time(21, 30)
    hh, mm = int(m.group(1)), int(m.group(2))
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    return time(hh, mm)


def should_show_reminder(now_dt: datetime, reminder_t: time, window_min: int) -> bool:
    target = datetime.combine(now_dt.date(), reminder_t)
    delta = abs((now_dt - target).total_seconds()) / 60.0
    return delta <= float(window_min)


def count_pending_today(d: date) -> int:
    ensure_daily_rows(d)
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT COUNT(*) FROM daily_logs dl
        JOIN plans p ON p.id=dl.plan_id
        WHERE dl.log_date=? AND p.active=1 AND dl.status='pending'
        """,
        (d.isoformat(),),
    ).fetchone()
    conn.close()
    return int(row[0] if row else 0)


def build_daily_ics(reminder_t: time) -> str:
    # Recurring daily event, floating local time
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    start = datetime.combine(today_local(), reminder_t).strftime("%Y%m%dT%H%M%S")
    uid = f"failog-reminder-{dtstamp}@local"
    # Keep it simple; users can import into Google/Apple Calendar.
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//failog//Reminder//EN
CALSCALE:GREGORIAN
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp}
DTSTART:{start}
DURATION:PT10M
RRULE:FREQ=DAILY
SUMMARY:failog 데일리 체크 리마인더
DESCRIPTION:오늘의 계획을 성공/실패로 체크하고, 실패라면 이유를 한 문장 기록해요.
END:VEVENT
END:VCALENDAR
"""
    return ics


# -----------------------------
# UI
# -----------------------------
def main():
    st.set_page_config(page_title="failog", page_icon="🧭", layout="wide")
    init_db()

    # Reminder polling
    reminder_enabled = get_setting("reminder_enabled", "true").lower() == "true"
    reminder_time = parse_hhmm(get_setting("reminder_time", "21:30"))
    reminder_window = int(get_setting("reminder_window_min", "15"))
    poll_sec = int(get_setting("reminder_poll_sec", "60"))

    if reminder_enabled and st_autorefresh is not None:
        st_autorefresh(interval=poll_sec * 1000, key="reminder_refresh")

    # In-app reminder banner
    if reminder_enabled:
        pending = count_pending_today(today_local())
        if pending > 0 and should_show_reminder(datetime.now(), reminder_time, reminder_window):
            st.toast(f"리마인더: 아직 체크하지 않은 항목이 {pending}개 있어요. (오늘만 가볍게 정리해도 충분해요)", icon="⏰")
            st.info(f"⏰ 오늘 체크가 아직 {pending}개 남아 있어요. 실패여도 괜찮아요. 한 문장만 남기면 내일이 쉬워져요.")

    st.title(APP_TITLE)
    st.caption("실패는 데이터예요. 비난 없이, 조건을 조정하는 코칭으로 바꿔요.")

    with st.expander("🔐 OpenAI 설정(선택)", expanded=False):
        st.write("- 환경변수 `OPENAI_API_KEY`가 있으면 더 섬세한 분류/코칭이 가능합니다.")
        st.write("- 없으면 로컬 규칙 기반으로 동작합니다.")
        st.code(
            "export OPENAI_API_KEY='YOUR_KEY'\n"
            "export FAILOG_OPENAI_MODEL='gpt-4o-mini'  # 선택\n"
            "streamlit run failog_app.py",
            language="bash",
        )

    tab_daily, tab_report, tab_analysis, tab_manage = st.tabs(
        ["✅ 데일리 체크", "🗓️ 주간 리포트", "📈 원인 트렌드 & 코칭", "⚙️ 관리(계획/원인/알림)"]
    )

    # -------------------------
    # Tab 1: Daily
    # -------------------------
    with tab_daily:
        colL, colR = st.columns([1, 2])

        with colL:
            selected_date = st.date_input("날짜", value=today_local(), key="daily_date")
            ensure_daily_rows(selected_date)
            st.subheader("오늘의 한 줄 코칭")
            st.write("실패는 ‘내가 부족함’이 아니라, **조건이 안 맞았다는 신호**일 때가 많아요.")

        with colR:
            st.subheader("데일리 계획 리스트")
            df = get_daily_logs(selected_date)
            causes_df = list_causes(active_only=True)
            cause_names = causes_df["name"].tolist()

            if df.empty:
                st.warning("활성화된 계획이 없어요. '관리'에서 계획을 추가해 주세요.")
            else:
                for _, row in df.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 6])
                        with c1:
                            st.markdown(f"**{row['plan_title']}**")
                            st.caption(f"상태: `{row['status']}`")
                            if row.get("cause_name"):
                                st.caption(f"원인: {row['cause_name']} ({row.get('cause_source','')}, {row.get('cause_confidence',0):.2f})")

                        with c2:
                            b1, b2, b3 = st.columns([1, 1, 1])
                            with b1:
                                if st.button("성공 ✅", key=f"succ_{row['id']}"):
                                    update_log_success(int(row["id"]))
                                    st.success("성공 체크 완료!")
                                    st.rerun()
                            with b2:
                                if st.button("대기 ↩️", key=f"pend_{row['id']}"):
                                    update_log_pending(int(row["id"]))
                                    st.info("대기로 되돌렸어요.")
                                    st.rerun()
                            with b3:
                                # placeholder for spacing
                                st.write("")

                            reason_key = f"reason_{row['id']}"
                            cause_key = f"cause_{row['id']}"

                            default_reason = row["reason"] if row["reason"] else ""
                            reason = st.text_input("실패 이유(한 문장)", value=default_reason, key=reason_key)

                            # cause selection
                            default_cause = row["cause_name"] if row["cause_name"] in cause_names else "자동 분류"
                            options = ["자동 분류"] + cause_names
                            cause_sel = st.selectbox("원인 카테고리", options=options, index=options.index(default_cause) if default_cause in options else 0, key=cause_key)

                            if st.button("실패 ❌ 저장", key=f"fail_save_{row['id']}"):
                                if cause_sel == "자동 분류":
                                    cause, conf, src = classify_reason(reason, prefer_openai=True)
                                else:
                                    cause, conf, src = cause_sel, 1.0, "user"
                                update_log_fail(int(row["id"]), reason, cause, src, conf)
                                st.warning("실패 체크 저장 완료! 기록을 남긴 것 자체가 이미 다음 성공 확률을 올렸어요.")
                                st.rerun()

    # -------------------------
    # Tab 2: Weekly Report
    # -------------------------
    with tab_report:
        st.subheader("습관/목표별 주간 리포트")
        end_d = st.date_input("리포트 종료일", value=today_local(), key="report_end")
        ws = week_start(end_d)
        we = ws + timedelta(days=6)
        st.caption(f"주간 범위: {ws.isoformat()} ~ {we.isoformat()} (월~일)")

        logs = get_logs_range(ws, we, active_only=False)
        if logs.empty:
            st.info("이 주차에는 기록이 없어요.")
        else:
            # overall summary by plan
            def plan_week_summary(df: pd.DataFrame) -> pd.DataFrame:
                x = df.copy()
                x["is_success"] = (x["status"] == "success").astype(int)
                x["is_fail"] = (x["status"] == "fail").astype(int)
                x["is_pending"] = (x["status"] == "pending").astype(int)
                g = x.groupby(["plan_id", "plan_title"], as_index=False).agg(
                    success=("is_success", "sum"),
                    fail=("is_fail", "sum"),
                    pending=("is_pending", "sum"),
                )
                g["checked"] = g["success"] + g["fail"]
                g["success_rate"] = g.apply(lambda r: (r["success"] / r["checked"]) if r["checked"] else 0.0, axis=1)
                return g.sort_values(["success_rate", "checked"], ascending=[False, False])

            summary = plan_week_summary(logs)
            summary_show = summary.copy()
            summary_show["success_rate"] = (summary_show["success_rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(summary_show, use_container_width=True, hide_index=True)

            # per-plan details
            st.markdown("### 계획별 상세")
            failures = logs[logs["status"] == "fail"].copy()
            failures["cause_name"] = failures["cause_name"].fillna("기타(명확화 필요)")
            repeated_flags = detect_repeated_causes_2w(
                get_failures(ws - timedelta(days=21), we)  # look-back 포함: 반복 감지에 유리
            )

            plans = summary[["plan_id", "plan_title"]].values.tolist()
            for pid, title in plans:
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    sub = logs[logs["plan_id"] == pid].copy()
                    # streak calculation (simple: consecutive successes ending at week end)
                    sub["log_date"] = pd.to_datetime(sub["log_date"]).dt.date
                    sub_sorted = sub.sort_values("log_date")
                    streak = 0
                    # calculate ending streak up to we
                    by_date = {r["log_date"]: r["status"] for _, r in sub_sorted.iterrows()}
                    d = we
                    while d >= ws:
                        stt = by_date.get(d, "pending")
                        if stt == "success":
                            streak += 1
                            d -= timedelta(days=1)
                        else:
                            break

                    succ = int((sub["status"] == "success").sum())
                    fail = int((sub["status"] == "fail").sum())
                    pend = int((sub["status"] == "pending").sum())
                    checked = succ + fail
                    rate = (succ / checked) if checked else 0.0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("성공", succ)
                    c2.metric("실패", fail)
                    c3.metric("대기", pend)
                    c4.metric("주간 성공률", f"{rate*100:.1f}%")

                    st.caption(f"주간 마감 기준 연속 성공(대략): {streak}일")

                    # Top causes
                    fsub = failures[failures["plan_id"] == pid]
                    if fsub.empty:
                        st.success("이번 주에는 실패 기록이 없어요. 이 페이스가 ‘기본값’이 되도록 가볍게 유지해요.")
                    else:
                        topc = fsub["cause_name"].value_counts().head(3)
                        st.write("실패 Top 원인:")
                        for cause, cnt in topc.items():
                            rep = repeated_flags.get((int(pid), str(cause)), False)
                            tag = " (2주+ 반복 신호)" if rep else ""
                            st.write(f"- {cause}: {cnt}회{tag}")

                        with st.expander("실패 기록(이유/원인) 보기", expanded=False):
                            view = fsub[["log_date", "reason", "cause_name", "cause_source", "cause_confidence"]].copy()
                            st.dataframe(view, use_container_width=True, hide_index=True)

    # -------------------------
    # Tab 3: Trends & Coaching
    # -------------------------
    with tab_analysis:
        st.subheader("원인 카테고리 트렌드 & 코칭")
        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            days = st.selectbox("분석 기간(일)", [7, 14, 21, 30, 60, 90], index=1, key="an_days")
        with colB:
            end_d = st.date_input("종료일", value=today_local(), key="an_end")
        with colC:
            st.caption("저장된 원인 카테고리를 기준으로 파이/트렌드를 그리고, 공통 원인 3개 이내 코칭을 생성해요.")
            st.caption("같은 원인이 2주 이상 반복되면(원인 단위) 해당 원인에 창의적 대안을 추가합니다.")

        start_d = end_d - timedelta(days=int(days) - 1)

        failures_df = get_failures(start_d, end_d)

        # Backfill missing causes in the window (optional toggle)
        st.markdown("#### 원인 저장 상태")
        missing = int(failures_df["cause_name"].isna().sum()) if not failures_df.empty else 0
        st.write(f"- 이 기간 실패 중 원인 미저장: **{missing}건**")
        backfill = st.checkbox("이 기간의 원인 미저장 실패를 자동 분류해서 DB에 저장(추천)", value=False)

        if backfill and missing > 0:
            # classify and store
            for _, r in failures_df[failures_df["cause_name"].isna()].iterrows():
                cid = int(r["id"])
                reason = r["reason"] or ""
                cause, conf, src = classify_reason(reason, prefer_openai=True)
                # update row without changing status
                conn = get_conn()
                conn.execute(
                    """
                    UPDATE daily_logs
                    SET cause_name=?, cause_source=?, cause_confidence=?, cause_updated_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (cause, src, float(conf), now_iso(), now_iso(), cid),
                )
                conn.commit()
                conn.close()
            st.success("자동 분류 저장 완료! (이제 다음 분석이 더 정확해져요)")
            failures_df = get_failures(start_d, end_d)

        if failures_df.empty:
            st.info("이 기간엔 실패 기록이 없어요. 👍 지금의 리듬을 유지해도 충분히 좋습니다.")
        else:
            # normalize cause
            failures_df = failures_df.copy()
            failures_df["cause_name"] = failures_df["cause_name"].fillna("기타(명확화 필요)")
            failures_df["log_date"] = pd.to_datetime(failures_df["log_date"]).dt.date

            # Pie chart
            st.markdown("#### 원인 분포(파이)")
            pie_df = failures_df["cause_name"].value_counts().reset_index()
            pie_df.columns = ["cause", "count"]
            st.dataframe(pie_df, use_container_width=True, hide_index=True)
            # Streamlit native charts are limited; use bar as a clear default
            st.bar_chart(pie_df.set_index("cause"))

            # Trend (weekly)
            st.markdown("#### 원인 트렌드(주차별)")
            tmp = failures_df.copy()
            tmp["week"] = tmp["log_date"].apply(lambda d: week_start(d).isoformat())
            trend = tmp.groupby(["week", "cause_name"]).size().reset_index(name="count")
            pivot = trend.pivot(index="week", columns="cause_name", values="count").fillna(0).sort_index()
            st.line_chart(pivot)

            # Repeated cause flags (within window)
            repeated_flags = detect_repeated_causes_2w(failures_df)

            # Build coaching payload (cause-based)
            items = []
            for _, r in failures_df.iterrows():
                pid = int(r["plan_id"])
                cause = str(r["cause_name"])
                items.append(
                    {
                        "plan_title": r["plan_title"],
                        "date": str(r["log_date"]),
                        "reason": r["reason"] or "",
                        "cause": cause,
                        "repeated_2w": bool(repeated_flags.get((pid, cause), False)),
                    }
                )

            st.markdown("#### 코칭 생성")
            colX, colY = st.columns([1, 3])
            with colX:
                run_btn = st.button("코칭 생성/갱신", type="primary", key="coach_run")
            with colY:
                st.caption("OpenAI 키가 있으면 더 자연스럽고 섬세하게, 없으면 로컬 규칙 기반으로 코칭을 생성합니다.")

            if run_btn or ("coach_result" not in st.session_state):
                result, engine = run_coaching(items)
                st.session_state["coach_result"] = result
                st.session_state["coach_engine"] = engine

            result = st.session_state.get("coach_result", {})
            engine = st.session_state.get("coach_engine", "Local")
            st.write(f"사용 엔진: **{engine}**")

            top_causes = result.get("top_causes", []) if isinstance(result.get("top_causes", []), list) else []
            if not top_causes:
                st.info("아직 분류할 만큼 데이터가 부족해요. 실패 이유를 한 문장이라도 더 쌓아보면 정확도가 올라가요.")
            else:
                st.markdown("### 공통 원인 TOP (최대 3)")
                for i, c in enumerate(top_causes, start=1):
                    with st.container(border=True):
                        st.markdown(f"### {i}) {c.get('cause','(원인)')}")
                        st.write(c.get("summary", ""))

                        st.markdown("**실행 가능한 개선 조언(현실 버전)**")
                        for tip in (c.get("actionable_advice") or [])[:6]:
                            st.write(f"- {tip}")

                        creative = c.get("creative_advice_when_repeated_2w") or []
                        if creative:
                            st.markdown("**2주 이상 반복 시: 완전히 다른 각도의 대안(창의 버전)**")
                            for tip in creative[:6]:
                                st.write(f"- {tip}")

                st.caption(result.get("tone_note", ""))

            with st.expander("🔎 이번 분석에 사용된 데이터 보기", expanded=False):
                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)

    # -------------------------
    # Tab 4: Manage (Plans / Causes / Reminder)
    # -------------------------
    with tab_manage:
        st.subheader("관리")

        subtab_plans, subtab_causes, subtab_reminder, subtab_fix = st.tabs(
            ["계획", "원인 카테고리", "알림(리마인더)", "데이터 정리/수정"]
        )

        # Plans
        with subtab_plans:
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown("#### 새 계획 추가")
                new_title = st.text_input("계획/습관 이름", placeholder="예: 영어 단어 20개 / 운동 20분 / 논문 1페이지", key="new_plan")
                if st.button("추가", key="add_plan_btn"):
                    if not new_title.strip():
                        st.error("계획 이름을 입력해 주세요.")
                    else:
                        add_plan(new_title.strip())
                        st.success("추가 완료!")
                        st.rerun()

                st.markdown("---")
                st.markdown("#### 운영 팁")
                st.write("- 계획은 작을수록 성공률이 올라가요.")
                st.write("- 실패 이유는 길게 쓰지 않아도 돼요. 한 문장으로 충분해요.")
                st.write("- 반복 실패는 ‘의지’보다 설계/환경의 신호일 때가 많아요.")

            with col2:
                st.markdown("#### 내 계획 목록")
                plans_df = list_plans(active_only=False)
                if plans_df.empty:
                    st.info("아직 계획이 없어요. 왼쪽에서 추가해 주세요.")
                else:
                    for _, r in plans_df.iterrows():
                        with st.container(border=True):
                            a, b, c = st.columns([4, 2, 2])
                            with a:
                                st.markdown(f"**{r['title']}**")
                                st.caption(f"생성: {r['created_at']}")
                            with b:
                                active = bool(r["active"])
                                st.write("상태:", "활성 ✅" if active else "비활성 ⛔")
                            with c:
                                if active:
                                    if st.button("비활성화", key=f"deact_{r['id']}"):
                                        set_plan_active(int(r["id"]), False)
                                        st.rerun()
                                else:
                                    if st.button("활성화", key=f"act_{r['id']}"):
                                        set_plan_active(int(r["id"]), True)
                                        st.rerun()

        # Causes taxonomy
        with subtab_causes:
            st.markdown("#### 원인 카테고리 목록(분류 기준)")
            causes_df = list_causes(active_only=False)
            if causes_df.empty:
                st.warning("원인 카테고리가 없어요.")
            else:
                for _, r in causes_df.iterrows():
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([3, 5, 2])
                        with c1:
                            st.markdown(f"**{r['name']}**")
                            st.caption("활성 ✅" if int(r["active"]) == 1 else "비활성 ⛔")
                        with c2:
                            st.write(r["description"] or "")
                            try:
                                kws = json.loads(r["keywords"] or "[]")
                            except Exception:
                                kws = []
                            if kws:
                                st.caption("키워드: " + ", ".join(kws))
                        with c3:
                            if int(r["active"]) == 1:
                                if st.button("비활성화", key=f"cause_off_{r['id']}"):
                                    set_cause_active(int(r["id"]), False)
                                    st.rerun()
                            else:
                                if st.button("활성화", key=f"cause_on_{r['id']}"):
                                    set_cause_active(int(r["id"]), True)
                                    st.rerun()

            st.markdown("---")
            st.markdown("#### 새 원인 카테고리 추가")
            name = st.text_input("이름", placeholder="예: 장소/도구 문제", key="cause_new_name")
            desc = st.text_area("설명", placeholder="이 카테고리가 포함하는 실패의 공통 특징", key="cause_new_desc")
            kws_raw = st.text_input("키워드(쉼표로 구분)", placeholder="예: 장소, 카페, 노트북, 준비물", key="cause_new_kws")
            if st.button("원인 추가", key="cause_add_btn"):
                if not name.strip():
                    st.error("이름은 필수예요.")
                else:
                    kws = [x.strip() for x in kws_raw.split(",") if x.strip()]
                    try:
                        add_cause(name, desc, kws)
                        st.success("추가 완료!")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("이미 같은 이름의 원인이 있어요. 이름을 바꿔주세요.")

        # Reminder
        with subtab_reminder:
            st.markdown("#### 리마인더 설정")
            enabled = st.toggle("리마인더 켜기", value=reminder_enabled, key="rem_en")
            rt = st.text_input("리마인더 시간(HH:MM)", value=get_setting("reminder_time", "21:30"), key="rem_time")
            wm = st.number_input("표시 허용 오차(분)", min_value=1, max_value=120, value=int(get_setting("reminder_window_min", "15")), key="rem_win")
            ps = st.number_input("앱 내 체크 주기(초)", min_value=10, max_value=600, value=int(get_setting("reminder_poll_sec", "60")), key="rem_poll")

            if st.button("설정 저장", key="rem_save"):
                upsert_setting("reminder_enabled", "true" if enabled else "false")
                upsert_setting("reminder_time", rt.strip())
                upsert_setting("reminder_window_min", str(int(wm)))
                upsert_setting("reminder_poll_sec", str(int(ps)))
                st.success("저장했어요. (앱이 켜져 있을 때 설정 시간에 배너가 떠요)")
                st.rerun()

            st.markdown("---")
            st.markdown("#### 캘린더(구글/애플 등)로 리마인더 받기(.ics)")
            t = parse_hhmm(rt)
            ics = build_daily_ics(t)
            st.download_button(
                "📥 매일 리마인더 .ics 다운로드",
                data=ics.encode("utf-8"),
                file_name="failog_daily_reminder.ics",
                mime="text/calendar",
            )
            st.caption("다운로드 후 캘린더에 가져오기(import) 하면, 앱을 안 켜도 OS/캘린더 알림을 받을 수 있어요.")

        # Fix / Edit existing causes on logs (manual correction)
        with subtab_fix:
            st.markdown("#### 실패 기록의 원인 수정(정확도 개선)")
            d1 = st.date_input("시작일", value=today_local() - timedelta(days=14), key="fix_s")
            d2 = st.date_input("종료일", value=today_local(), key="fix_e")
            df = get_failures(d1, d2)
            if df.empty:
                st.info("선택한 기간에 실패 기록이 없어요.")
            else:
                df = df.copy()
                df["cause_name"] = df["cause_name"].fillna("기타(명확화 필요)")
                st.dataframe(df[["id", "log_date", "plan_title", "reason", "cause_name", "cause_source", "cause_confidence"]],
                             use_container_width=True, hide_index=True)

                st.markdown("원인 수정:")
                causes_df = list_causes(active_only=True)
                cause_names = causes_df["name"].tolist()
                target_id = st.number_input("수정할 log id", min_value=int(df["id"].min()), max_value=int(df["id"].max()), value=int(df["id"].min()), step=1)
                new_cause = st.selectbox("새 원인", options=cause_names, index=0)
                if st.button("원인 업데이트", key="fix_update"):
                    conn = get_conn()
                    conn.execute(
                        """
                        UPDATE daily_logs
                        SET cause_name=?, cause_source='user', cause_confidence=1.0, cause_updated_at=?, updated_at=?
                        WHERE id=?
                        """,
                        (new_cause, now_iso(), now_iso(), int(target_id)),
                    )
                    conn.commit()
                    conn.close()
                    st.success("업데이트 완료! 다음 분석부터 더 정확해져요.")
                    st.rerun()

    st.markdown("---")
    st.caption(f"© failog • Timezone hint: {DEFAULT_TZ} • DB: {DB_PATH}")


if __name__ == "__main__":
    main()
