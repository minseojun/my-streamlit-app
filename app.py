import json
import re
import sqlite3
import uuid
from datetime import date, datetime, timedelta, time
from typing import Optional, List, Dict, Any

import pandas as pd
import streamlit as st
import altair as alt
from zoneinfo import ZoneInfo
from streamlit_local_storage import LocalStorage

# --- 라이브러리 체크 ---
try:
    from streamlit_autorefresh import st_autorefresh
except:
    st_autorefresh = None

try:
    from openai import OpenAI
except:
    OpenAI = None

# --- 설정 ---
KST = ZoneInfo("Asia/Seoul")
DB_PATH = "planner.db"

# --- UI & CSS (기존 디자인 유지) ---
def inject_css():
    st.markdown("""
    <style>
    .block-container { max-width: 1120px; padding-top: 1.0rem; }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 420px at 30% 0%, rgba(160,196,242,0.28), rgba(255,255,255,0) 60%),
                    linear-gradient(180deg, rgba(160,196,242,0.18) 0%, rgba(255,255,255,1) 55%);
    }
    .card { border: 1px solid rgba(160,196,242,0.58); border-radius: 18px; padding: 14px; background: rgba(255,255,255,0.94); box-shadow: 0 10px 26px rgba(160,196,242,0.14); margin-bottom:15px; }
    .task { border: 1px solid rgba(160,196,242,0.46); border-radius: 16px; padding: 10px; background: rgba(255,255,255,0.95); margin-top: 8px; }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; border:1px solid rgba(160,196,242,0.60); font-size:0.82rem; background: rgba(255,255,255,0.80); }
    .pill-strong { background: rgba(160,196,242,0.28); border-color: rgba(160,196,242,0.88); }
    </style>
    """, unsafe_allow_html=True)

# --- 로컬 스토리지 핵심 수정 (TypeError 해결) ---
def ls() -> LocalStorage:
    if "ls_obj" not in st.session_state:
        st.session_state["ls_obj"] = LocalStorage()
    return st.session_state["ls_obj"]

def ls_get(key: str) -> Optional[str]:
    # TypeError 방지를 위해 key 인자를 명시하지 않고 호출
    val = ls().getItem(key)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None

def ls_set(key: str, val: str):
    ls().setItem(key, val)

# --- DB 핸들링 (유저별 격리) ---
def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    c = conn()
    c.execute("""CREATE TABLE IF NOT EXISTS habits (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, title TEXT, dow_mask TEXT, active INTEGER DEFAULT 1, created_at TEXT, updated_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, task_date TEXT, text TEXT, source TEXT, habit_id INTEGER, status TEXT DEFAULT 'todo', fail_reason TEXT, created_at TEXT, updated_at TEXT, UNIQUE(user_id, task_date, source, habit_id, text))""")
    c.commit()
    c.close()

def now_iso():
    return datetime.now(KST).isoformat(timespec="seconds")

# --- 데이터 백업/복구 (기기 독립성의 핵심) ---
def save_snapshot(user_id: str):
    try:
        c = conn()
        habits = pd.read_sql_query("SELECT * FROM habits WHERE user_id=?", c, params=(user_id,)).to_dict(orient="records")
        tasks = pd.read_sql_query("SELECT * FROM tasks WHERE user_id=?", c, params=(user_id,)).to_dict(orient="records")
        c.close()
        snapshot = {"habits": habits, "tasks": tasks, "at": now_iso()}
        ls_set(f"failog_snap_{user_id}", json.dumps(snapshot, ensure_ascii=False))
    except:
        pass

def restore_from_snapshot_if_needed(user_id: str):
    c = conn()
    row = c.execute("SELECT COUNT(*) FROM tasks WHERE user_id=?", (user_id,)).fetchone()
    if row and row[0] > 0: 
        c.close()
        return

    snap_str = ls_get(f"failog_snap_{user_id}")
    if not snap_str: 
        c.close()
        return

    try:
        data = json.loads(snap_str)
        cur = c.cursor()
        for h in data.get("habits", []):
            cur.execute("INSERT OR IGNORE INTO habits (id, user_id, title, dow_mask, active, created_at, updated_at) VALUES (?,?,?,?,?,?,?)", 
                        (h['id'], user_id, h['title'], h['dow_mask'], h['active'], h.get('created_at', now_iso()), h.get('updated_at', now_iso())))
        for t in data.get("tasks", []):
            cur.execute("INSERT OR IGNORE INTO tasks (id, user_id, task_date, text, source, habit_id, status, fail_reason, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (t['id'], user_id, t['task_date'], t['text'], t['source'], t.get('habit_id'), t['status'], t.get('fail_reason'), t.get('created_at', now_iso()), t.get('updated_at', now_iso())))
        c.commit()
    except:
        pass
    finally:
        c.close()

# --- 비즈니스 로직 (기존 기능 복구) ---
def list_habits(user_id: str, active_only=True):
    c = conn()
    q = f"SELECT * FROM habits WHERE user_id=? {'AND active=1' if active_only else ''} ORDER BY id DESC"
    df = pd.read_sql_query(q, c, params=(user_id,))
    c.close()
    return df

def add_plan_task(user_id: str, d: date, text: str):
    if not text.strip(): return
    c = conn()
    c.execute("INSERT INTO tasks (user_id, task_date, text, source, created_at, updated_at) VALUES (?,?,?,?,?,?)",
              (user_id, d.isoformat(), text.strip(), "plan", now_iso(), now_iso()))
    c.commit()
    c.close()
    save_snapshot(user_id)

def update_task_status(user_id: str, tid: int, status: str, reason: str = None):
    c = conn()
    c.execute("UPDATE tasks SET status=?, fail_reason=?, updated_at=? WHERE id=? AND user_id=?", (status, reason, now_iso(), tid, user_id))
    c.commit()
    c.close()
    save_snapshot(user_id)

# --- AI 및 분석 기능 (기존 프롬프트 및 로직 보존) ---
# ... (기존의 llm_overall_coaching, compute_user_signals 등은 그대로 유지됨)
def compute_user_signals(user_id: str, days: int = 28):
    end = date.today()
    start = end - timedelta(days=days - 1)
    c = conn()
    df = pd.read_sql_query("SELECT * FROM tasks WHERE user_id=? AND task_date BETWEEN ? AND ?", c, params=(user_id, start.isoformat(), end.isoformat()))
    c.close()
    if df.empty: return {"has_data": False}
    df["is_fail"] = df["status"] == "fail"
    return {"has_data": True, "counts": {"total": len(df), "fail": int(df["is_fail"].sum())}}

# --- 화면 구현 ---
def screen_planner(user_id):
    st.markdown("## 📅 Planner")
    if "selected_date" not in st.session_state: st.session_state.selected_date = date.today()
    sel = st.session_state.selected_date
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        new_d = st.date_input("날짜 선택", sel)
        if new_d != sel:
            st.session_state.selected_date = new_d
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"### {sel.isoformat()}")
        with st.form("add_p", clear_on_submit=True):
            txt = st.text_input("새로운 계획")
            if st.form_submit_button("추가"):
                add_plan_task(user_id, sel, txt)
                st.rerun()
        
        c = conn()
        tasks = pd.read_sql_query("SELECT * FROM tasks WHERE user_id=? AND task_date=? ORDER BY id DESC", c, params=(user_id, sel.isoformat()))
        c.close()
        
        for _, r in tasks.iterrows():
            st.markdown(f"<div class='task'><b>{r['text']}</b> ({r['status']})</div>", unsafe_allow_html=True)
            c_s, c_f = st.columns(2)
            if c_s.button("✅ 성공", key=f"s_{r['id']}"):
                update_task_status(user_id, r['id'], "success")
                st.rerun()
            if c_f.button("❌ 실패", key=f"f_{r['id']}"):
                st.session_state[f"fail_mode_{r['id']}"] = True
            
            if st.session_state.get(f"fail_mode_{r['id']}"):
                reason = st.text_input("원인 입력", key=f"re_{r['id']}")
                if st.button("원인 저장", key=f"rs_{r['id']}"):
                    update_task_status(user_id, r['id'], "fail", reason)
                    st.session_state[f"fail_mode_{r['id']}"] = False
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- 메인 실행 로직 ---
def main():
    st.set_page_config(page_title="FAILOG", layout="wide")
    inject_css()
    init_db()

    # 1. 기기 식별 (UID 로딩 및 생성)
    uid = ls_get("failog_uid")
    if uid is None:
        # UID가 아직 없으면 생성하고 저장 후 리런
        if "new_uid_generated" not in st.session_state:
            new_id = str(uuid.uuid4())
            ls_set("failog_uid", new_id)
            st.session_state.new_uid_generated = new_id
        st.info("기기 정보를 확인 중입니다... 잠시만 기다려주세요.")
        st.rerun()
        return

    user_id = uid

    # 2. 데이터 복구 (서버 리셋 시 localStorage에서 복구)
    restore_from_snapshot_if_needed(user_id)

    # 3. OpenAI 설정 로드 (로컬 저장 체크 시)
    stored_key = ls_get("failog_oa_key")
    stored_model = ls_get("failog_oa_model")
    if stored_key and "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = stored_key
    if stored_model and "openai_model" not in st.session_state:
        st.session_state.openai_model = stored_model

    # 4. 앱 구조 (네비게이션)
    st.title("🧊 FAILOG")
    menu = st.sidebar.radio("Menu", ["Planner", "Failure Report", "Settings"])
    
    if menu == "Planner":
        screen_planner(user_id)
    elif menu == "Settings":
        st.subheader("Settings")
        api_key = st.text_input("OpenAI Key", value=st.session_state.get("openai_api_key", ""), type="password")
        model = st.text_input("Model", value=st.session_state.get("openai_model", "gpt-4o-mini"))
        save_on = st.toggle("로컬 저장 (기기 기억)", value=bool(stored_key))
        if st.button("Save Settings"):
            st.session_state.openai_api_key = api_key
            st.session_state.openai_model = model
            if save_on:
                ls_set("failog_oa_key", api_key)
                ls_set("failog_oa_model", model)
            else:
                ls_set("failog_oa_key", "")
            st.success("저장되었습니다!")

    st.sidebar.markdown(f"---")
    st.sidebar.caption(f"Device: {user_id[:8]}")

if __name__ == "__main__":
    main()
