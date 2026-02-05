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

# 필수 라이브러리 체크
from streamlit_local_storage import LocalStorage

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- 설정 ---
KST = ZoneInfo("Asia/Seoul")
DB_PATH = "planner.db"
ACCENT = "#A0C4F2"

# --- UI/CSS ---
def inject_css():
    st.markdown(f"""
    <style>
    .block-container {{ max-width: 1120px; padding-top: 1.0rem; }}
    [data-testid="stAppViewContainer"] {{
        background: radial-gradient(1200px 420px at 30% 0%, rgba(160,196,242,0.28), rgba(255,255,255,0) 60%),
                    linear-gradient(180deg, rgba(160,196,242,0.18) 0%, rgba(255,255,255,1) 55%);
    }}
    .card {{
        border: 1px solid rgba(160,196,242,0.58); border-radius: 18px;
        padding: 14px; background: rgba(255,255,255,0.94);
        box-shadow: 0 10px 26px rgba(160,196,242,0.14); margin-bottom: 20px;
    }}
    .task {{
        border: 1px solid rgba(160,196,242,0.46); border-radius: 16px;
        padding: 10px; background: rgba(255,255,255,0.95); margin-top: 8px;
    }}
    .pill {{
        display:inline-flex; align-items:center; gap:6px; padding:4px 10px;
        border-radius:999px; border:1px solid rgba(160,196,242,0.60);
        font-size:0.82rem; background: rgba(255,255,255,0.80);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- LocalStorage Helper ---
def ls() -> LocalStorage:
    if "ls_obj" not in st.session_state:
        st.session_state["ls_obj"] = LocalStorage()
    return st.session_state["ls_obj"]

def ls_set(key: str, val: str):
    ls().setItem(key, val)

# --- DB & Core Functions ---
def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA foreign_keys = ON;")
    return c

def init_db():
    c = conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, title TEXT, 
            dow_mask TEXT, active INTEGER DEFAULT 1, created_at TEXT, updated_at TEXT
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, task_date TEXT, 
            text TEXT, source TEXT, habit_id INTEGER, status TEXT DEFAULT 'todo', 
            fail_reason TEXT, created_at TEXT, updated_at TEXT,
            UNIQUE(user_id, task_date, source, habit_id, text)
        )""")
    c.commit()
    c.close()

def save_snapshot(user_id: str):
    """현재 유저의 데이터를 localStorage에 백업"""
    c = conn()
    habits = pd.read_sql_query("SELECT * FROM habits WHERE user_id=?", c, params=(user_id,))
    tasks = pd.read_sql_query("SELECT * FROM tasks WHERE user_id=?", c, params=(user_id,))
    c.close()
    
    snapshot = {
        "habits": habits.to_dict(orient="records"),
        "tasks": tasks.to_dict(orient="records"),
        "timestamp": datetime.now().isoformat()
    }
    ls_set(f"failog_snap_{user_id}", json.dumps(snapshot, ensure_ascii=False))

def restore_from_snapshot(user_id: str):
    """DB가 비어있을 때 localStorage 백업본에서 복구"""
    snap_str = st.session_state.get(f"ls_snap_{user_id}")
    if not snap_str: return

    c = conn()
    count = c.execute("SELECT COUNT(*) FROM tasks WHERE user_id=?", (user_id,)).fetchone()[0]
    if count > 0: 
        c.close()
        return

    data = json.loads(snap_str)
    cur = c.cursor()
    for h in data.get("habits", []):
        cur.execute("INSERT OR IGNORE INTO habits VALUES (?,?,?,?,?,?,?)", list(h.values()))
    for t in data.get("tasks", []):
        cur.execute("INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?)", list(t.values()))
    c.commit()
    c.close()

# --- 주요 로직 함수들 (생략 없이 유지) ---
def list_habits(user_id: str, active_only=True):
    c = conn()
    q = "SELECT * FROM habits WHERE user_id=?"
    if active_only: q += " AND active=1"
    df = pd.read_sql_query(q + " ORDER BY id DESC", c, params=(user_id,))
    c.close()
    return df

def add_plan_task(user_id: str, d: date, text: str):
    if not text.strip(): return
    c = conn()
    now = datetime.now().isoformat()
    c.execute("INSERT INTO tasks (user_id, task_date, text, source, created_at, updated_at) VALUES (?,?,?,?,?,?)",
              (user_id, d.isoformat(), text, 'plan', now, now))
    c.commit()
    c.close()
    save_snapshot(user_id)

def update_task_status(user_id: str, tid: int, status: str, reason: str = None):
    c = conn()
    now = datetime.now().isoformat()
    c.execute("UPDATE tasks SET status=?, fail_reason=?, updated_at=? WHERE id=? AND user_id=?",
              (status, reason, now, tid, user_id))
    c.commit()
    c.close()
    save_snapshot(user_id)

# --- 화면 구성 ---
def screen_planner(user_id):
    st.subheader("📅 데일리 플래너")
    
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = date.today()
    
    sel_date = st.session_state.selected_date
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        new_date = st.date_input("날짜 선택", sel_date)
        if new_date != sel_date:
            st.session_state.selected_date = new_date
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"### {sel_date.isoformat()}")
        
        with st.form("add_plan", clear_on_submit=True):
            t_input = st.text_input("새 계획")
            if st.form_submit_button("추가"):
                add_plan_task(user_id, sel_date, t_input)
                st.rerun()
        
        tasks = pd.read_sql_query("SELECT * FROM tasks WHERE user_id=? AND task_date=?", 
                                 conn(), params=(user_id, sel_date.isoformat()))
        
        for _, r in tasks.iterrows():
            with st.container():
                c_a, c_b, c_c = st.columns([3, 1, 1])
                c_a.write(f"**{r['text']}** ({r['status']})")
                if c_b.button("✅", key=f"done_{r['id']}"):
                    update_task_status(user_id, r['id'], 'success')
                    st.rerun()
                if c_c.button("❌", key=f"fail_{r['id']}"):
                    st.session_state[f"fail_mode_{r['id']}"] = True
                
                if st.session_state.get(f"fail_mode_{r['id']}"):
                    reason = st.text_input("실패 원인", key=f"reason_{r['id']}")
                    if st.button("저장", key=f"save_r_{r['id']}"):
                        update_task_status(user_id, r['id'], 'fail', reason)
                        st.session_state[f"fail_mode_{r['id']}"] = False
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- 메인 실행부 (가장 중요) ---
def main():
    inject_css()
    init_db()

    # 1. LocalStorage 비동기 값 로드 대기
    # st.session_state에 값이 들어올 때까지 '기기 확인 중' 단계 유지
    storage = ls()
    uid = storage.getItem("failog_uid", key="ls_uid")
    
    # 2. UID가 없다면 생성 및 저장 (처음 접속)
    if uid is None:
        st.write("기기 정보를 확인하고 있습니다...")
        # 아주 짧은 딜레이 후 재시도하게 함 (비동기 처리)
        if "init_retry" not in st.session_state:
            st.session_state.init_retry = True
            new_uid = str(uuid.uuid4())
            storage.setItem("failog_uid", new_uid)
        st.rerun()
        return

    user_id = uid
    
    # 3. 데이터 복구 시도 (UID 확정 직후)
    # 로컬 스토리지에 저장된 스냅샷을 가져옴
    snap = storage.getItem(f"failog_snap_{user_id}", key=f"ls_snap_{user_id}")
    restore_from_snapshot(user_id)

    # 4. OpenAI 설정 로드
    stored_key = storage.getItem("failog_oa_key", key="ls_oa_key")
    stored_model = storage.getItem("failog_oa_model", key="ls_oa_model")
    
    if "openai_api_key" not in st.session_state and stored_key:
        st.session_state.openai_api_key = stored_key
    if "openai_model" not in st.session_state and stored_model:
        st.session_state.openai_model = stored_model or "gpt-4o-mini"

    # --- 사이드바 및 네비게이션 ---
    st.title("🧊 FAILOG")
    menu = st.sidebar.radio("메뉴", ["플래너", "실패 리포트", "설정"])

    if menu == "플래너":
        screen_planner(user_id)
    elif menu == "실패 리포트":
        st.write("준비 중인 기능입니다.")
    elif menu == "설정":
        st.subheader("🔑 OpenAI 설정")
        key_input = st.text_input("API Key", value=st.session_state.get("openai_api_key", ""), type="password")
        model_input = st.text_input("Model", value=st.session_state.get("openai_model", "gpt-4o-mini"))
        save_local = st.checkbox("로컬 저장 (기기 기억)", value=bool(stored_key))
        
        if st.button("설정 적용"):
            st.session_state.openai_api_key = key_input
            st.session_state.openai_model = model_input
            if save_local:
                ls_set("failog_oa_key", key_input)
                ls_set("failog_oa_model", model_input)
            else:
                ls_set("failog_oa_key", "")
            st.success("설정이 적용되었습니다.")

    # 하단 유저 식별자 표시 (디버깅용)
    st.sidebar.markdown(f"---")
    st.sidebar.caption(f"Device ID: {user_id[:8]}...")

if __name__ == "__main__":
    main()
