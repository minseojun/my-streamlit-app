import random
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="나의 AI 챗봇", page_icon="🤖", layout="centered")

st.title("🤖 나의 AI 챗봇")

# =========================
# 90년대 랜덤 노래 UI 데이터
# =========================
# 원하시면 더 많이/다른 장르로 확장해드릴게요.
NINETIES_SONGS = [
    # K-POP / Korean
    {"title": "To Heaven", "artist": "조성모", "year": 1998},
    {"title": "아로하", "artist": "쿨", "year": 1996},
    {"title": "애상", "artist": "쿨", "year": 1998},
    {"title": "너에게", "artist": "서태지와 아이들", "year": 1994},
    {"title": "난 알아요", "artist": "서태지와 아이들", "year": 1992},
    {"title": "바람", "artist": "윤도현", "year": 1997},
    {"title": "이별택시", "artist": "김연우(원곡)", "year": 1998},
    {"title": "커플", "artist": "젝스키스", "year": 1998},
    {"title": "Candy", "artist": "H.O.T.", "year": 1996},
    {"title": "해변의 여인", "artist": "쿨", "year": 1997},
    {"title": "하늘만 허락한 사랑", "artist": "엄정화", "year": 1999},
    {"title": "슬픈 언약식", "artist": "김정민", "year": 1994},
    {"title": "그대에게", "artist": "무한궤도", "year": 1990},
    {"title": "보고싶다", "artist": "김범수", "year": 1999},
    {"title": "서시", "artist": "신성우", "year": 1993},

    # Pop / International
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "year": 1991},
    {"title": "Wonderwall", "artist": "Oasis", "year": 1995},
    {"title": "My Heart Will Go On", "artist": "Celine Dion", "year": 1997},
    {"title": "Baby One More Time", "artist": "Britney Spears", "year": 1998},
    {"title": "I Want It That Way", "artist": "Backstreet Boys", "year": 1999},
    {"title": "Torn", "artist": "Natalie Imbruglia", "year": 1997},
    {"title": "Losing My Religion", "artist": "R.E.M.", "year": 1991},
    {"title": "No Scrubs", "artist": "TLC", "year": 1999},
    {"title": "Wannabe", "artist": "Spice Girls", "year": 1996},
    {"title": "Creep", "artist": "Radiohead", "year": 1992},
]

def pick_random_90s_song():
    return random.choice(NINETIES_SONGS)

# =========================
# Sidebar: API Key 입력
# =========================
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# =========================
# 90년대 노래 랜덤 UI (상단)
# =========================
st.subheader("🎵 오늘의 랜덤 90년대 노래")

# 세션 상태 초기화
if "song" not in st.session_state:
    st.session_state.song = pick_random_90s_song()

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔀 랜덤으로 뽑기", use_container_width=True):
        st.session_state.song = pick_random_90s_song()

with col2:
    if st.button("🧹 초기화(기본곡)", use_container_width=True):
        st.session_state.song = pick_random_90s_song()

song = st.session_state.song
st.markdown(
    f"""
**곡명:** {song['title']}  
**아티스트:** {song['artist']}  
**연도:** {song['year']}
"""
)

st.divider()

# =========================
# 챗봇 대화 UI
# =========================
# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요"):
    if not api_key:
        st.error("⚠️ 사이드바에서 API Key를 입력해주세요!")
    else:
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            client = OpenAI(api_key=api_key)

            # (선택) 현재 랜덤 노래 정보를 시스템/컨텍스트로 살짝 넣고 싶다면 아래처럼 추가 가능
            # 지금은 "UI 추가"만 원하신 것 같아 기본값으로는 그대로 두었습니다.
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
            )

            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
