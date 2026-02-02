import streamlit as st
import requests

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="나와 어울리는 영화는?", page_icon="🎬", layout="wide")

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

GENRE_MAP = {
    "로맨스/드라마": {"label": "로맨스/드라마", "tmdb_ids": [10749, 18]},
    "액션/어드벤처": {"label": "액션/어드벤처", "tmdb_ids": [28]},  # 어드벤처(12)는 요구사항에 없어서 제외
    "SF/판타지": {"label": "SF/판타지", "tmdb_ids": [878, 14]},
    "코미디": {"label": "코미디", "tmdb_ids": [35]},
}

questions = [
    {
        "q": "1) 오랜만에 시간이 비었어. 오늘 밤, 너는 어떤 무드로 영화를 보고 싶어?",
        "options": [
            ("잔잔하게 감정에 몰입하면서 여운을 느끼고 싶어", "로맨스/드라마"),
            ("심장 뛰는 전개! 몰입감 있는 액션으로 스트레스 날리고 싶어", "액션/어드벤처"),
            ("현실을 벗어나 새로운 세계관에 푹 빠지고 싶어", "SF/판타지"),
            ("가볍게 웃고 기분 전환하고 싶어", "코미디"),
        ],
    },
    {
        "q": "2) 친구랑 영화 취향 얘기 중! 너를 가장 설레게 하는 요소는?",
        "options": [
            ("관계와 감정선, 그리고 공감되는 성장 이야기", "로맨스/드라마"),
            ("박진감 넘치는 추격전/전투, 스케일 큰 장면", "액션/어드벤처"),
            ("시간여행, 마법, 외계/미래 같은 상상력 폭발 설정", "SF/판타지"),
            ("대사/상황이 빵빵 터지는 유머와 케미", "코미디"),
        ],
    },
    {
        "q": "3) 시험 끝난 날! 너는 어떤 방식으로 '해방감'을 즐겨?",
        "options": [
            ("조용히 감정 정리하면서 위로받는 이야기로 힐링", "로맨스/드라마"),
            ("몸이 들썩! 시원한 한 방이 있는 통쾌함", "액션/어드벤처"),
            ("현실 탈출! 완전히 다른 차원의 경험", "SF/판타지"),
            ("웃음으로 다 털어내기! 아무 생각 없이 즐기기", "코미디"),
        ],
    },
    {
        "q": "4) 영화 속 주인공이 된다면, 너는 어떤 타입이야?",
        "options": [
            ("사람 마음을 움직이며 관계 속에서 성장하는 주인공", "로맨스/드라마"),
            ("위기 속에서도 돌파하는 해결사/모험가", "액션/어드벤처"),
            ("세계의 비밀을 풀거나 특별한 능력을 가진 인물", "SF/판타지"),
            ("분위기 메이커! 사건을 웃음으로 바꾸는 인물", "코미디"),
        ],
    },
    {
        "q": "5) 너의 ‘인생 영화’ 후보에 가장 가까운 느낌은?",
        "options": [
            ("몇 년이 지나도 마음이 찡하고 생각나는 이야기", "로맨스/드라마"),
            ("명장면이 뇌리에 박히는 레전드 스케일/전개", "액션/어드벤처"),
            ("설정이 신선해서 계속 파고들고 싶어지는 작품", "SF/판타지"),
            ("힘들 때마다 보면 기분 좋아지는 웃긴 작품", "코미디"),
        ],
    },
]

# -----------------------------
# 헬퍼 함수
# -----------------------------
def analyze_answers(selected_labels):
    """최다 선택 장르 반환(동점이면 우선순위 적용)"""
    scores = {k: 0 for k in GENRE_MAP.keys()}
    for label in selected_labels:
        scores[label] += 1

    priority = ["로맨스/드라마", "액션/어드벤처", "SF/판타지", "코미디"]
    max_score = max(scores.values())
    top = [k for k, v in scores.items() if v == max_score]
    for p in priority:
        if p in top:
            return p, scores
    return top[0], scores


def discover_movies_by_genre(api_key, genre_ids, min_rating=0.0, limit=5):
    """
    TMDB discover로 인기 영화 가져오기 + 평점 필터 적용
    - vote_average.gte = min_rating
    - 결과가 5개 미만이면 페이지 넘겨가며 채움(최대 몇 페이지까지만)
    """
    base_url = "https://api.themoviedb.org/3/discover/movie"
    movies = []
    page = 1
    max_pages = 5  # 너무 오래 돌지 않도록 제한

    # 1차: 장르 ids 콤마(AND 조건)로 시도
    def fetch(with_genres_value, page_no):
        params = {
            "api_key": api_key,
            "language": "ko-KR",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "page": page_no,
            "with_genres": with_genres_value,
            "vote_average.gte": float(min_rating),
            "vote_count.gte": 50,  # 평점 신뢰도(너무 소수표본 방지) - 필요 없으면 제거 가능
        }
        r = requests.get(base_url, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])

    with_genres = ",".join(map(str, genre_ids))
    while len(movies) < limit and page <= max_pages:
        movies.extend(fetch(with_genres, page))
        page += 1

    # 결과가 부족하고 장르가 2개 이상이면(AND가 빡셈) 첫 장르만으로 fallback
    if len(movies) < limit and len(genre_ids) > 1:
        movies = []
        page = 1
        with_genres = str(genre_ids[0])
        while len(movies) < limit and page <= max_pages:
            movies.extend(fetch(with_genres, page))
            page += 1

    return movies[:limit]


def reason_text(genre_label, scores):
    if genre_label == "로맨스/드라마":
        return f"감정선과 공감 포인트를 고른 선택이 많았어요(선택 {scores[genre_label]}개). 잔잔하게 몰입하는 이야기가 잘 맞아요."
    if genre_label == "액션/어드벤처":
        return f"통쾌한 전개와 긴장감을 선호하는 선택이 많았어요(선택 {scores[genre_label]}개). 시원한 액션으로 스트레스 해소 추천!"
    if genre_label == "SF/판타지":
        return f"상상력/세계관을 중시하는 선택이 많았어요(선택 {scores[genre_label]}개). 현실 탈출용으로 딱이에요."
    if genre_label == "코미디":
        return f"웃음과 가벼운 텐션을 고른 선택이 많았어요(선택 {scores[genre_label]}개). 기분 전환엔 코미디가 최고!"
    return "당신의 선택 패턴을 기반으로 추천했어요!"


def safe_text(x, fallback="정보 없음"):
    if x is None:
        return fallback
    x = str(x).strip()
    return x if x else fallback


def simple_movie_reason(best_genre):
    if best_genre == "로맨스/드라마":
        return "감정선과 여운이 기대되는 작품이에요."
    if best_genre == "액션/어드벤처":
        return "몰입감 있는 전개로 스트레스 해소에 좋아요."
    if best_genre == "SF/판타지":
        return "세계관/설정이 매력적인 작품일 확률이 높아요."
    return "가볍게 즐기며 기분 전환하기 좋아요."


# -----------------------------
# UI: 헤더/소개
# -----------------------------
st.title("🎬 나와 어울리는 영화는?")
st.write("5개의 질문에 답하면, 당신과 가장 잘 맞는 영화 장르와 인기 영화를 추천해드려요! 🍿")

# -----------------------------
# 사이드바: TMDB + 평점 필터
# -----------------------------
st.sidebar.header("🔧 추천 설정")
TMDB_API_KEY = st.sidebar.text_input("TMDB API Key", type="password")
min_rating = st.sidebar.slider("최소 평점 필터", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
st.sidebar.caption("최소 평점 이상인 영화만 추천해요. (TMDB vote_average 기준)")

# -----------------------------
# 질문 UI
# -----------------------------
st.subheader("📝 질문에 답해 주세요")

selected_labels = []
for idx, item in enumerate(questions, start=1):
    option_texts = [t for (t, _label) in item["options"]]
    choice = st.radio(item["q"], option_texts, index=None, key=f"q_{idx}")

    if choice is None:
        selected_labels.append(None)
    else:
        label = next(lbl for (t, lbl) in item["options"] if t == choice)
        selected_labels.append(label)

st.divider()

# -----------------------------
# 결과 보기 버튼
# -----------------------------
col_a, col_b = st.columns([1, 3])
with col_a:
    clicked = st.button("결과 보기", type="primary")
with col_b:
    st.caption("※ 모든 질문에 답하고, 사이드바에 TMDB API Key를 입력하면 추천 영화 5개를 보여줘요.")

if clicked:
    # 1) 답변 체크
    if any(x is None for x in selected_labels):
        st.warning("모든 질문에 답해 주세요! 🙂")
        st.stop()

    # 2) 장르 결정
    best_genre, scores = analyze_answers(selected_labels)
    genre_info = GENRE_MAP[best_genre]

    # 3) API 키 체크
    if not TMDB_API_KEY:
        st.info("사이드바에 TMDB API Key를 입력해주세요.")
        st.stop()

    # 4) TMDB에서 해당 장르 인기 영화 가져오기 (로딩 스피너)
    with st.spinner("TMDB에서 인기 영화를 불러오는 중..."):
        try:
            movies = discover_movies_by_genre(
                api_key=TMDB_API_KEY,
                genre_ids=genre_info["tmdb_ids"],
                min_rating=min_rating,
                limit=5,
            )
        except requests.RequestException as e:
            st.error(f"TMDB 호출 중 오류가 발생했어요: {e}")
            st.stop()

    # 5) 결과 화면
    st.markdown(f"## ✨ 당신에게 딱인 장르는: **{genre_info['label']}**!")
    st.write("🎯 **이 장르를 추천하는 이유**")
    st.success(reason_text(best_genre, scores))

    st.caption(f"🔎 현재 필터: 최소 평점 **{min_rating:.1f}** 이상")

    if not movies:
        st.warning("조건에 맞는 영화가 부족해요 😢  \n평점 필터를 낮추거나(예: 5.5), 다시 시도해 보세요!")
        st.stop()

    st.markdown("### 🍿 지금 뜨는 인기 영화 5편")

    # 3열 카드 형태로 배치
    cols = st.columns(3)
    for i, m in enumerate(movies):
        with cols[i % 3]:
            title = safe_text(m.get("title"))
            vote = m.get("vote_average")
            overview = safe_text(m.get("overview"), fallback="줄거리 정보가 없어요.")
            poster_path = m.get("poster_path")
            release_date = safe_text(m.get("release_date"), "개봉일 정보 없음")

            with st.container(border=True):
                # 포스터
                if poster_path:
                    st.image(f"{POSTER_BASE_URL}{poster_path}", use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)

                # 제목/평점
                st.markdown(f"**{title}**")
                if vote is not None:
                    st.caption(f"⭐ 평점: {float(vote):.1f}/10")
                else:
                    st.caption("⭐ 평점: 정보 없음")

                # 추천 이유(간단)
                st.write(f"💡 **이 영화를 추천하는 이유:** {simple_movie_reason(best_genre)}")

                # 상세 정보 (expander)
                with st.expander("자세히 보기"):
                    st.write(overview)
                    st.caption(f"📅 개봉일: {release_date}")
                    popularity = m.get("popularity")
                    if popularity is not None:
                        st.caption(f"🔥 인기도: {float(popularity):.0f}")
