import math
from datetime import datetime
from typing import Dict, List, Tuple

import requests
import streamlit as st

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="🎬 나와 어울리는 영화는?", page_icon="🎬", layout="wide")

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# 4차원 취향 벡터 축: R(로맨스/드라마), A(액션/어드벤처), S(SF/판타지), C(코미디)
AXES = ["R", "A", "S", "C"]

# TMDB 장르 ID -> 취향 축 매핑
TMDB_GENRE_TO_AXIS = {
    18: "R",      # Drama
    10749: "R",   # Romance
    28: "A",      # Action
    878: "S",     # Science Fiction
    14: "S",      # Fantasy
    35: "C",      # Comedy
}

AXIS_LABEL = {
    "R": "로맨스/드라마",
    "A": "액션/어드벤처",
    "S": "SF/판타지",
    "C": "코미디",
}

# =========================================================
# Questions (choice -> axis)
# =========================================================
questions = [
    {
        "q": "1) 오랜만에 시간이 비었어. 오늘 밤, 너는 어떤 무드로 영화를 보고 싶어?",
        "options": [
            ("잔잔하게 감정에 몰입하면서 여운을 느끼고 싶어", "R"),
            ("심장 뛰는 전개! 몰입감 있는 액션으로 스트레스 날리고 싶어", "A"),
            ("현실을 벗어나 새로운 세계관에 푹 빠지고 싶어", "S"),
            ("가볍게 웃고 기분 전환하고 싶어", "C"),
        ],
    },
    {
        "q": "2) 친구랑 영화 취향 얘기 중! 너를 가장 설레게 하는 요소는?",
        "options": [
            ("관계와 감정선, 그리고 공감되는 성장 이야기", "R"),
            ("박진감 넘치는 추격전/전투, 스케일 큰 장면", "A"),
            ("시간여행, 마법, 외계/미래 같은 상상력 폭발 설정", "S"),
            ("대사/상황이 빵빵 터지는 유머와 케미", "C"),
        ],
    },
    {
        "q": "3) 시험 끝난 날! 너는 어떤 방식으로 '해방감'을 즐겨?",
        "options": [
            ("조용히 감정 정리하면서 위로받는 이야기로 힐링", "R"),
            ("몸이 들썩! 시원한 한 방이 있는 통쾌함", "A"),
            ("현실 탈출! 완전히 다른 차원의 경험", "S"),
            ("웃음으로 다 털어내기! 아무 생각 없이 즐기기", "C"),
        ],
    },
    {
        "q": "4) 영화 속 주인공이 된다면, 너는 어떤 타입이야?",
        "options": [
            ("사람 마음을 움직이며 관계 속에서 성장하는 주인공", "R"),
            ("위기 속에서도 돌파하는 해결사/모험가", "A"),
            ("세계의 비밀을 풀거나 특별한 능력을 가진 인물", "S"),
            ("분위기 메이커! 사건을 웃음으로 바꾸는 인물", "C"),
        ],
    },
    {
        "q": "5) 너의 ‘인생 영화’ 후보에 가장 가까운 느낌은?",
        "options": [
            ("몇 년이 지나도 마음이 찡하고 생각나는 이야기", "R"),
            ("명장면이 뇌리에 박히는 레전드 스케일/전개", "A"),
            ("설정이 신선해서 계속 파고들고 싶어지는 작품", "S"),
            ("힘들 때마다 보면 기분 좋아지는 웃긴 작품", "C"),
        ],
    },
]


# =========================================================
# Helpers (math / text)
# =========================================================
def safe_text(x, fallback="정보 없음"):
    if x is None:
        return fallback
    s = str(x).strip()
    return s if s else fallback


def normalize_vec(v: Dict[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(v[k] ** 2 for k in AXES))
    if norm <= 1e-9:
        return {k: 0.0 for k in AXES}
    return {k: v[k] / norm for k in AXES}


def cosine(u: Dict[str, float], m: Dict[str, float]) -> float:
    return sum(u[k] * m[k] for k in AXES)


def dominant_axis(u: Dict[str, float]) -> str:
    return max(AXES, key=lambda k: u.get(k, 0.0))


def movie_axis_vec(genre_ids: List[int]) -> Dict[str, float]:
    v = {k: 0.0 for k in AXES}
    for gid in genre_ids or []:
        axis = TMDB_GENRE_TO_AXIS.get(gid)
        if axis:
            v[axis] += 1.0
    return normalize_vec(v)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def recency_bonus(release_date: str) -> float:
    """
    최근작 소폭 보너스 (0~1).
    - 0년차: 1.0
    - 10년 이상: 0.0
    """
    try:
        dt = datetime.strptime(release_date, "%Y-%m-%d")
        years = (datetime.now() - dt).days / 365.25
        return clamp01(1.0 - (years / 10.0))
    except Exception:
        return 0.3  # 날짜 없으면 약한 기본값


def build_user_vec_from_answers(answer_axes: List[str]) -> Dict[str, float]:
    v = {k: 0.0 for k in AXES}
    for a in answer_axes:
        v[a] += 1.0
    return normalize_vec(v)


def genre_ids_for_candidate_pool(u: Dict[str, float]) -> List[int]:
    """
    사용자 벡터 상위 2~3축 기반으로 후보 풀 확장 (OR 형태로 수집)
    """
    ranked = sorted(AXES, key=lambda k: u[k], reverse=True)
    top = ranked[:3]  # 상위 3축까지 반영
    ids = []
    for ax in top:
        if ax == "R":
            ids += [18, 10749]
        elif ax == "A":
            ids += [28]
        elif ax == "S":
            ids += [878, 14]
        elif ax == "C":
            ids += [35]
    # 중복 제거
    return sorted(list(set(ids)))


# =========================================================
# TMDB fetch
# =========================================================
def tmdb_discover_pool(api_key: str, with_genres_or: List[int], min_rating: float, pages: int = 4) -> List[dict]:
    """
    discover/movie로 후보 풀 수집.
    with_genres는 OR로 넓게(파이프 |) 요청.
    """
    url = "https://api.themoviedb.org/3/discover/movie"
    all_results = []
    with_genres_value = "|".join(map(str, with_genres_or))

    for page in range(1, pages + 1):
        params = {
            "api_key": api_key,
            "language": "ko-KR",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "page": page,
            "with_genres": with_genres_value,
            "vote_average.gte": float(min_rating),
            "vote_count.gte": 50,  # 신뢰도(필요 없으면 사이드바로 빼도 됨)
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        all_results.extend(data.get("results", []))

    # 중복 제거(id 기준)
    uniq = {}
    for m in all_results:
        mid = m.get("id")
        if mid is not None and mid not in uniq:
            uniq[mid] = m
    return list(uniq.values())


# =========================================================
# Scoring + Recommender (1) Scoring  + (2) Bandit Explore/Exploit
# =========================================================
def score_movies(
    u: Dict[str, float],
    movies: List[dict],
    w_fit=0.50,
    w_rating=0.20,
    w_votes=0.15,
    w_pop=0.10,
    w_recent=0.05,
) -> List[Tuple[dict, float, float]]:
    """
    return list of (movie, total_score, fit_score)
    fit_score: cosine(u, movie_vec) in [0..1] (clamped)
    """
    if not movies:
        return []

    pops = [float(m.get("popularity", 0.0) or 0.0) for m in movies]
    pop_min, pop_max = min(pops), max(pops)
    pop_den = (pop_max - pop_min) if (pop_max - pop_min) > 1e-9 else 1.0

    # vote_count log scaling 기준
    vote_counts = [int(m.get("vote_count", 0) or 0) for m in movies]
    vc_max = max(vote_counts) if vote_counts else 1
    vc_max = max(vc_max, 1)

    scored = []
    for m in movies:
        mv = movie_axis_vec(m.get("genre_ids", []))

        fit = clamp01((cosine(u, mv) + 1) / 2)  # cosine은 0~1로 나오지만 안전 처리
        # 실제론 0~1 범위라 (cos+1)/2 하면 0.5~1이 될 수 있으므로 아래처럼 정리:
        # -> cosine이 음수 나올 일이 거의 없지만, 축이 0이면 0이라서 0.5가 되어버림.
        # 그래서 cosine 그대로를 0~1 clamp로 사용:
        fit = clamp01(cosine(u, mv))

        rating = float(m.get("vote_average", 0.0) or 0.0) / 10.0
        rating = clamp01(rating)

        vc = int(m.get("vote_count", 0) or 0)
        votes = math.log1p(vc) / math.log1p(vc_max)  # 0~1
        votes = clamp01(votes)

        pop = float(m.get("popularity", 0.0) or 0.0)
        pop_n = (pop - pop_min) / pop_den
        pop_n = clamp01(pop_n)

        recent = recency_bonus(m.get("release_date", ""))

        total = (
            w_fit * fit +
            w_rating * rating +
            w_votes * votes +
            w_pop * pop_n +
            w_recent * recent
        )

        scored.append((m, float(total), float(fit)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def pick_exploit_explore(
    scored: List[Tuple[dict, float, float]],
    u: Dict[str, float],
    k_total: int = 5,
    k_exploit: int = 3,
    epsilon: float = 0.35,
    exclude_ids: set | None = None,
) -> List[Tuple[dict, float, float]]:
    """
    Bandit 스타일: exploit(상위) + explore(의외성/다양성) 섞기
    - exploit: 상위 점수에서 고정 선택
    - explore: 남은 후보 중에서 (1 - fit) 가 큰 것 중 score도 일정 이상인 것을 선택
    - epsilon: explore 선택을 더 랜덤하게 섞는 정도
    """
    exclude_ids = exclude_ids or set()

    # 이미 본 영화 제외
    filtered = [x for x in scored if (x[0].get("id") not in exclude_ids)]
    if not filtered:
        return []

    k_exploit = min(k_exploit, k_total)
    exploit = filtered[:k_exploit]

    remaining = filtered[k_exploit:]
    if not remaining:
        return exploit[:k_total]

    # explore 후보: "너무 동떨어진 것"은 피하려고 score 하한을 둠
    # (상위 점수의 70% 이상인 것들 중에서 의외성 높은 것)
    top_score = exploit[0][1] if exploit else filtered[0][1]
    threshold = top_score * 0.70
    explore_pool = [x for x in remaining if x[1] >= threshold]
    if len(explore_pool) < (k_total - k_exploit):
        explore_pool = remaining  # 부족하면 그냥 남은 것에서

    # 의외성 = (1 - fit) * 0.7 + (score) * 0.3 로 가중
    candidates = []
    for m, total, fit in explore_pool:
        surprise = (1.0 - clamp01(fit))
        utility = 0.7 * surprise + 0.3 * clamp01(total)
        candidates.append((m, total, fit, utility))

    # utility 상위부터 선택하되, epsilon으로 약간 섞기(가끔 랜덤)
    candidates.sort(key=lambda x: x[3], reverse=True)

    explore = []
    need = k_total - k_exploit
    for _ in range(need):
        if not candidates:
            break
        if len(candidates) == 1:
            pick = candidates.pop(0)
            explore.append(pick[:3])
            continue

        # epsilon 확률로 상위 몇 개 중 랜덤
        import random
        if random.random() < epsilon:
            window = min(6, len(candidates))
            idx = random.randrange(0, window)
            pick = candidates.pop(idx)
        else:
            pick = candidates.pop(0)
        explore.append(pick[:3])

    return (exploit + explore)[:k_total]


def update_user_vector_bandit(
    u: Dict[str, float],
    feedback: Dict[int, int],
    movie_vecs: Dict[int, Dict[str, float]],
    alpha: float = 0.35,
) -> Dict[str, float]:
    """
    feedback: {movie_id: +1(like), -1(dislike), 0(neutral)}
    u_new = normalize(u + alpha * sum(feedback_i * mv_i))
    """
    delta = {k: 0.0 for k in AXES}
    for mid, fb in feedback.items():
        if fb == 0:
            continue
        mv = movie_vecs.get(mid)
        if not mv:
            continue
        for k in AXES:
            delta[k] += fb * mv[k]

    u2 = {k: u[k] + alpha * delta[k] for k in AXES}
    return normalize_vec(u2)


# =========================================================
# Session state init
# =========================================================
if "stage" not in st.session_state:
    st.session_state.stage = "quiz"  # quiz -> results
if "user_vec" not in st.session_state:
    st.session_state.user_vec = None
if "base_user_vec" not in st.session_state:
    st.session_state.base_user_vec = None
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []  # list of (movie, total, fit)


# =========================================================
# UI: Header
# =========================================================
st.title("🎬 나와 어울리는 영화는?")
st.write("5개의 질문에 답하면, **취향 벡터 기반 점수화 추천(Scoring)** + **탐색/활용 밴딧(Explore/Exploit)** 로직으로 영화를 추천해드려요! 🍿")

# =========================================================
# Sidebar settings
# =========================================================
st.sidebar.header("🔧 추천 설정")
TMDB_API_KEY = st.sidebar.text_input("TMDB API Key", type="password")

min_rating = st.sidebar.slider("최소 평점 필터", 0.0, 10.0, 6.5, 0.5)
epsilon = st.sidebar.slider("탐색(Explore) 랜덤성", 0.0, 1.0, 0.35, 0.05,
                            help="높을수록 '의외의 추천'이 더 랜덤하게 섞여요.")
alpha = st.sidebar.slider("피드백 반영 강도(학습률)", 0.0, 1.0, 0.35, 0.05,
                          help="좋아요/별로 피드백이 다음 추천에 얼마나 강하게 반영될지 조절해요.")

st.sidebar.divider()
if st.sidebar.button("🔁 테스트 다시하기(초기화)"):
    st.session_state.stage = "quiz"
    st.session_state.user_vec = None
    st.session_state.base_user_vec = None
    st.session_state.seen_ids = set()
    st.session_state.last_recs = []
    # 라디오 상태 초기화를 위해 키를 바꿔주거나 rerun
    st.rerun()

# =========================================================
# Quiz UI (only when stage == quiz)
# =========================================================
if st.session_state.stage == "quiz":
    st.subheader("📝 질문에 답해 주세요")

    answer_axes = []
    for idx, item in enumerate(questions, start=1):
        option_texts = [t for (t, _ax) in item["options"]]
        choice = st.radio(item["q"], option_texts, index=None, key=f"q_{idx}")

        if choice is None:
            answer_axes.append(None)
        else:
            ax = next(ax for (t, ax) in item["options"] if t == choice)
            answer_axes.append(ax)

    st.divider()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        clicked = st.button("결과 보기", type="primary")
    with col_b:
        st.caption("※ 사이드바에 TMDB API Key를 입력해야 영화 추천을 불러올 수 있어요.")

    if clicked:
        if any(x is None for x in answer_axes):
            st.warning("모든 질문에 답해 주세요! 🙂")
            st.stop()
        if not TMDB_API_KEY:
            st.info("사이드바에 TMDB API Key를 입력해주세요.")
            st.stop()

        u0 = build_user_vec_from_answers(answer_axes)
        st.session_state.base_user_vec = u0
        st.session_state.user_vec = u0
        st.session_state.stage = "results"
        st.session_state.seen_ids = set()
        st.session_state.last_recs = []
        st.rerun()

# =========================================================
# Results UI (stage == results)
# =========================================================
if st.session_state.stage == "results":
    u = st.session_state.user_vec
    if not u:
        st.error("내부 상태 오류: user_vec이 비어 있어요. 사이드바에서 초기화를 눌러 다시 시작해 주세요.")
        st.stop()

    if not TMDB_API_KEY:
        st.info("사이드바에 TMDB API Key를 입력해주세요.")
        st.stop()

    # 결과 타이틀(현재 취향 벡터 기준)
    dom = dominant_axis(u)
    st.markdown(f"## ✨ 당신에게 딱인 장르는: **{AXIS_LABEL[dom]}**!")

    # 취향 벡터 표시(간단)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("로/드", f"{u['R']:.2f}")
    c2.metric("액/어", f"{u['A']:.2f}")
    c3.metric("SF/판", f"{u['S']:.2f}")
    c4.metric("코미디", f"{u['C']:.2f}")

    st.caption(f"🔎 필터: 최소 평점 **{min_rating:.1f}** 이상 · 탐색 랜덤성 **{epsilon:.2f}** · 학습률 **{alpha:.2f}**")

    # -----------------------------------------------------
    # Fetch + Recommend
    # -----------------------------------------------------
    def make_recommendations() -> List[Tuple[dict, float, float]]:
        pool_genres = genre_ids_for_candidate_pool(u)

        with st.spinner("TMDB에서 후보 영화를 모으고, 취향 점수로 랭킹을 계산하는 중..."):
            pool = tmdb_discover_pool(
                api_key=TMDB_API_KEY,
                with_genres_or=pool_genres,
                min_rating=min_rating,
                pages=4,  # 후보 풀 넉넉히
            )

            if not pool:
                return []

            scored = score_movies(u, pool)

            recs = pick_exploit_explore(
                scored=scored,
                u=u,
                k_total=5,
                k_exploit=3,
                epsilon=epsilon,
                exclude_ids=st.session_state.seen_ids,
            )
            return recs

    # 처음 들어왔거나, last_recs가 비어있으면 생성
    if not st.session_state.last_recs:
        try:
            st.session_state.last_recs = make_recommendations()
        except requests.RequestException as e:
            st.error(f"TMDB 호출 중 오류가 발생했어요: {e}")
            st.stop()

    recs = st.session_state.last_recs
    if not recs:
        st.warning("조건에 맞는 추천이 부족해요 😢  \n평점 필터를 낮추거나(예: 5.5), 다시 시도해 보세요!")
        st.stop()

    st.markdown("### 🍿 추천 영화 5편 (3개는 취향 적합도 상위, 2개는 새로운 취향 탐색용)")

    # -----------------------------------------------------
    # Show cards + feedback
    # -----------------------------------------------------
    movie_vec_cache = {}
    feedback_choices = {}  # movie_id -> (-1/0/+1)

    cols = st.columns(3)
    for i, (m, total, fit) in enumerate(recs):
        mid = m.get("id")
        title = safe_text(m.get("title"))
        vote = m.get("vote_average")
        overview = safe_text(m.get("overview"), fallback="줄거리 정보가 없어요.")
        poster_path = m.get("poster_path")
        release_date = safe_text(m.get("release_date"), "개봉일 정보 없음")
        genre_ids = m.get("genre_ids", []) or []

        mv = movie_axis_vec(genre_ids)
        movie_vec_cache[mid] = mv

        # 간단 추천 이유 생성 (점수 기반)
        fit_pct = int(round(fit * 100))
        rating = float(vote or 0.0)
        why = f"취향 적합도 **{fit_pct}%** · 평점 **{rating:.1f}/10**"
        if dom == "R":
            why += " · 감정선/여운 포인트가 맞을 확률↑"
        elif dom == "A":
            why += " · 시원한 전개/액션 텐션 기대"
        elif dom == "S":
            why += " · 세계관/설정 몰입감 기대"
        else:
            why += " · 가볍게 웃기 좋은 텐션"

        with cols[i % 3]:
            with st.container(border=True):
                # Poster
                if poster_path:
                    st.image(f"{POSTER_BASE_URL}{poster_path}", use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)

                # Title + score
                st.markdown(f"**{title}**")
                if vote is not None:
                    st.caption(f"⭐ 평점: {float(vote):.1f}/10 · 🎯 추천 점수: {total:.3f}")
                else:
                    st.caption(f"⭐ 평점: 정보 없음 · 🎯 추천 점수: {total:.3f}")

                st.write(f"💡 **이 영화를 추천하는 이유:** {why}")

                # Feedback (Bandit)
                fb = st.radio(
                    "피드백",
                    ["👍 보고 싶다", "😐 보통", "👎 별로"],
                    horizontal=True,
                    key=f"fb_{mid}",
                )
                feedback_choices[mid] = +1 if fb.startswith("👍") else (-1 if fb.startswith("👎") else 0)

                with st.expander("자세히 보기"):
                    st.write(overview)
                    st.caption(f"📅 개봉일: {release_date}")
                    pop = m.get("popularity")
                    vc = m.get("vote_count")
                    if pop is not None:
                        st.caption(f"🔥 인기도: {float(pop):.0f}")
                    if vc is not None:
                        st.caption(f"🗳️ 투표수: {int(vc)}")

    st.divider()

    # -----------------------------------------------------
    # Apply feedback & refresh
    # -----------------------------------------------------
    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])

    with c1:
        apply_btn = st.button("🧠 피드백 반영하고 다시 추천", type="primary")
    with c2:
        refresh_btn = st.button("🎲 같은 취향으로 새 추천", help="피드백 반영 없이, 탐색/활용만 다시 섞어 추천해요.")
    with c3:
        st.caption("팁: 👍/👎 피드백을 반영하면 다음 추천이 당신 취향으로 점점 조정돼요(밴딧 업데이트).")

    if refresh_btn:
        # seen_ids에 현재 추천을 추가해서 겹침 감소
        for (m, _, _) in recs:
            mid = m.get("id")
            if mid is not None:
                st.session_state.seen_ids.add(mid)

        try:
            st.session_state.last_recs = make_recommendations()
        except requests.RequestException as e:
            st.error(f"TMDB 호출 중 오류가 발생했어요: {e}")
            st.stop()
        st.rerun()

    if apply_btn:
        # seen_ids에 현재 추천 추가
        for (m, _, _) in recs:
            mid = m.get("id")
            if mid is not None:
                st.session_state.seen_ids.add(mid)

        # 밴딧 업데이트
        u_new = update_user_vector_bandit(
            u=st.session_state.user_vec,
            feedback=feedback_choices,
            movie_vecs=movie_vec_cache,
            alpha=alpha,
        )
        st.session_state.user_vec = u_new

        # 새 추천 생성
        try:
            st.session_state.last_recs = make_recommendations()
        except requests.RequestException as e:
            st.error(f"TMDB 호출 중 오류가 발생했어요: {e}")
            st.stop()
        st.rerun()
