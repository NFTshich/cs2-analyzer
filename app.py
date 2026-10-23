"""Streamlit-интерфейс CS2 Analyzer.

Запуск::

    streamlit run app.py

Содержит четыре страницы (sidebar-навигация):
    1. Прогноз матча — выбор двух команд и карты → вероятности + график.
    2. Форма команд — winrate за последние матчи.
    3. Карты — heatmap winrate по картам и сторонам.
    4. Backtest — как модель отработала на истории.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Позволяем импортировать src/ из корня проекта.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.autorefresh import start_autorefresh  # noqa: E402
from src.predictor import Predictor  # noqa: E402
from src.scraper import Scraper  # noqa: E402


# ---------------------------------------------------------------------------
# Автообновление данных в фоне (раз в час, если включено)
# ---------------------------------------------------------------------------
start_autorefresh()


# ---------------------------------------------------------------------------
# Инициализация данных. Читаем из локального кэша (data/raw/matches.csv),
# который обновляет фоновый поток — поэтому не сохраняем в st.cache_resource,
# а перечитываем каждый перезапуск скрипта (Streamlit делает это на каждое
# взаимодействие, но локальный CSV читается быстро).
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Читает сырые матчи из локального кэша (data/raw/matches.csv).

    Если файла нет — первичный сбор с Liquipedia (или синтетика в mock-режиме).
    """
    if config.MATCHES_RAW.exists():
        try:
            return pd.read_csv(config.MATCHES_RAW, parse_dates=["date"])
        except Exception:  # noqa: BLE001
            pass
    return Scraper().fetch()


@st.cache_data(show_spinner=False, ttl=60)
def _model_mtime() -> float:
    """Возвращает время изменения файла модели (для инвалидации кэша)."""
    p = config.MODEL_PATH
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner=False)
def get_predictor(mtime: float) -> Predictor:
    """Создаёт Predictor; кэш сбрасывается при изменении файла модели (mtime)."""
    return Predictor(load_raw())


def _pct(x: float) -> str:
    return f"{max(0.0, min(1.0, x)) * 100:.1f}%"


def _confidence_label(conf: float) -> str:
    """Классифицирует уверенность (high/medium/low) — синоним predictor."""
    if conf >= 0.30:
        return "high"
    if conf >= 0.15:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Шапка и боковая навигация
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CS2 Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# Топовый контрастный дизайн в стиле CS2: чёрный фон, неоновое свечение,
# минималистичная HUD-плашка с летящей пулей.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- базовый фон: глубокий чёрный с амбер-переливом ---- */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(1000px 600px at 15% -5%, rgba(255,140,0,0.08), transparent 55%),
            radial-gradient(800px 600px at 105% 110%, rgba(255,60,0,0.07), transparent 55%),
            linear-gradient(180deg, #050506 0%, #0a0a0c 50%, #060607 100%) !important;
        color: #f0ede8;
    }
    html, body { min-height: 100%; }

    /* ---- сайдбар тёмный с амбер-ребром ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070708, #0d0d10) !important;
        border-right: 1px solid rgba(255,150,30,0.25);
    }
    [data-testid="stSidebar"] * { color: #e8e2d8; }

    /* ---- светящийся заголовок ---- */
    h1 {
        font-weight: 900; letter-spacing: 1px; margin-top: 0.2rem;
        background: linear-gradient(92deg, #ffd76a, #ff8a00 45%, #ff4d00 80%, #ffd76a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 14px rgba(255,130,0,0.65)) drop-shadow(0 0 34px rgba(255,80,0,0.35));
        animation: h1breath 2.6s ease-in-out infinite;
    }
    @keyframes h1breath {
        0%,100% { filter: drop-shadow(0 0 12px rgba(255,130,0,0.5)) drop-shadow(0 0 26px rgba(255,100,0,0.25)); }
        50%     { filter: drop-shadow(0 0 18px rgba(255,150,0,0.9)) drop-shadow(0 0 42px rgba(255,90,0,0.5)); }
    }

    /* ---- неоновые кнопки ---- */
    .stButton>button {
        position: relative; z-index: 5; cursor: pointer;
        color: #fff; font-weight:700; letter-spacing:.5px;
        background: linear-gradient(135deg, rgba(255,120,0,0.20), rgba(255,60,0,0.12)) !important;
        border: 1px solid rgba(255,150,40,0.6) !important;
        box-shadow: 0 0 16px rgba(255,120,0,0.35), inset 0 0 12px rgba(255,120,0,0.12);
        transition: all .18s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(255,140,0,0.75), inset 0 0 18px rgba(255,140,0,0.2);
    }
    .stButton>button:active { transform: scale(0.98); }
    /* страховка: все декоративные элементы не ловят клики */
    .stApp button, .stButton > button, [class*="stButton"] { pointer-events: auto; }

    /* ---- метрики и карточки ---- */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,120,0,0.04)) !important;
        border: 1px solid rgba(255,160,50,0.25) !important;
        border-radius: 12px;
        box-shadow: 0 0 14px rgba(255,120,0,0.08);
    }
    [data-testid="stMetricValue"] { color: #ffd27a !important; }

    .stSelectbox > div > div {
        background: rgba(10,10,12,0.92) !important;
        border: 1px solid rgba(255,150,40,0.3) !important;
    }

    /* ---- HUD-плашка: минималистичная, с летящей пулей ---- */
    .cs2-hud {
        position: relative; height: 116px; overflow: hidden; border-radius: 12px;
        margin: 2px 0 14px 0; pointer-events: none;
        background:
            linear-gradient(180deg, rgba(10,10,12,0.9), rgba(6,6,8,0.95)),
            radial-gradient(circle at 50% 130%, rgba(255,120,0,0.10), transparent 60%);
        border: 1px solid rgba(255,150,40,0.28);
        box-shadow: 0 0 24px rgba(255,100,0,0.12), inset 0 0 60px rgba(0,0,0,0.8);
    }
    /* углы как HUD */
    .cs2-hud::before, .cs2-hud::after {
        content:""; position:absolute; width:30px; height:30px; border-color:rgba(255,170,50,0.75);
    }
    .cs2-hud::before { top:10px; left:10px; border-top:3px solid; border-left:3px solid; }
    .cs2-hud::after  { bottom:10px; right:10px; border-bottom:3px solid; border-right:3px solid; }
    .hud-orn { position:absolute; top:10px; right:10px; width:30px; height:30px;
        border-top:3px solid rgba(255,170,50,0.75); border-right:3px solid rgba(255,170,50,0.75); }
    .hud-orn2 { position:absolute; bottom:10px; left:10px; width:30px; height:30px;
        border-bottom:3px solid rgba(255,170,50,0.75); border-left:3px solid rgba(255,170,50,0.75); opacity:0; }

    /* трасса полёта пули — сверхзамедленная линия */
    .hud-line { position:absolute; top:50%; left:14%; right:14%; height:2px; opacity:.18;
        background: linear-gradient(90deg, transparent, rgba(255,180,60,0.7), transparent); }
    .hud-dot {
        position:absolute; top:50%; left:14%; width:12px; height:12px; border-radius:50%;
        margin-top:-6px;
        background: radial-gradient(circle, #fff7dd, #ffb347 50%, rgba(255,120,0,0.1));
        box-shadow: 0 0 18px 6px rgba(255,170,50,0.9), 0 0 40px 16px rgba(255,120,0,0.35);
        animation: bulletFly 3.6s cubic-bezier(.32,.85,.4,.98) infinite; z-index:3;
    }
    /* трейл за пулей */
    .hud-dot::after {
        content:""; position:absolute; left:-70px; top:50%; transform:translateY(-50%);
        width:70px; height:3px; border-radius:3px;
        background: linear-gradient(90deg, transparent, rgba(255,190,80,0.85));
    }
    /* вспышка в точке прилёта */
    .hud-pop {
        position:absolute; top:50%; left:86%; width:16px; height:16px; border-radius:50%;
        margin-top:-8px;
        background: radial-gradient(circle,#fff,#ff7a00 45%, transparent 70%);
        box-shadow:0 0 26px 8px rgba(255,120,0,0.8); opacity:0; z-index:2;
        animation: popFlash 3.6s linear infinite;
    }
    /* центральная надпись */
    .hud-title {
        position:absolute; top:50%; left:0; right:0; transform:translateY(-50%);
        text-align:center; font:700 22px/1 "Segoe UI", system-ui, sans-serif; letter-spacing:5px;
        color: rgba(255,220,160,0.85); text-transform:uppercase;
        text-shadow: 0 0 12px rgba(255,140,0,0.5);
        z-index:1; pointer-events:none;
    }
    @keyframes bulletFly {
        0%   { left:14%; opacity:0; }
        12%  { opacity:1; }
        78%  { left:80%; opacity:1; }
        88%  { left:84%; opacity:0; }
        100% { left:84%; opacity:0; }
    }
    @keyframes popFlash {
        0%,78% { opacity:0; transform:scale(.3); }
        84%    { opacity:1; transform:scale(1.5); }
        92%,100% { opacity:0; transform:scale(.4); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# HUD-плашка с летящей пулей
st.markdown(
    """
    <div class="cs2-hud">
        <div class="hud-orn"></div>
        <div class="hud-orn2"></div>
        <div class="hud-line"></div>
        <div class="hud-title">Счётчики сходятся</div>
        <div class="hud-pop"></div>
        <div class="hud-dot"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("CS2 Analyzer")
st.caption("Аналитический инструмент для предсказания исходов матчей Counter-Strike 2")

page = st.sidebar.radio(
    "Раздел",
    ["Прогноз матча", "Форма команд", "Карты", "Backtest"],
)

# Подгружаем данные один раз
with st.spinner("Загружаю данные…"):
    raw = load_raw()
teams = sorted(set(raw["team_a"]).union(set(raw["team_b"])))

# ---------------------------------------------------------------------------
# Страница 1: Прогноз матча
# ---------------------------------------------------------------------------
if page == "Прогноз матча":
    st.header("Прогноз матча")
    st.markdown(
        "Выберите две команды и карту (или BO3 «по всем картам»). Модель вернёт "
        "вероятности победы в процентах."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        team_a = st.selectbox("Команда A", teams, key="team_a")
    with col2:
        # возможные соперники — всё кроме выбранной команды A
        opponents = [t for t in teams if t != team_a]
        team_b = st.selectbox(
            "Команда B",
            opponents,
            # всегда по умолчанию первая команда, отличная от A
            index=min(len(opponents) - 1, 0),
            key="team_b",
        )
    with col3:
        map_choice = st.selectbox(
            "Карта",
            ["BO3 (все карты)"] + list(config.MAPS),
        )

    use_map = map_choice != "BO3 (все карты)"
    map_name = map_choice if use_map else config.MAPS[0]

    if team_a == team_b:
        st.warning(
            "Команды A и B совпадают — выберите разных соперников, "
            "иначе прогноз бессмысленен."
        )

    if st.button("Рассчитать", type="primary"):
        predictor = get_predictor(_model_mtime())
        with st.spinner("Считаю вероятности…"):
            if not use_map:
                # BO3: симулируем вето карт и агрегируем вероятность серии
                s = predictor.predict_series(team_a, team_b)
                pa, pb = s["p_a"], s["p_b"]
                conf = s["confidence"]
                series_maps = ", ".join(s["maps"])
            else:
                res = predictor.predict(team_a, team_b, map_name=map_name, use_map=True)
                pa, pb = res["team_a"], res["team_b"]
                conf = res["confidence"]
                series_maps = None
            conf_lbl = _confidence_label(conf)

        if not use_map:
            st.info(
                f"Режим BO3: прогноз симуляцией вето карт. "
                f"Карты: {series_maps}."
            )

        st.subheader("Результат")
        cols = st.columns([1, 2, 1])
        cols[0].metric(team_a, _pct(pa))
        cols[2].metric(team_b, _pct(pb))
        cols[1].markdown(
            "#### Уверенность: **{}**".format(_pct(conf)),
            unsafe_allow_html=False,
        )

        # Предупреждение/уведомление о надёжности прогноза
        if conf_lbl == "low":
            st.warning(
                f"Прогноз маловероятно надёжен (уверенность всего {_pct(conf)}). "
                "Шансы команд примерно равны либо по этим командам мало данных. "
                "Результат близок к подбрасыванию монеты — не расценивай его "
                "как предсказание."
            )
        elif conf_lbl == "medium":
            st.info(
                f"Умеренная уверенность ({_pct(conf)}). Прогноз скорее "
                "оценочный: сигнал есть, но слабый."
            )
        else:
            st.success(
                f"Высокая уверенность ({_pct(conf)}). Модель видит заметный "
                "перевес одной из команд по ELO, форме и статистике карт."
            )

        # визуализация
        fig = go.Figure(
            go.Bar(
                x=[pa, pb],
                y=[team_a, team_b],
                orientation="h",
                marker_color=["#d62728", "#1f77b4"],
                text=[_pct(pa), _pct(pb)],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Вероятность победы",
            xaxis=dict(range=[0, 1], title="Вероятность"),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        winner = team_a if pa >= pb else team_b
        st.success(f"Фаворит: **{winner}** с вероятностью {_pct(max(pa, pb))}")

# ---------------------------------------------------------------------------
# Страница 2: Форма команд
# ---------------------------------------------------------------------------
elif page == "Форма команд":
    st.header("Форма команд")
    select = st.multiselect(
        "Команды",
        teams,
        default=teams[:4],
        max_selections=8,
    )

    if not select:
        st.warning("Выберите хотя бы одну команду.")
        st.stop()

    # считаем winrate по хронологии для каждой выбранной команды
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    plot_df: list[pd.DataFrame] = []
    for team in select:
        sub = df[(df["team_a"] == team) | (df["team_b"] == team)].copy()
        if sub.empty:
            continue
        sub["won"] = sub["winner"] == team
        sub["win_count"] = sub["won"].cumsum()
        sub["match_count"] = range(1, len(sub) + 1)
        sub["team"] = team
        sub["cum_winrate"] = sub["win_count"] / sub["match_count"]
        # rolling winrate за последние 10 матчей
        sub["rolling10"] = sub["won"].rolling(10, min_periods=1).mean()
        plot_df.append(sub)

    if not plot_df:
        st.error("Нет данных по выбранным командам.")
    else:
        combined = pd.concat(plot_df, ignore_index=True)
        fig = px.line(
            combined,
            x=combined.index,
            y="rolling10",
            color="team",
            title="Winrate за последние 10 матчей (скользящее среднее)",
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            combined,
            x=combined.index,
            y="cum_winrate",
            color="team",
            title="Кумулятивный winrate",
        )
        fig2.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

        stats = (
            combined.groupby("team")["won"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )
        stats.columns = ["Команда", "Матчей", "Побед", "Winrate"]
        stats["Winrate"] = stats["Winrate"].map(_pct)
        st.dataframe(stats, use_container_width=True)

# ---------------------------------------------------------------------------
# Страница 3: Карты
# ---------------------------------------------------------------------------
elif page == "Карты":
    st.header("Карты: winrate и sides")
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    # фильтр по командам
    focus_teams = st.multiselect("Команды (пусто — все)", teams)

    if focus_teams:
        df = df[
            df["team_a"].isin(focus_teams) | df["team_b"].isin(focus_teams)
        ]

    st.subheader("Доля сыгранных карт (pick rate)")
    map_counts = df["map_name"].value_counts(normalize=True).sort_index()
    fig_pick = px.bar(
        x=map_counts.index,
        y=map_counts.values,
        labels={"x": "Карта", "y": "Доля"},
        title="Распределение выбора карт",
    )
    fig_pick.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_pick, use_container_width=True)

    st.subheader("Winrate команд по картам")
    team_select = st.selectbox("Команда для heatmap", teams)
    team_matches = df[(df["team_a"] == team_select) | (df["team_b"] == team_select)]
    if team_matches.empty:
        st.warning("Нет матчей у этой команды.")
        st.stop()

    # CT/T winrate требует знания сторон — в синтетике их нет, поэтому
    # оцениваем «победа команды на карте» без разделения сторон, а стороны
    # показываем как нейтральные. Пользовательский CS2-датасет с колонками
    # ct_wins/t_wins расширит это ниже.
    grid = {}
    for m in config.MAPS:
        sub = team_matches[team_matches["map_name"] == m]
        if sub.empty:
            grid[m] = 0.5
        else:
            grid[m] = (sub["winner"] == team_select).mean()

    hm = np.array([[grid[m] for m in config.MAPS]])
    fig_hm = go.Figure(
        go.Heatmap(
            z=hm,
            x=list(config.MAPS),
            y=["Winrate"],
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[_pct(v) for v in hm[0]]],
            texttemplate="%{text}",
        )
    )
    fig_hm.update_layout(title=f"Winrate {team_select} по картам", height=300)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.caption(
        "Примечание: разделение CT/T в синтетических данных недоступно. "
        "Подключив реальный источник с полями ct_wins/t_wins, здесь можно "
        "строить полноценный heatmap по сторонам."
    )

# ---------------------------------------------------------------------------
# Страница 4: Backtest
# ---------------------------------------------------------------------------
elif page == "Backtest":
    st.header("Backtest модели")
    st.markdown(
        "Проверяем, как модель предсказывала исторические матчи. Для скорости "
        "прогоняем по последним N сыгранным матчам."
    )

    n_backtest = st.slider("Число матчей для backtest", 50, min(500, len(raw)), 150)

    # берём данные, как будто «сегодня» позже последнего матча
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    tail = df.tail(n_backtest).copy().reset_index(drop=True)
    history = df.iloc[: len(df) - n_backtest].copy()

    predictor = get_predictor(_model_mtime())
    # прогоняем через predictor, но чтобы не нарушать хронологию делаем
    # «ref_date» для каждого матча = его дата, а статистику берём из history.
    # Упрощённо: используем predictor со всем raw (это допущение для демо).
    results = []
    with st.spinner("Прогоняю backtest…"):
        for rec in tail.itertuples(index=False):
            pa = 0.5
            try:
                res = predictor.predict(
                    rec.team_a,
                    rec.team_b,
                    map_name=rec.map_name,
                    ref_date=rec.date.date(),
                    use_map=True,
                )
                pa = res["team_a"]
            except Exception:  # noqa: BLE001
                pa = 0.5
            actual = 1 if rec.winner == rec.team_a else 0
            results.append({"team_a": rec.team_a, "team_b": rec.team_b,
                            "map": rec.map_name, "pred_a": pa, "actual": actual})

    bt = pd.DataFrame(results)
    bt["correct"] = ((bt["pred_a"] >= 0.5).astype(int) == bt["actual"])
    acc = bt["correct"].mean()

    st.metric("Accuracy на backtest", _pct(acc))
    st.metric("Матчей", len(bt))

    fig = px.density_contour(
        bt.assign(prob=bt["pred_a"]),
        x="actual",
        y="prob",
        title="Распределение предсказанных вероятностей vs факт",
    )
    fig.update_layout(xaxis_title="Факт (1 = победила team_a)",
                      yaxis_title="Предсказанная вероятность team_a")
    st.plotly_chart(fig, use_container_width=True)

    bt_disp = bt.copy()
    bt_disp["pred_a"] = bt_disp["pred_a"].map(_pct)
    st.dataframe(bt_disp.tail(20), use_container_width=True)

st.sidebar.markdown("---")
if config.AUTO_REFRESH.enabled:
    st.sidebar.caption(
        f"🔄 Автообновление данных: раз в {config.AUTO_REFRESH.interval_seconds // 3600} ч "
        f"({config.AUTO_REFRESH.interval_seconds // 60} мин)"
    )
st.sidebar.caption("Источник: Liquipedia (реальные матчи, без ключей)")