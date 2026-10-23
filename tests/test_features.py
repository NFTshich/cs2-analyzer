# Тесты для фиче-инжиниринга.

import pandas as pd
import pytest

from src import config
from src.features import FeatureEngine, build_features
from src.utils import generate_synthetic_matches


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Небольшой детерминированный датасет для проверки."""
    rows = [
        dict(match_id="m1", date="2024-01-01", team_a="NAVI", team_b="G2",
             map_name="Mirage", score_a=2, score_b=0, winner="NAVI",
             event_name="E1", tier="S", rank_a=1, rank_b=3, best_of=3),
        dict(match_id="m2", date="2024-01-02", team_a="G2", team_b="FaZe",
             map_name="Inferno", score_a=1, score_b=2, winner="FaZe",
             event_name="E1", tier="S", rank_a=3, rank_b=2, best_of=3),
        dict(match_id="m3", date="2024-01-03", team_a="NAVI", team_b="FaZe",
             map_name="Nuke", score_a=2, score_b=1, winner="NAVI",
             event_name="E2", tier="A", rank_a=1, rank_b=2, best_of=3),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_generate_synthetic_columns() -> None:
    df = generate_synthetic_matches(n_matches=100, seed=7)
    assert len(df) == 100
    for col in [
        "match_id", "date", "team_a", "team_b", "map_name", "score_a",
        "score_b", "winner", "event_name", "tier", "rank_a", "rank_b", "best_of",
    ]:
        assert col in df.columns, f"missing column: {col}"
    assert df["date"].is_monotonic_increasing


def test_engine_no_target_leakage(sample_df: pd.DataFrame) -> None:
    """Первый матч команды не должен использовать результаты из будущего."""
    eng = FeatureEngine(sample_df)
    out = eng.build()
    # Для первого матча (m1) у NAVI ещё нет истории до него:
    first_row = out.iloc[0]
    assert first_row["a_overall"] == 0.5  # нейтральный приори
    # Целевая переменная корректна
    assert out["target"].tolist() == [1, 0, 1]


def test_feature_count(sample_df: pd.DataFrame) -> None:
    eng = FeatureEngine(sample_df)
    out = eng.build()
    assert "form5_diff" in out.columns
    assert "map_diff" in out.columns
    assert "h2h_diff" in out.columns


def test_build_features_has_onehot(sample_df: pd.DataFrame) -> None:
    out = build_features(sample_df)
    assert f"map_{config.MAPS[0]}" in out.columns
    assert out["target"].dtype == int or out["target"].dtype.kind == "i"


def test_swap_columns_preserves_symmetry() -> None:
    """Разностные признаки при зеркалировании должны менять знак."""
    from src.model import _swap_columns

    df = pd.DataFrame({
        "team_a": ["NAVI", "G2"],
        "team_b": ["Spirit", "FaZe"],
        "a_elo": [1258.0, 1200.0],
        "b_elo": [1329.0, 1150.0],
        "elo_diff": [1258 - 1329, 1200 - 1150],
        "form5_diff": [0.1, -0.2],
        "target": [1, 0],
        "map_mirage": [1, 0],
    })
    sw = _swap_columns(df)
    # a_elo должен стать старым b_elo
    assert sw["a_elo"].tolist() == df["b_elo"].tolist()
    # elo_diff должен сменить знак
    assert (sw["elo_diff"] == -df["elo_diff"]).all()
    assert (sw["form5_diff"] == -df["form5_diff"]).all()


def test_roster_freshness_tracks_change() -> None:
    """Признак свежести состава падает при ротации и растёт при стабильности."""
    from src.features import TeamTracker

    tr = TeamTracker()
    roster_a = ("s1mple", "b1t", "iM", "w0nderful", "jL")
    roster_b = ("s1mple", "b1t", "iM", "w0nderful", "jL")
    # первые 5 матчей — состав стабилен
    for i in range(5):
        tr.record(pd.Timestamp(f"2024-01-0{i+1}").date(), True, "Mirage", "Opp",
                  opponent_elo=1000.0, roster=roster_a)
    # после 5 стабильных матчей свежесть уже не 0
    assert tr.roster_freshness() > 0.0
    # меняем состав -> свежесть падает к 0
    roster_new = ("s1mple", "b1t", "iM", "w0nderful", "zont1x")
    tr.record(pd.Timestamp("2024-02-01").date(), True, "Nuke", "Opp2",
              opponent_elo=1000.0, roster=roster_new)
    assert tr.roster_freshness() < 0.05


def test_parse_roster_json() -> None:
    """Парсер состава из JSON."""
    from src.features import _parse_roster

    assert _parse_roster(None) is None
    assert _parse_roster('["dev1ce", "s1mple"]') == ("dev1ce", "s1mple")