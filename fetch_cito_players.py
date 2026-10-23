"""Докачка индивидуальных статов игроков из Cito по никам.

Берёт уникальные ники из составов команд (matches.csv) для команд из топ-40
рейтинга Cito и тянет их карьерные статы (kills/deaths/kd/headshot) через
/cs2/players/{nick}/stats. Результаты кэшируются в
data/processed/cito_player_stats.parquet.

Бесплатный лимит — 500 запросов/мес, поэтому скрипт поддерживает:
  * инкрементальность (пропускает уже скачанных);
  * лимит на запуск (LIMIT_PER_RUN, по умолчанию 40) — можно докачивать
    несколькими запусками;

Запуск:
  python fetch_cito_players.py          # до +40 ников за запуск
  python fetch_cito_players.py --limit 60
"""

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.cito import CitoClient  # noqa: E402

#: сколько ников обработать за один запуск
LIMIT_PER_RUN = 40
RATE_LIMIT_WAIT = 6.5

OUT_PATH = config.PROCESSED_DIR / "cito_player_ratings.parquet"


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if pd.notna(s) else ""


def _collect_nicks() -> list[tuple[str, str, str]]:
    """Возвращает уникальные (nick, team, roster_json) для топ-40 команд."""
    raw = pd.read_csv(config.MATCHES_RAW)
    rk = pd.read_parquet(config.PROCESSED_DIR / "cito_rankings.parquet")
    rank_map = {_norm(t): t for t in rk["team"]}

    seen: dict[str, tuple] = {}
    for _, r in raw.iterrows():
        for col, teamcol in (("roster_a", "team_a"), ("roster_b", "team_b")):
            js = r[col]
            if pd.isna(js):
                continue
            try:
                nicks = json.loads(js)
            except Exception:  # noqa: BLE001
                continue
            tc = _norm(r[teamcol])
            if tc not in rank_map:
                continue
            team_full = rank_map[tc]
            for nk in nicks:
                key = _norm(nk)
                if key and key not in seen:
                    seen[key] = (key, team_full, str(nk))
    # отсортируем для детерминизма
    return [seen[k] for k in sorted(seen)]


def main() -> None:
    limit = LIMIT_PER_RUN
    if "--limit" in sys.argv:
        try:
            limit = int(sys.argv[sys.argv.index("--limit") + 1])
        except (ValueError, IndexError):
            pass

    if not config.CITO_API_KEY:
        logger.error("CITO_API_KEY не задан.")
        sys.exit(1)

    # уже скачанные (восстанавливаем dict-строки)
    rows = []
    done: set = set()
    if OUT_PATH.exists():
        df0 = pd.read_parquet(OUT_PATH)
        if not df0.empty and "player" in df0.columns:
            rows = df0.to_dict("records")
            done = {row["nick_norm"] if "nick_norm" in row else _norm(row["player"]) for row in rows}

    nicks = _collect_nicks()
    todo = [t for t in nicks if t[0] not in done]
    logger.info("Ников всего: {}, скачано: {}, осталось: {} (лимит запуска: {})",
                len(nicks), len(done), len(todo), limit)

    client = CitoClient()
    fetched = 0
    for key, team, nick in todo:
        if fetched >= limit:
            logger.info("Достигнут лимит запуска ({}). Запусти ещё раз для докачки.", limit)
            break
        try:
            car = client.player_career(key, game="cs2")
            time.sleep(RATE_LIMIT_WAIT)
            s = car if isinstance(car, dict) else {}
            tot = s.get("totals") or {}
            n_deaths = tot.get("deaths", 0) or 1
            kd = round((tot.get("kills", 0) or 0) / n_deaths, 3) if n_deaths else 0.0
            rows.append({
                "player": str(nick),
                "team": team,
                "nick_norm": key,
                "maps": tot.get("maps") or s.get("maps"),
                "rating": tot.get("rating"),
                "adr": tot.get("adr"),
                "kast": tot.get("kast"),
                "kills": tot.get("kills", 0),
                "deaths": tot.get("deaths", 0),
                "assists": tot.get("assists", 0),
                "headshot_pct": tot.get("headshotPercent", 0),
                "kd": kd,
            })
            fetched += 1
            logger.info("  +{} {} ({}) rating={} kd={}", fetched, str(nick)[:16], team,
                        tot.get("rating"), kd)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  {} не найден/ошибка: {}", str(nick)[:20], getattr(exc, "code", str(exc)[:40]))

    out = pd.DataFrame(rows)
    if len(out) > 1:
        out = out.drop_duplicates(subset=["nick_norm"], keep="last")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Сохранено: {} игроков в {}", len(out), OUT_PATH)


if __name__ == "__main__":
    main()