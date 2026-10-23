"""Загрузка данных Cito API для модели (рейтинг команд + статы игроков).

Разово тянет с Cito API:
  1. мировой рейтинг команд (data/processed/cito_rankings.parquet);
  2. профили топ-команд (составы: playerNickname, playerSlug, rating, mapsPlayed)
     — используется БЕЗ отдельных запросов на игрока;
  3. опционально игровые статы игроков через /players/{slug}/stats
     (если FETCH_PLAYER_STATS=1) — в data/processed/cito_players.parquet.

Все ответы кэшируются на диск, поэтому повторные запуски почти не тратят
бесплатный лимит 500 запросов/месяц.

Запуск:
  python fetch_cito_data.py            # рейтинг + составы (без лишних запросов)
  python fetch_cito_data.py --players  # + докачать статы игроков (больше запросов)
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.cito import CitoClient  # noqa: E402

#: сколько топ-команд берём для профилей и игроков
TOP_TEAMS = 40
#: сколько игроков на команду (ядро состава)
PLAYERS_PER_TEAM = 5
#: пауза между запросами, сек (бесплатный план: 10 запросов/минуту => >=6 сек)
RATE_LIMIT_WAIT = 6.5
#: брать ли глубокие статы игрока через отдельные запросы
FETCH_PLAYER_STATS = "--players" in sys.argv


def main() -> None:
    if not config.CITO_API_KEY:
        logger.error("CITO_API_KEY не задан. Добавь его в data/.env")
        sys.exit(1)

    client = CitoClient()
    fetch_players = FETCH_PLAYER_STATS or os.getenv("CITO_FETCH_PLAYERS", "0") == "1"

    # 1) мировой рейтинг
    ranking = client.world_ranking(limit=100)
    if not ranking:
        logger.error("Cito вернул пустой рейтинг — проверь ключ/лимит.")
        sys.exit(1)
    rank_df = pd.DataFrame([
        {
            "team": r.get("teamName"),
            "team_id": r.get("teamId"),
            "rank": r.get("rank"),
            "points": r.get("points"),
            "region": r.get("region"),
        }
        for r in ranking
    ])
    rank_df.to_parquet(config.PROCESSED_DIR / "cito_rankings.parquet", index=False)
    logger.info("Рейтинг: {} команд сохранён", len(rank_df))

    # 2) профили команд + игроки (из состава — без лишних запросов)
    player_rows: list[dict] = []
    team_profiles: list[dict] = []
    for row in rank_df.head(TOP_TEAMS).itertuples(index=False):
        tid = row.team_id
        if not tid:
            continue
        prof = client.team_profile(tid)
        if not prof:
            continue
        time.sleep(RATE_LIMIT_WAIT)

        players = prof.get("activePlayers") or prof.get("roster") or []
        players = [p for p in players if isinstance(p, dict)]
        # имя состава: nick или полное имя
        roster_names = [
            p.get("playerNickname") or p.get("playerName") or p.get("nick") or p.get("name")
            for p in players
        ]
        team_profiles.append({
            "team": row.team,
            "team_id": tid,
            "rank": row.rank,
            "points": row.points,
            "world_ranking": prof.get("worldRanking"),
            "valve_ranking": prof.get("valveRanking"),
            "roster": roster_names,
        })

        # игроки состава: rating/maps есть прямо в статье состава
        for p in players[:PLAYERS_PER_TEAM]:
            pname = p.get("playerNickname") or p.get("playerName") or p.get("nick") or p.get("name")
            pslug = p.get("playerSlug") or p.get("slug")
            pid = p.get("playerId") or p.get("id")
            rating = p.get("rating")
            maps = p.get("mapsPlayed")
            row_out = {
                "team": row.team,
                "player": pname or pslug or str(pid),
                "player_id": pid,
                "slug": pslug,
                "maps": maps,
                "rating": rating,
                "role": p.get("role"),
            }
            # если просили глубокие статы и есть slug — докачиваем K/D и т.п.
            if fetch_players and pslug:
                stats = client.player_stats(pslug)
                time.sleep(RATE_LIMIT_WAIT)
                totals = (stats.get("totals") or {}) if isinstance(stats, dict) else {}
                kd = 0.0
                n_deaths = totals.get("deaths", 0) or 1
                if n_deaths:
                    kd = round((totals.get("kills", 0) or 0) / n_deaths, 3)
                row_out.update({
                    "kills": totals.get("kills", 0),
                    "deaths": totals.get("deaths", 0),
                    "assists": totals.get("assists", 0),
                    "headshot_pct": totals.get("headshotPercent", 0),
                    "kd": kd,
                })
            player_rows.append(row_out)

        logger.info("  {} (rank {}): {}/{} игроков", row.team, row.rank,
                    len(roster_names), PLAYERS_PER_TEAM)

    players_df = pd.DataFrame(player_rows)
    players_df.to_parquet(config.PROCESSED_DIR / "cito_players.parquet", index=False)
    logger.info("Игроков в базе: {} (глубокие статы: {})",
                len(players_df), bool(fetch_players))

    with open(config.PROCESSED_DIR / "cito_teams.json", "w", encoding="utf-8") as f:
        json.dump(team_profiles, f, ensure_ascii=False, indent=2)
    logger.info("Команд в профилях: {}", len(team_profiles))
    logger.info("Готово. Файлы cito_*.parquet в data/processed/.")


if __name__ == "__main__":
    main()