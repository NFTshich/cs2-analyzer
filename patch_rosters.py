"""Обогащает matches.csv составами команд (TeamCard) из Liquipedia.

Составы извлекаются с ГЛАВНЫХ страниц турниров (там находятся шаблоны
{{TeamCard}}) и применяются ко всем матчам этого турнира (по event_name),
т.к. состав команды на турнир обычно постоянный. Это быстрый способ
перезаполнить roster_a/roster_b без полного пересканирования всех подстраниц.

Запуск: python patch_rosters.py
"""

import sys
from pathlib import Path
import re

import mwparserfromhell
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.liquipedia import LiquipediaClient  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def extract_teamcard_rosters(client: LiquipediaClient, page: str) -> dict:
    """Возвращает {нормализованное_имя: [игроки]} из TeamCard страницы."""
    wikitext = client.get_wikitext(page)
    if not wikitext:
        return {}
    code = mwparserfromhell.parse(wikitext)
    rosters: dict[str, list[str]] = {}
    for t in code.filter_templates():
        if str(t.name).strip().lower() != "teamcard":
            continue

        def gp(name: str) -> str:
            try:
                val = str(t.get(name)).strip()
            except (ValueError, KeyError):
                return ""
            if "=" in val and val.split("=", 1)[0].strip().lower() == name.lower():
                val = val.split("=", 1)[1].strip()
            return val

        team = gp("team")
        if not team:
            continue
        players = [
            gp(x)
            for x in ("p1", "p2", "p3", "p4", "p5")
            if gp(x) and gp(x).lower() not in ("tbd", "tba", "")
        ]
        if players:
            rosters.setdefault(_norm(team), players)
    return rosters


def main() -> None:
    csv_path = config.MATCHES_RAW
    df = pd.read_csv(csv_path)
    client = LiquipediaClient()
    if "roster_a" not in df.columns:
        df["roster_a"] = ""
        df["roster_b"] = ""

    # match_id имеет формат "page|index" — страница содержит реальный путь
    # с '/'. Для подстраниц (page/X) составы берём с КОРНЕВОЙ страницы турнира
    # (отсекаем /<sub> части до первого уровня).
    main_pages = set(config.LIQUIPEDIA.tournament_pages)

    def _root_page(match_id: str) -> str:
        # "ESL/Pro League/Season 20|12" -> "ESL/Pro League/Season 20"
        page = str(match_id).rsplit("|", 1)[0]
        return page

    # сопоставим корневую страницу -> составы
    page_rosters: dict[str, dict] = {}
    for mid in df["match_id"].dropna().unique():
        page = _root_page(mid)
        if page in page_rosters:
            continue
        # главная страница турнира = первая часть до '/' уровня, но лучше:
        # ищем в конфиге точное или префикс-совпадение на корневую страницу
        # обработка: извлекаем составы только для известных турниров конфига
        matched = None
        for p in main_pages:
            if page.startswith(p) or p.startswith(page):
                matched = p
                break
        if matched is None:
            # попробуем саму страницу как корневую
            page_rosters[page] = {}
            continue
        try:
            rosters = extract_teamcard_rosters(client, matched)
            page_rosters[page] = rosters
            logger.info("{} -> {} составов", matched, len(rosters))
        except Exception as exc:  # noqa: BLE001
            logger.warning("{}: ошибка {}", matched, exc)
            page_rosters[page] = {}

    # применяем составы к матчам
    def _fill(row, which: str) -> str:
        page = _root_page(row["match_id"])
        rosters = page_rosters.get(page, {})
        if not rosters:
            return row[which]
        team = row["team_a"] if which == "roster_a" else row["team_b"]
        pl = rosters.get(_norm(team), [])
        import json

        return json.dumps(pl, ensure_ascii=False) if pl else row[which]

    df["roster_a"] = df.apply(lambda r: _fill(r, "roster_a"), axis=1)
    df["roster_b"] = df.apply(lambda r: _fill(r, "roster_b"), axis=1)

    n_a = int(df["roster_a"].str.len().gt(2).sum())
    n_b = int(df["roster_b"].str.len().gt(2).sum())
    logger.info("Обновлено: roster_a заполнен у {} матчей, roster_b у {}", n_a, n_b)

    df.to_csv(csv_path, index=False)
    logger.info("Состав file {} обновлён ({} строк).", csv_path, len(df))


if __name__ == "__main__":
    main()