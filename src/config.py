"""Конфигурация проекта CS2 Analyzer.

Все пути, настройки и параметры вынесены сюда — никаких хардкодов в остальном коде.
Значения, которые пользователь может переопределить, читаются из переменных окружения
(файл ``.env``) через ``python-dotenv``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Базовые пути. Проект строится вокруг корневой папки, в которой лежит src/ и data/.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Убеждаемся, что папки существуют (idempotent).
for _dir in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Загрузка .env (не ошибка, если файла нет — тогда работают дефолты).
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")


def _env_str(key: str, default: str) -> str:
    """Читает строку из окружения с дефолтом."""
    value = os.getenv(key)
    return value.strip() if value else default


def _env_int(key: str, default: int) -> int:
    """Читает целое число из окружения с дефолтом."""
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Параметры сбора данных.
# ---------------------------------------------------------------------------
class ScraperConfig(BaseModel):
    """Параметры, управляющие сбором матчей."""

    #: число матчей в синтетическом датасете-заглушке
    synthetic_matches: int = Field(default=1200, ge=100, le=10000)
    #: задержка между запросами к HLTV (секунды) — уважаем rate-limit
    request_delay: float = Field(default=2.5, ge=0.5, le=10.0)
    #: таймаут HTTP-запросов (секунды)
    timeout: int = Field(default=15, ge=5, le=60)
    #: User-Agent, чтобы сайт не блокировал по умолчанию
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    )
    #: режим HLTV: "sync" — через свой парсер, "mock" — только синтетика
    mode: str = Field(default="mock")


class LiquipediaConfig(BaseModel):
    """Параметры источника Liquipedia.

    Liquipedia — открытый, без ключей источник; используется в режиме ``sync``.
    """

    #: категории, из которых берём завершённые турниры
    categories: tuple[str, ...] = ("Finished Tournaments",)
    #: максимум страниц турниров при авто-обходе
    max_pages: int = Field(default=40, ge=1, le=200)
    #: список конкретных страниц турниров (обход по умолчанию)
    tournament_pages: tuple[str, ...] = (
        # --- ESL Pro League (длинная история, много матчей) ---
        "ESL/Pro League/Season 11",
        "ESL/Pro League/Season 12",
        "ESL/Pro League/Season 13",
        "ESL/Pro League/Season 14",
        "ESL/Pro League/Season 15",
        "ESL/Pro League/Season 16",
        "ESL/Pro League/Season 17",
        "ESL/Pro League/Season 18",
        "ESL/Pro League/Season 19",
        "ESL/Pro League/Season 20",
        "ESL/Pro League/Season 21",
        "ESL/Pro League/Season 22",
        "ESL/Pro League/Season 23",
        # --- IEM 2023-2025 ---
        "Intel Extreme Masters/2023/Katowice",
        "Intel Extreme Masters/2023/Cologne",
        "Intel Extreme Masters/2023/Dallas",
        "Intel Extreme Masters/2024/Katowice",
        "Intel Extreme Masters/2024/Cologne",
        "Intel Extreme Masters/2024/Dallas",
        "Intel Extreme Masters/2024/Chengdu",
        "Intel Extreme Masters/2024/Rio",
        "Intel Extreme Masters/2025/Katowice",
        "Intel Extreme Masters/2025/Cologne",
        "Intel Extreme Masters/2025/Melbourne",
        # --- BLAST Premier 2022-2024 ---
        "BLAST/Premier/2022/Spring",
        "BLAST/Premier/2022/Fall",
        "BLAST/Premier/2023/Spring",
        "BLAST/Premier/2023/World Final",
        "BLAST/Premier/2024/Spring",
        "BLAST/Premier/2024/Fall",
        "BLAST/Premier/2024/World Final",
        # --- Мажоры ---
        "BLAST/Major/2023/Paris",
        "PGL/2024/Copenhagen",
        "Perfect World/Major/2024/Shanghai",
        "StarLadder/2025/Major",
        "PGL/2025/Astana",
        "PGL/2025/Bucharest",
        "PGL/2025/Cluj-Napoca",
        # --- Прочие крупные ---
        "Esports World Cup/2024",
        "Esports World Cup/2025",
        "Gamers8/2023",
        "Roobet Cup/2023",
        "Thunderpick/World Championship/2024",
        "Thunderpick/World Championship/2023",
        "BetBoom Dacha/2024/Belgrade/1",
        "CS Asia Championships/2023",
        "CCT/2026/Europe/Series 10",
    )
    #: при обходе категорий искать также дочерние страницы турнира
    include_subpages: bool = True


class AutoRefreshConfig(BaseModel):
    """Настройки автообновления данных в фоне.

    Когда Streamlit-приложение запущено постоянic, фоновый поток раз в
    ``interval_seconds`` докачивает новые матчи с Liquipedia и (если появились
    новые) переобучает модель. Управляется переменной окружения.
    """

    #: включить автообновление?
    enabled: bool = True
    #: период между пылесосами (секунды) — по умолчанию 3600 (раз в час)
    interval_seconds: int = Field(default=3600, ge=300, le=86400)
    #: доля запросов, передаваемая в скрапер (задержка между запросами)
    request_delay: float = 1.5


# ---------------------------------------------------------------------------
# Параметры фиче-инжиниринга.
# ---------------------------------------------------------------------------
class FeatureConfig(BaseModel):
    """Окна расчёта признаков."""

    #: окна формы команды (winrate за последние N матчей)
    form_windows: tuple[int, ...] = (5, 10, 30)
    #: окно recent form (последние 30 дней) — в днях
    recent_days: int = 30
    #: штраф за недавнюю замену в составе (доли к вероятности), если данные есть
    roster_penalty: float = 0.03


# ---------------------------------------------------------------------------
# Параметры модели.
# ---------------------------------------------------------------------------
class ModelConfig(BaseModel):
    """Гиперпараметры и прочее для обучения."""

    #: baseline-модель
    logreg_c: float = 0.5
    #: продвинутая модель
    rf_estimators: int = 300
    rf_depth: int | None = 7
    gb_estimators: int = 200
    gb_depth: int = 3
    gb_lr: float = 0.05
    #: валидация
    cv_splits: int = 5
    random_state: int = 42


# ---------------------------------------------------------------------------
# Единый объект-конфиг, к которому обращается весь код.
# ---------------------------------------------------------------------------

#: пути, загружаются из .env (префикс CS2_)
MATCHES_RAW = RAW_DIR / _env_str("CS2_MATCHES_RAW", "matches.csv")
MATCHES_PROCESSED = PROCESSED_DIR / _env_str(
    "CS2_MATCHES_PROCESSED", "features.parquet"
)
MODEL_PATH = MODELS_DIR / _env_str("CS2_MODEL_PATH", "model.pkl")
MODEL_META_PATH = MODELS_DIR / _env_str("CS2_MODEL_META", "model_meta.json")

SCRAPER = ScraperConfig(
    synthetic_matches=_env_int("CS2_SYNTHETIC_MATCHES", 1200),
    request_delay=_env_int("CS2_REQUEST_DELAY", 250) / 100.0,
    mode=_env_str("CS2_SCRAPER_MODE", "sync"),
)
LIQUIPEDIA = LiquipediaConfig(
    max_pages=_env_int("CS2_LIQUIPEDIA_MAX_PAGES", 40),
)
AUTO_REFRESH = AutoRefreshConfig(
    enabled=_env_str("CS2_AUTO_REFRESH", "1").lower() not in ("0", "false", "no"),
    interval_seconds=_env_int("CS2_AUTO_REFRESH_INTERVAL", 3600),
)
#: ключ Cito API (CS2 статистика) — из .env
CITO_API_KEY = _env_str("CITO_API_KEY", "")
#: базовый URL Cito API
CITO_BASE = "https://api.citoapi.com/api/v1"
#: папка кэша Cito-запросов (чтобы не жечь лимит 500/мес)
CITO_CACHE_DIR = DATA_DIR / "cito_cache"
CITO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = FeatureConfig()
MODEL = ModelConfig()

#: карты, используемые в CS2 (используется для one-hot признаков карт)
MAPS: tuple[str, ...] = (
    "Ancient",
    "Anubis",
    "Dust2",
    "Inferno",
    "Mirage",
    "Nuke",
    "Vertigo",
)