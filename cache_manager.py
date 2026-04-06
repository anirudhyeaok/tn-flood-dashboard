"""
cache_manager.py — Background Data Cache for TN Flood Intelligence
==================================================================
Runs a background APScheduler thread that pre-fetches Open-Meteo data
for all 38 TN districts on a schedule. Flask reads from cache instantly.

Cache files (written to flood_dashboard/cache/):
  cache/live_cache.json     — refreshed every 15 minutes
  cache/forecast_cache.json — refreshed every 60 minutes
  cache/cyclone_cache.json  — refreshed every 10 minutes

Why this matters:
  Without caching: 38 sequential HTTP calls on every page load (~15-30s, often timeout)
  With caching:    Single file read on every page load (<1ms)

Scheduler: APScheduler BackgroundScheduler (runs in same process as Flask,
no separate worker needed for single-server deployment).
"""

import os
import json
import logging
import threading
from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

# ── Cache directory ───────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

LIVE_CACHE_FILE     = os.path.join(CACHE_DIR, "live_cache.json")
FORECAST_CACHE_FILE = os.path.join(CACHE_DIR, "forecast_cache.json")
CYCLONE_CACHE_FILE  = os.path.join(CACHE_DIR, "cyclone_cache.json")

# ── Staleness thresholds (seconds) ───────────────────────────────────────────
LIVE_MAX_AGE_S     = 20 * 60   # 20 min — warn if live data older than this
FORECAST_MAX_AGE_S = 90 * 60   # 90 min
CYCLONE_MAX_AGE_S  = 15 * 60   # 15 min

# ── Write lock (prevent partial reads during write) ───────────────────────────
_write_lock = threading.Lock()

log = logging.getLogger("cache_manager")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ── File I/O ──────────────────────────────────────────────────────────────────

def _write_cache(filepath: str, data: dict):
    """Atomically writes cache to disk with a written_at timestamp."""
    data["_cache_written_at"] = datetime.now(timezone.utc).isoformat()
    with _write_lock:
        tmp = filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, filepath)  # atomic on all platforms
    log.info(f"Cache written: {os.path.basename(filepath)}")


def _read_cache(filepath: str) -> dict | None:
    """Reads cache from disk. Returns None if file missing or corrupt."""
    if not os.path.exists(filepath):
        return None
    try:
        with _write_lock:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Cache read failed ({os.path.basename(filepath)}): {e}")
        return None


def _cache_age_seconds(data: dict) -> float:
    """Returns how many seconds ago the cache was written. 999999 if unknown."""
    written = data.get("_cache_written_at")
    if not written:
        return 999999
    try:
        written_dt = datetime.fromisoformat(written)
        return (datetime.now(timezone.utc) - written_dt).total_seconds()
    except Exception:
        return 999999


def is_cache_fresh(filepath: str, max_age_s: int) -> bool:
    data = _read_cache(filepath)
    if not data:
        return False
    return _cache_age_seconds(data) < max_age_s


# ── Scheduled jobs ────────────────────────────────────────────────────────────

def refresh_live_cache():
    """Fetches real-time rainfall for all districts and writes to cache."""
    log.info("Refreshing live cache...")
    try:
        from live_engine import get_live_risk_map
        data = get_live_risk_map()
        _write_cache(LIVE_CACHE_FILE, data)
    except Exception as e:
        log.error(f"Live cache refresh failed: {e}")


def refresh_forecast_cache():
    """Fetches 3-day forecast for all districts and writes to cache."""
    log.info("Refreshing forecast cache...")
    try:
        from forecast_engine import get_forecast_risk_map
        data = get_forecast_risk_map()
        _write_cache(FORECAST_CACHE_FILE, data)
    except Exception as e:
        log.error(f"Forecast cache refresh failed: {e}")


def refresh_cyclone_cache():
    """Scans all districts for cyclone-force winds and writes to cache."""
    log.info("Refreshing cyclone cache...")
    try:
        from cyclone_tracker import scan_cyclone_activity
        data = scan_cyclone_activity()
        _write_cache(CYCLONE_CACHE_FILE, data)
    except Exception as e:
        log.error(f"Cyclone cache refresh failed: {e}")


# ── Public read API (used by app.py) ──────────────────────────────────────────

def get_live_data() -> dict:
    """
    Returns live risk map from cache.
    Falls back to a fresh fetch if cache is missing or too old.
    Always returns a dict — never raises.
    """
    data = _read_cache(LIVE_CACHE_FILE)
    if data is None or _cache_age_seconds(data) > LIVE_MAX_AGE_S:
        log.warning("Live cache stale or missing — fetching synchronously.")
        refresh_live_cache()
        data = _read_cache(LIVE_CACHE_FILE)
    return data or {"_meta": {"source": "Cache unavailable", "fetched_at": "Unknown", "mode": "LIVE"}}


def get_forecast_data() -> dict:
    """
    Returns forecast map from cache.
    Falls back to a fresh fetch if cache is missing or too old.
    """
    data = _read_cache(FORECAST_CACHE_FILE)
    if data is None or _cache_age_seconds(data) > FORECAST_MAX_AGE_S:
        log.warning("Forecast cache stale or missing — fetching synchronously.")
        refresh_forecast_cache()
        data = _read_cache(FORECAST_CACHE_FILE)
    return data or {"_meta": {"source": "Cache unavailable", "fetched_at": "Unknown", "mode": "FORECAST"}}


def get_cyclone_data() -> dict:
    """
    Returns cyclone scan from cache.
    Falls back to a fresh fetch if cache is missing or too old.
    """
    data = _read_cache(CYCLONE_CACHE_FILE)
    if data is None or _cache_age_seconds(data) > CYCLONE_MAX_AGE_S:
        log.warning("Cyclone cache stale or missing — fetching synchronously.")
        refresh_cyclone_cache()
        data = _read_cache(CYCLONE_CACHE_FILE)
    return data or {
        "is_active": False, "system_type": "No Active System",
        "severity": -1, "affected_districts": [],
        "approximate_track": [], "max_wind_district": None,
        "max_wind_kmh": 0.0, "fetched_at": "Unknown",
        "source": "Cache unavailable",
    }


def get_cache_status() -> dict:
    """
    Returns metadata about cache freshness.
    Used by the attribution panel in the UI.
    """
    def _status(filepath, max_age):
        data = _read_cache(filepath)
        if not data:
            return {"fresh": False, "age_s": None, "written_at": "Never"}
        age = _cache_age_seconds(data)
        return {
            "fresh":      age < max_age,
            "age_s":      int(age),
            "age_label":  f"{int(age // 60)}m {int(age % 60)}s ago",
            "written_at": data.get("_cache_written_at", "Unknown"),
        }

    return {
        "live":     _status(LIVE_CACHE_FILE,     LIVE_MAX_AGE_S),
        "forecast": _status(FORECAST_CACHE_FILE, FORECAST_MAX_AGE_S),
        "cyclone":  _status(CYCLONE_CACHE_FILE,  CYCLONE_MAX_AGE_S),
    }


# ── Scheduler setup ───────────────────────────────────────────────────────────

_scheduler = None

def _on_job_event(event):
    if event.exception:
        log.error(f"Scheduled job failed: {event.job_id} — {event.exception}")

def start_scheduler():
    """
    Starts the APScheduler background thread.
    Call once at app startup (from app.py).
    Safe to call multiple times — won't start a second scheduler.
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        log.info("Scheduler already running.")
        return

    _scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    _scheduler.add_listener(_on_job_event, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    # Schedule jobs
    _scheduler.add_job(refresh_live_cache,     "interval", minutes=15,  id="live",     misfire_grace_time=60)
    _scheduler.add_job(refresh_forecast_cache, "interval", minutes=60,  id="forecast", misfire_grace_time=120)
    _scheduler.add_job(refresh_cyclone_cache,  "interval", minutes=10,  id="cyclone",  misfire_grace_time=60)

    _scheduler.start()
    log.info("Background scheduler started. Jobs: live(15m), forecast(60m), cyclone(10m)")

    # Prime caches immediately on startup in a separate thread
    # so Flask starts instantly and caches fill in the background
    def _prime():
        log.info("Priming caches on startup...")
        refresh_cyclone_cache()
        refresh_live_cache()
        refresh_forecast_cache()
        log.info("Cache priming complete.")

    threading.Thread(target=_prime, daemon=True, name="cache-primer").start()


def stop_scheduler():
    """Cleanly shuts down the scheduler. Call on app teardown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("Scheduler stopped.")