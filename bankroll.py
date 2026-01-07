#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bankroll.py — Bankroll Manager Pokerdom-oriented.

Commands:
  init
  status
  add-session
  add
  history
  report
  config

Data file:
  bankroll_data.json (created next to this script)

Python 3.10+
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple


DATA_FILENAME = "bankroll_data.json"

DEFAULT_CASH_LIMITS = [2, 5, 10, 25, 50, 100, 200]  # BB value in currency; buy-in = 100 * BB
SUPPORTED_CURRENCIES = ["RUB", "USD", "EUR"]
SUPPORTED_MODES = ["cash", "mtt"]
SUPPORTED_STRATEGIES = ["aggressive", "standard", "conservative"]

# Strategy rules:
CASH_STRATEGIES = {
    "aggressive": {"buyins_for_limit": 25, "move_down_threshold_buyins": 20},
    "standard": {"buyins_for_limit": 35, "move_down_threshold_buyins": 25},
    "conservative": {"buyins_for_limit": 50, "move_down_threshold_buyins": 40},
}

MTT_STRATEGIES = {
    "aggressive": {"avg_buyin_multiplier": 100},
    "standard": {"avg_buyin_multiplier": 150},
    "conservative": {"avg_buyin_multiplier": 200},
}


# --------------------------
# Utilities
# --------------------------

def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def parse_iso_dt(value: str) -> datetime:
    # Accept full ISO "YYYY-MM-DDTHH:MM:SS" and also "YYYY-MM-DDTHH:MM"
    try:
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"Invalid ISO datetime: {value}") from e


def read_date_yyyy_mm_dd(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"Invalid date (expected YYYY-MM-DD): {s}") from e


def format_money(x: float, currency: str) -> str:
    return f"{x:,.2f} {currency}".replace(",", " ")


def yn_prompt(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes", "да", "д"):
            return True
        if ans in ("n", "no", "нет", "н"):
            return False
        print("Введите Y или N.")


def input_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    choices_lc = [c.lower() for c in choices]
    hint = f" ({'/'.join(choices)})"
    if default:
        hint += f" [default: {default}]"
    while True:
        v = input(prompt + hint + ": ").strip()
        if not v and default:
            v = default
        if v.lower() in choices_lc:
            # return canonical case from choices
            return choices[choices_lc.index(v.lower())]
        print(f"Неверное значение. Допустимо: {', '.join(choices)}")


def input_float(prompt: str, default: Optional[float] = None, allow_negative: bool = True) -> float:
    hint = ""
    if default is not None:
        hint = f" [default: {default}]"
    while True:
        raw = input(prompt + hint + ": ").strip().replace(" ", "").replace(",", ".")
        if not raw and default is not None:
            return float(default)
        try:
            val = float(raw)
            if not allow_negative and val < 0:
                print("Значение не может быть отрицательным.")
                continue
            return val
        except ValueError:
            print("Введите число (например 1234.56).")


def input_int(prompt: str, default: Optional[int] = None, allow_negative: bool = False) -> int:
    hint = ""
    if default is not None:
        hint = f" [default: {default}]"
    while True:
        raw = input(prompt + hint + ": ").strip()
        if not raw and default is not None:
            return int(default)
        try:
            val = int(raw)
            if not allow_negative and val < 0:
                print("Значение не может быть отрицательным.")
                continue
            return val
        except ValueError:
            print("Введите целое число.")


def data_path() -> str:
    # Store next to script (or current directory if __file__ is unavailable)
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    return os.path.join(base, DATA_FILENAME)


def load_data() -> Dict[str, Any]:
    path = data_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден файл данных: {path}. Запустите: python bankroll.py init")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Файл данных повреждён (JSON). Сделайте backup и пересоздайте через init: {path}") from e


def save_data(data: Dict[str, Any]) -> None:
    path = data_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def ensure_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal schema healing for future-proofing
    data.setdefault("profile", {})
    prof = data["profile"]
    prof.setdefault("currency", "RUB")
    prof.setdefault("room", "Pokerdom")
    prof.setdefault("mode", "cash")
    prof.setdefault("strategy", "standard")
    prof.setdefault("cash_limits", DEFAULT_CASH_LIMITS)

    prof.setdefault("cash_rules", CASH_STRATEGIES.get(prof["strategy"], CASH_STRATEGIES["standard"]).copy())
    prof.setdefault("mtt_rules", MTT_STRATEGIES.get(prof["strategy"], MTT_STRATEGIES["standard"]).copy())

    data.setdefault("bankroll", {})
    data["bankroll"].setdefault("current", 0.0)
    data["bankroll"].setdefault("created_at", now_iso())

    data.setdefault("ledger", [])
    if not isinstance(data["ledger"], list):
        data["ledger"] = []
    return data


def recalc_bankroll_from_ledger(data: Dict[str, Any]) -> float:
    total = 0.0
    for e in data.get("ledger", []):
        t = e.get("type")
        if t == "session":
            total += float(e.get("profit", 0.0))
        elif t == "adjustment":
            total += float(e.get("amount", 0.0))
        # unknown entries ignored
    data["bankroll"]["current"] = round(total, 2)
    return data["bankroll"]["current"]


def limit_to_label(bb_value: int) -> str:
    return f"NL{bb_value}"


def parse_limit_label(label: str) -> Optional[int]:
    # Accept NL10/nl10/10
    s = label.strip().lower()
    if s.startswith("nl"):
        s = s[2:]
    try:
        return int(s)
    except ValueError:
        return None


def buyin_for_limit(bb_value: int) -> float:
    # buy-in = 100 * BB (simplified)
    return 100.0 * float(bb_value)


# --------------------------
# Recommendation logic
# --------------------------

@dataclass
class CashRecommendation:
    recommended_limit: Optional[int]  # BB value
    next_shot_limit: Optional[int]
    move_down_limit: Optional[int]
    move_down_threshold: Optional[float]
    next_shot_threshold: Optional[float]
    min_required_for_min_limit: float


def calc_cash_recommendation(bankroll: float, limits: List[int], rules: Dict[str, Any]) -> CashRecommendation:
    if not limits:
        limits = DEFAULT_CASH_LIMITS[:]
    limits = sorted(set(int(x) for x in limits))
    buyins_for_limit = int(rules.get("buyins_for_limit", 35))
    move_down_threshold_buyins = int(rules.get("move_down_threshold_buyins", 25))

    # min bankroll for lowest limit
    min_required = buyins_for_limit * buyin_for_limit(limits[0])

    # find highest recommended
    recommended = None
    for bb in limits:
        if bankroll >= buyins_for_limit * buyin_for_limit(bb):
            recommended = bb
        else:
            break

    if recommended is None:
        # too low even for min limit
        return CashRecommendation(
            recommended_limit=None,
            next_shot_limit=limits[0],
            move_down_limit=None,
            move_down_threshold=None,
            next_shot_threshold=buyins_for_limit * buyin_for_limit(limits[0]),
            min_required_for_min_limit=min_required,
        )

    idx = limits.index(recommended)

    # next shot
    next_shot = limits[idx + 1] if idx + 1 < len(limits) else None
    next_shot_threshold = None
    if next_shot is not None:
        next_shot_threshold = buyins_for_limit * buyin_for_limit(next_shot)

    # move down
    move_down = limits[idx - 1] if idx - 1 >= 0 else None
    move_down_threshold = move_down_threshold_buyins * buyin_for_limit(recommended)

    return CashRecommendation(
        recommended_limit=recommended,
        next_shot_limit=next_shot,
        move_down_limit=move_down,
        move_down_threshold=move_down_threshold,
        next_shot_threshold=next_shot_threshold,
        min_required_for_min_limit=min_required,
    )


@dataclass
class MttRecommendation:
    recommended_abi: float


def calc_mtt_recommendation(bankroll: float, rules: Dict[str, Any]) -> MttRecommendation:
    mult = int(rules.get("avg_buyin_multiplier", 150))
    abi = bankroll / float(mult) if mult > 0 else 0.0
    return MttRecommendation(recommended_abi=round(abi, 2))


# --------------------------
# Stats
# --------------------------

def ledger_sorted(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    led = data.get("ledger", [])
    # Sort by timestamp if exists; fall back to insertion order
    def key(e: Dict[str, Any]) -> Tuple[int, str]:
        ts = e.get("timestamp")
        try:
            return (0, parse_iso_dt(ts).isoformat()) if ts else (1, "")
        except Exception:
            return (2, "")
    return sorted(led, key=key)


def last_sessions_in_days(data: Dict[str, Any], days: int) -> List[Dict[str, Any]]:
    cutoff = datetime.now() - timedelta(days=days)
    res = []
    for e in data.get("ledger", []):
        if e.get("type") != "session":
            continue
        ts = e.get("timestamp")
        try:
            dt = parse_iso_dt(ts)
        except Exception:
            continue
        if dt >= cutoff:
            res.append(e)
    return res


def find_last_session(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sessions = [e for e in data.get("ledger", []) if e.get("type") == "session" and e.get("timestamp")]
    if not sessions:
        return None
    # max by timestamp
    best = None
    best_dt = None
    for e in sessions:
        try:
            dt = parse_iso_dt(e["timestamp"])
        except Exception:
            continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best = e
    return best


# --------------------------
# Printing
# --------------------------

def print_header(data: Dict[str, Any]) -> None:
    prof = data["profile"]
    print("Pokerdom Bankroll Manager")
    print("-" * 28)
    print(f"Room: {prof.get('room', 'Pokerdom')}")
    print()


def print_status(data: Dict[str, Any], short: bool = False) -> None:
    data = ensure_schema(data)
    recalc_bankroll_from_ledger(data)

    prof = data["profile"]
    currency = prof["currency"]
    mode = prof["mode"]
    strategy = prof["strategy"]
    bankroll = float(data["bankroll"]["current"])

    if not short:
        print_header(data)

    print(f"Bankroll: {format_money(bankroll, currency)}")
    print(f"Mode: {mode} | Strategy: {strategy}")
    print()

    if mode == "cash":
        limits = prof.get("cash_limits", DEFAULT_CASH_LIMITS)
        rules = prof.get("cash_rules", CASH_STRATEGIES.get(strategy, CASH_STRATEGIES["standard"]))
        rec = calc_cash_recommendation(bankroll, limits, rules)

        if rec.recommended_limit is None:
            print("Recommended limit: (bankroll too low)")
            print(f"Minimum for {limit_to_label(min(limits))}: {format_money(rec.min_required_for_min_limit, currency)}")
            print(f"First target: {limit_to_label(min(limits))} at bankroll >= {format_money(rec.next_shot_threshold or 0.0, currency)}")
        else:
            print(f"Recommended limit: {limit_to_label(rec.recommended_limit)}")

            if rec.move_down_limit is not None and rec.move_down_threshold is not None:
                print(
                    f"Move down if bankroll < {format_money(rec.move_down_threshold, currency)}"
                    f" -> {limit_to_label(rec.move_down_limit)}"
                )
            else:
                # already at lowest limit
                if rec.move_down_threshold is not None:
                    print(f"Move down threshold (lowest): {format_money(rec.move_down_threshold, currency)}")

            if rec.next_shot_limit is not None and rec.next_shot_threshold is not None:
                print(
                    f"Next shot at bankroll >= {format_money(rec.next_shot_threshold, currency)}"
                    f" -> {limit_to_label(rec.next_shot_limit)}"
                )
            else:
                print("Next shot: (already at highest configured limit)")

        # warning if last session limit above recommended
        last = find_last_session(data)
        if last and rec.recommended_limit is not None:
            last_limit_label = last.get("limit")
            last_limit_val = parse_limit_label(str(last_limit_label)) if last_limit_label else None
            if last_limit_val is not None and last_limit_val > rec.recommended_limit:
                print()
                print("WARNING: Последний сыгранный лимит выше рекомендованного по BRM.")

    else:  # mtt
        rules = prof.get("mtt_rules", MTT_STRATEGIES.get(strategy, MTT_STRATEGIES["standard"]))
        rec = calc_mtt_recommendation(bankroll, rules)
        print(f"Recommended ABI: {format_money(rec.recommended_abi, currency)}")

    # last 7 days stats
    sessions_7d = last_sessions_in_days(data, 7)
    if sessions_7d:
        total = sum(float(s.get("profit", 0.0)) for s in sessions_7d)
        avg = total / len(sessions_7d)
        print()
        print(
            f"Last 7 days: sessions={len(sessions_7d)} | "
            f"profit={format_money(total, currency)} | avg={format_money(avg, currency)}"
        )
    else:
        print()
        print("Last 7 days: sessions=0")

    if not short:
        print()


def print_history(data: Dict[str, Any], n: int = 10, entry_type: Optional[str] = None) -> None:
    data = ensure_schema(data)
    currency = data["profile"]["currency"]
    items = list(reversed(ledger_sorted(data)))  # newest first
    if entry_type:
        items = [e for e in items if e.get("type") == entry_type]

    if not items:
        print("История пуста.")
        return

    items = items[:n]
    print(f"History (last {len(items)}):")
    print("-" * 60)
    for e in items:
        ts = e.get("timestamp", "N/A")
        t = e.get("type", "unknown")

        if t == "session":
            profit = float(e.get("profit", 0.0))
            game = e.get("game", "cash")
            limit = e.get("limit") or e.get("buyin") or "-"
            dur = e.get("duration_min")
            note = e.get("note", "")
            dur_s = f", {dur} min" if dur is not None else ""
            print(f"{ts} | session | {game} | {limit} | {format_money(profit, currency)}{dur_s}")
            if note:
                print(f"  note: {note}")
        elif t == "adjustment":
            amount = float(e.get("amount", 0.0))
            cat = e.get("category", "other")
            note = e.get("note", "")
            print(f"{ts} | adjustment | {cat} | {format_money(amount, currency)}")
            if note:
                print(f"  note: {note}")
        else:
            print(f"{ts} | {t} | {e}")
    print("-" * 60)


def entries_in_range(data: Dict[str, Any], start: datetime, end: datetime) -> List[Dict[str, Any]]:
    res = []
    for e in data.get("ledger", []):
        ts = e.get("timestamp")
        if not ts:
            continue
        try:
            dt = parse_iso_dt(ts)
        except Exception:
            continue
        if start <= dt < end:
            res.append(e)
    return res


def print_report(data: Dict[str, Any], start: datetime, end: datetime) -> None:
    data = ensure_schema(data)
    currency = data["profile"]["currency"]

    items = entries_in_range(data, start, end)
    sessions = [e for e in items if e.get("type") == "session"]
    if not sessions:
        print("Нет сессий за выбранный период.")
        return

    total_profit = sum(float(s.get("profit", 0.0)) for s in sessions)
    avg_profit = total_profit / len(sessions)

    # top/bottom sessions by profit
    sessions_sorted = sorted(sessions, key=lambda x: float(x.get("profit", 0.0)))
    worst = sessions_sorted[:3]
    best = list(reversed(sessions_sorted[-3:]))

    print("Report")
    print("-" * 60)
    print(f"From: {start.date().isoformat()} To: {(end - timedelta(seconds=1)).date().isoformat()}")
    print(f"Sessions: {len(sessions)}")
    print(f"Total profit: {format_money(total_profit, currency)}")
    print(f"Average per session: {format_money(avg_profit, currency)}")
    print()

    def fmt_s(e: Dict[str, Any]) -> str:
        ts = e.get("timestamp", "N/A")
        game = e.get("game", "cash")
        lim = e.get("limit") or (f"BI {e.get('buyin')}" if e.get("buyin") is not None else "-")
        prof = float(e.get("profit", 0.0))
        return f"{ts} | {game} | {lim} | {format_money(prof, currency)}"

    print("Top 3 best sessions:")
    for e in best:
        print("  " + fmt_s(e))
    print()
    print("Top 3 worst sessions:")
    for e in worst:
        print("  " + fmt_s(e))
    print("-" * 60)


# --------------------------
# Commands
# --------------------------

def cmd_init(_: argparse.Namespace) -> None:
    path = data_path()
    if os.path.exists(path):
        overwrite = yn_prompt(f"Файл данных уже существует: {path}. Перезаписать?", default=False)
        if not overwrite:
            print("Ок, init отменён.")
            return

    currency = input_choice("Валюта", SUPPORTED_CURRENCIES, default="RUB")
    mode = input_choice("Режим", SUPPORTED_MODES, default="cash")
    strategy = input_choice("Стратегия", SUPPORTED_STRATEGIES, default="standard")
    start_bankroll = input_float("Стартовый банкролл", default=0.0, allow_negative=False)

    profile: Dict[str, Any] = {
        "currency": currency,
        "room": "Pokerdom",
        "mode": mode,
        "strategy": strategy,
        "cash_limits": DEFAULT_CASH_LIMITS[:],
        "cash_rules": CASH_STRATEGIES[strategy].copy(),
        "mtt_rules": MTT_STRATEGIES[strategy].copy(),
    }

    data: Dict[str, Any] = {
        "profile": profile,
        "bankroll": {
            "current": 0.0,  # will be recalculated
            "created_at": now_iso(),
        },
        "ledger": [],
    }

    # Seed with initial adjustment if non-zero
    if abs(start_bankroll) > 1e-9:
        data["ledger"].append({
            "id": str(uuid.uuid4()),
            "type": "adjustment",
            "amount": float(start_bankroll),
            "category": "deposit" if start_bankroll >= 0 else "other",
            "note": "initial bankroll",
            "timestamp": now_iso(),
        })

    ensure_schema(data)
    recalc_bankroll_from_ledger(data)
    save_data(data)

    print(f"Создан файл данных: {path}")
    print("Дальше: python bankroll.py status")


def cmd_status(_: argparse.Namespace) -> None:
    data = ensure_schema(load_data())
    print_status(data, short=False)


def cmd_add_session(args: argparse.Namespace) -> None:
    data = ensure_schema(load_data())
    prof = data["profile"]
    currency = prof["currency"]

    mode = prof["mode"]
    # keep "game" field but align to mode; allow override with --game if user wants
    game = args.game or mode

    if game not in ("cash", "mtt"):
        print("game должен быть cash или mtt.")
        sys.exit(2)

    if game == "cash":
        # limit
        default_limit = None
        last = find_last_session(data)
        if last and last.get("game") == "cash" and last.get("limit"):
            default_limit = str(last.get("limit"))
        raw_limit = input(f"Лимит (например NL10){' [default: '+default_limit+']' if default_limit else ''}: ").strip()
        if not raw_limit and default_limit:
            raw_limit = default_limit
        if not raw_limit:
            print("Лимит обязателен.")
            sys.exit(2)
        limit_val = parse_limit_label(raw_limit)
        if limit_val is None:
            print("Не удалось распознать лимит. Пример: NL10")
            sys.exit(2)
        limit_label = limit_to_label(limit_val)
        buyin = None
    else:
        # buy-in for mtt
        buyin = input_float("Бай-ин турнира (число)", default=None, allow_negative=False)
        limit_label = None

    profit = input_float(f"Профит/лосс за сессию ({currency}, может быть отриц.)", allow_negative=True)
    duration_min = input_int("Длительность (мин), 0 если неизвестно", default=0, allow_negative=False)
    note = input("Заметка (опционально): ").strip()

    entry: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "type": "session",
        "game": game,
        "profit": float(profit),
        "duration_min": int(duration_min) if duration_min > 0 else None,
        "note": note,
        "timestamp": now_iso(),
    }
    if game == "cash":
        entry["limit"] = limit_label
    else:
        entry["buyin"] = float(buyin)

    data["ledger"].append(entry)
    recalc_bankroll_from_ledger(data)
    save_data(data)

    print("\nСессия добавлена.\n")
    print_status(data, short=True)


def cmd_add_adjustment(_: argparse.Namespace) -> None:
    data = ensure_schema(load_data())
    currency = data["profile"]["currency"]

    amount = input_float(f"Сумма корректировки ({currency}, может быть +/-)", allow_negative=True)
    category = input_choice("Категория", ["deposit", "cashout", "bonus", "other"], default="other")
    note = input("Заметка (опционально): ").strip()

    entry = {
        "id": str(uuid.uuid4()),
        "type": "adjustment",
        "amount": float(amount),
        "category": category,
        "note": note,
        "timestamp": now_iso(),
    }

    data["ledger"].append(entry)
    recalc_bankroll_from_ledger(data)
    save_data(data)

    print("\nКорректировка добавлена.\n")
    print_status(data, short=True)


def cmd_history(args: argparse.Namespace) -> None:
    data = ensure_schema(load_data())
    et = args.type
    if et and et not in ("session", "adjustment"):
        print("--type должен быть session или adjustment")
        sys.exit(2)
    print_history(data, n=args.n, entry_type=et)


def resolve_report_range(args: argparse.Namespace) -> Tuple[datetime, datetime]:
    today = date.today()
    # mutually exclusive: period vs from/to
    if args.period:
        p = args.period
        if p == "today":
            start_d = today
            end_d = today + timedelta(days=1)
        elif p == "week":
            start_d = today - timedelta(days=7)
            end_d = today + timedelta(days=1)
        elif p == "month":
            start_d = today - timedelta(days=30)
            end_d = today + timedelta(days=1)
        else:
            raise ValueError("Unknown period")
        return datetime.combine(start_d, datetime.min.time()), datetime.combine(end_d, datetime.min.time())

    # from/to
    if not args.from_date and not args.to_date:
        # default last 7 days including today
        start_d = today - timedelta(days=7)
        end_d = today + timedelta(days=1)
        return datetime.combine(start_d, datetime.min.time()), datetime.combine(end_d, datetime.min.time())

    if not args.from_date or not args.to_date:
        raise ValueError("Укажите оба параметра --from и --to, либо используйте --period.")

    start_d = read_date_yyyy_mm_dd(args.from_date)
    end_d_inclusive = read_date_yyyy_mm_dd(args.to_date)
    end_d = end_d_inclusive + timedelta(days=1)
    if end_d <= datetime.combine(start_d, datetime.min.time()):
        raise ValueError("--to должен быть >= --from")
    return datetime.combine(start_d, datetime.min.time()), datetime.combine(end_d, datetime.min.time())


def cmd_report(args: argparse.Namespace) -> None:
    data = ensure_schema(load_data())
    try:
        start, end = resolve_report_range(args)
    except ValueError as e:
        print(f"Ошибка: {e}")
        sys.exit(2)
    print_report(data, start, end)


def cmd_config(args: argparse.Namespace) -> None:
    data = ensure_schema(load_data())
    prof = data["profile"]

    if args.show:
        print("Config")
        print("-" * 60)
        print(json.dumps(prof, ensure_ascii=False, indent=2))
        return

    # interactive edit
    print("Текущие настройки:")
    print(json.dumps(prof, ensure_ascii=False, indent=2))
    print()

    if yn_prompt("Изменить валюту?", default=False):
        prof["currency"] = input_choice("Валюта", SUPPORTED_CURRENCIES, default=prof["currency"])

    if yn_prompt("Изменить режим (cash/mtt)?", default=False):
        prof["mode"] = input_choice("Режим", SUPPORTED_MODES, default=prof["mode"])

    if yn_prompt("Изменить стратегию?", default=False):
        prof["strategy"] = input_choice("Стратегия", SUPPORTED_STRATEGIES, default=prof["strategy"])
        # update rules to match new strategy (user can customize further later)
        prof["cash_rules"] = CASH_STRATEGIES[prof["strategy"]].copy()
        prof["mtt_rules"] = MTT_STRATEGIES[prof["strategy"]].copy()

    if yn_prompt("Изменить список cash-лимитов?", default=False):
        print("Текущие лимиты:", prof.get("cash_limits", DEFAULT_CASH_LIMITS))
        raw = input("Введите лимиты через запятую (пример: 2,5,10,25,50): ").strip()
        if raw:
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
            new_limits: List[int] = []
            for p in parts:
                try:
                    v = int(p)
                    if v > 0:
                        new_limits.append(v)
                except ValueError:
                    pass
            if new_limits:
                prof["cash_limits"] = sorted(set(new_limits))
            else:
                print("Не удалось распознать лимиты — оставляю без изменений.")

    ensure_schema(data)
    save_data(data)
    print("Настройки сохранены.")


# --------------------------
# CLI
# --------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bankroll.py",
        description="Bankroll Manager (Pokerdom-oriented), single-file CLI tool.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Первичная настройка и создание bankroll_data.json")

    sub.add_parser("status", help="Показать текущий банкролл и рекомендации")

    p_add_s = sub.add_parser("add-session", help="Добавить покерную сессию (profit/loss)")
    p_add_s.add_argument("--game", choices=["cash", "mtt"], help="Переопределить game для записи")

    sub.add_parser("add", help="Добавить ручную корректировку (депозит/кэшаут/бонус/прочее)")

    p_hist = sub.add_parser("history", help="Показать последние записи истории")
    p_hist.add_argument("--n", type=int, default=10, help="Сколько записей показать (default: 10)")
    p_hist.add_argument("--type", type=str, default=None, help="Фильтр: session или adjustment")

    p_rep = sub.add_parser("report", help="Отчёт по сессиям за период")
    p_rep.add_argument("--from", dest="from_date", type=str, help="YYYY-MM-DD")
    p_rep.add_argument("--to", dest="to_date", type=str, help="YYYY-MM-DD (inclusive)")
    p_rep.add_argument("--period", choices=["today", "week", "month"], help="Быстрый период")

    p_cfg = sub.add_parser("config", help="Показать или изменить настройки")
    p_cfg.add_argument("--show", action="store_true", help="Показать конфиг и выйти")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.cmd == "init":
            cmd_init(args)
        elif args.cmd == "status":
            cmd_status(args)
        elif args.cmd == "add-session":
            cmd_add_session(args)
        elif args.cmd == "add":
            cmd_add_adjustment(args)
        elif args.cmd == "history":
            cmd_history(args)
        elif args.cmd == "report":
            cmd_report(args)
        elif args.cmd == "config":
            cmd_config(args)
        else:
            parser.print_help()
            return 2
        return 0
    except FileNotFoundError as e:
        print(str(e))
        return 2
    except KeyboardInterrupt:
        print("\nОтменено пользователем.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())