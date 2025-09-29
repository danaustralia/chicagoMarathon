#!/usr/bin/env python3
"""
Chicago Hourly Weather Extractor (IEM ASOS)
-------------------------------------------
Fetch hourly weather for Chicago dates (e.g., Marathon weekends) and export CSVs.

Data source: IEM ASOS CSV endpoint (asos.py) with tz-localized timestamps.
Docs: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=

Features
- Input: list of dates (YYYY-MM-DD)
- Stations: defaults to KORD (O'Hare) + KMDW (Midway), you can change
- Local timezone selection (default America/Chicago)
- Output: one CSV per date; optional combined CSV
- Alignment: hourly grid with nearest observation within tolerance (default 40min)
- Columns: temperature_C, dewpoint_C, feels_like_C, wind_speed_mph, wind_gust_mph,
           wind_dir_deg, visibility_mi, pressure_inHg, precip_in, condition

Usage examples
--------------
# The three dates you requested (hourly 07:00..18:00 local)
python -u chicago_weather_hourly_multi.py \
  --dates 2023-10-08 2022-10-09 2021-10-10 \
  --start 07:00 --end 18:00 \
  --stations KORD KMDW \
  --tz America/Chicago \
  --out-dir . \
  --combine chicago_weather_hourly_all.csv \
  --verbose
  
  
python -u chicago_weather_hourly_multi.py --dates 2023-10-08 2022-10-09 2021-10-10 --start 07:00 --end 18:00 --stations KORD KMDW --tz America/Chicago --out-dir . --combine chicago_weather_hourly_all.csv --verbose

"""
import argparse
import csv
import io
import math
import sys
from typing import Dict, List, Optional

import pandas as pd
import requests

IEM_HELP = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help="

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChicagoMarathonWeather/1.0; +https://oraylis.de)"
}

DATA_FIELDS = [
    "tmpf", "dwpf",        # °F
    "drct", "sknt", "gust",# deg, knots, mph (gust per IEM CSV)
    "vsby",                # miles
    "alti",                # inHg
    "p01i",                # precip last hour (inches)
    "wxcodes"              # textual
]

def knots_to_mph(kn: float) -> float:
    return float(kn) * 1.150779448 if pd.notna(kn) else kn

def f_to_c(f: float) -> float:
    return (float(f) - 32.0) * 5.0/9.0 if pd.notna(f) else f

def compute_feels_like_c(temp_c: pd.Series, wind_mph: pd.Series) -> pd.Series:
    # Wind chill (simple) for T<=50F and wind>=3 mph; else ambient
    tf = temp_c * 9/5 + 32
    wc_f = 35.74 + 0.6215*tf - 35.75*(wind_mph**0.16) + 0.4275*tf*(wind_mph**0.16)
    cond = (tf <= 50) & (wind_mph >= 3)
    feels_f = tf.where(~cond, wc_f)
    return (feels_f - 32) * 5/9

def build_url(date_str: str, station: str, tz: str) -> str:
    y, m, d = date_str.split("-")
    base = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    parts = [f"data={f}" for f in DATA_FIELDS]
    params = (
        "&".join(parts)
        + f"&station={station}"
        + f"&year1={y}&month1={m}&day1={d}"
        + f"&year2={y}&month2={m}&day2={d}"
        + f"&tz={tz}"
        + "&format=onlycomma&latlon=no&elev=no&missing=M&trace=T"
        + "&direct=no&report_type=3"
    )
    return f"{base}?{params}"

def fetch_day(date_str: str, station: str, tz: str, timeout: int = 60) -> pd.DataFrame:
    url = build_url(date_str, station, tz)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return df
    df["valid"] = pd.to_datetime(df["valid"])  # already localized via tz param
    df = df.set_index("valid").sort_index()

    # Unit conversions
    if "tmpf" in df: df["temperature_C"] = df["tmpf"].astype(float).map(f_to_c)
    if "dwpf" in df: df["dewpoint_C"]    = df["dwpf"].astype(float).map(f_to_c)
    if "sknt" in df: df["wind_speed_mph"] = df["sknt"].astype(float).map(knots_to_mph)
    if "gust" in df: df["wind_gust_mph"] = pd.to_numeric(df["gust"], errors="coerce")
    if "drct" in df: df["wind_dir_deg"]  = pd.to_numeric(df["drct"], errors="coerce")
    if "vsby" in df: df["visibility_mi"] = pd.to_numeric(df["vsby"], errors="coerce")
    if "alti" in df: df["pressure_inHg"] = pd.to_numeric(df["alti"], errors="coerce")
    if "p01i" in df: df["precip_in"]     = pd.to_numeric(df["p01i"], errors="coerce")
    if "wxcodes" in df: df["condition"]  = df["wxcodes"]

    keep = ["temperature_C","dewpoint_C",
            "wind_speed_mph","wind_gust_mph","wind_dir_deg",
            "visibility_mi","pressure_inHg","precip_in","condition"]
    df = df[[c for c in keep if c in df.columns]]
    return df

def align_hourly(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                 tolerance: str = "40min") -> pd.DataFrame:
    # Nearest observation to each hourly mark within tolerance
    idx = pd.date_range(start_ts, end_ts, freq="60min")
    if df.empty:
        return pd.DataFrame(index=idx, columns=["temperature_C","dewpoint_C","feels_like_C",
                                                "wind_speed_mph","wind_gust_mph","wind_dir_deg",
                                                "visibility_mi","pressure_inHg","precip_in","condition"])
    # Ensure numeric cols are numeric (avoid interpolate warnings even though we do nearest)
    df = df.infer_objects(copy=False)
    out = df.reindex(idx, method="nearest", tolerance=pd.Timedelta(tolerance))
    # Compute feels-like if we have inputs
    if {"temperature_C","wind_speed_mph"}.issubset(out.columns):
        out["feels_like_C"] = compute_feels_like_c(out["temperature_C"], out["wind_speed_mph"])
    else:
        out["feels_like_C"] = out.get("temperature_C")
    return out

def combine_stations(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combo = pd.concat(frames, axis=0, copy=False)
    # Average numeric columns per timestamp
    num_cols = combo.select_dtypes(include="number").columns
    out = combo.groupby(level=0)[list(num_cols)].mean()
    # Condition: pick first non-null string per timestamp
    if "condition" in combo.columns:
        conds = (combo[["condition"]]
                 .groupby(level=0)
                 .agg(lambda s: next((x for x in s if isinstance(x, str) and x.strip()), None)))
        out["condition"] = conds["condition"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True, help="One or more dates YYYY-MM-DD")
    ap.add_argument("--start", default="07:00", help="Local start time HH:MM (default 07:00)")
    ap.add_argument("--end",   default="18:00", help="Local end time HH:MM (default 18:00)")
    ap.add_argument("--tz",    default="America/Chicago", help="Local timezone (default America/Chicago)")
    ap.add_argument("--stations", nargs="+", default=["KORD","KMDW"], help="ASOS station IDs")
    ap.add_argument("--out-dir", default=".", help="Directory for per-date CSV files")
    ap.add_argument("--combine", default=None, help="Optional combined CSV filename")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_all: List[pd.DataFrame] = []
    for date_str in args.dates:
        y, m, d = date_str.split("-")
        start_ts = pd.Timestamp(f"{date_str} {args.start}", tz=args.tz)
        end_ts   = pd.Timestamp(f"{date_str} {args.end}", tz=args.tz)
        station_frames = []
        for st in args.stations:
            try:
                df = fetch_day(date_str, st, args.tz)
                aligned = align_hourly(df, start_ts, end_ts)
                station_frames.append(aligned)
                if args.verbose:
                    print(f"[{date_str}] {st}: {aligned.notna().sum().to_dict()}")
            except Exception as e:
                print(f"[warn] {date_str} {st} failed: {e}", file=sys.stderr)
        combined = combine_stations(station_frames)
        combined.index.name = "time_local"
        combined["date"] = date_str
        # Save per-date
        out_path = Path(args.out_dir) / f"chicago_weather_{date_str}_hourly.csv"
        combined.to_csv(out_path)
        if args.verbose:
            print(f"[ok] wrote {out_path}")
        out_all.append(combined)

    if args.combine and out_all:
        combo = pd.concat(out_all).sort_index()
        combo.to_csv(Path(args.out_dir) / args.combine)
        if args.verbose:
            print(f"[ok] wrote combined -> {Path(args.out_dir) / args.combine}")

if __name__ == "__main__":
    main()
