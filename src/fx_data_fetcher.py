"""FX daily data fetcher supporting Stooq and FRED.

This module provides a small class to download daily FX rates for USDJPY and
EURUSD from free data sources and save them as CSV files. Missing values are
dropped, and the returned DataFrames are sorted by date.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime]


@dataclass(frozen=True)
class FXDataFetcher:
    """Fetch daily FX rates from Stooq or FRED.

    Parameters
    ----------
    source : Literal["stooq", "fred"]
        Data source to use.
    data_dir : Path
        Directory where CSV outputs will be saved.
    fred_api_key : Optional[str]
        API key required for FRED. Not needed for Stooq.
    timeout : int
        Timeout in seconds for HTTP requests.
    """

    source: Literal["stooq", "fred"]
    data_dir: Path
    fred_api_key: Optional[str] = None
    timeout: int = 30

    _FRED_SERIES = {
        "USDJPY": "DEXJPUS",
        "EURUSD": "DEXUSEU",
    }
    _STOOQ_BASE = "https://stooq.pl/q/d/l/"

    def __post_init__(self) -> None:
        normalized_source = self.source.lower()
        if normalized_source not in ("stooq", "fred"):
            raise ValueError("source must be 'stooq' or 'fred'")
        object.__setattr__(self, "source", normalized_source)

        resolved_data_dir = self.data_dir if isinstance(self.data_dir, Path) else Path(self.data_dir)
        resolved_data_dir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "data_dir", resolved_data_dir.resolve())

        if self.source == "fred" and not self.fred_api_key:
            raise ValueError("FRED source requires fred_api_key")

    def fetch_pair(
        self,
        pair: str,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
    ) -> pd.DataFrame:
        """Fetch FX data for a given pair.

        Parameters
        ----------
        pair : str
            Currency pair, e.g., "USDJPY" or "EURUSD".
        start_date : Optional[DateLike]
            Inclusive start date. If None, the source default is used.
        end_date : Optional[DateLike]
            Inclusive end date. If None, the source default is used.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ["date", "close"], sorted by date, with missing rows dropped.
        """

        pair_key = pair.strip().upper()
        if self.source == "stooq":
            df = self._fetch_from_stooq(pair_key)
        else:
            df = self._fetch_from_fred(pair_key, start_date, end_date)

        df = self._filter_by_dates(df, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data returned for {pair_key} from {self.source}")
        return df

    def fetch_and_save(
        self,
        pair: str,
        output_path: Optional[Path] = None,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
    ) -> Path:
        """Fetch FX data and save to CSV.

        Parameters
        ----------
        pair : str
            Currency pair to fetch.
        output_path : Optional[Path]
            Destination CSV path. Defaults to `<data_dir>/<pair>.csv` (lowercase).
        start_date : Optional[DateLike]
            Inclusive start date.
        end_date : Optional[DateLike]
            Inclusive end date.

        Returns
        -------
        Path
            Path to the written CSV file.
        """

        df = self.fetch_pair(pair=pair, start_date=start_date, end_date=end_date)
        destination = output_path or (self.data_dir / f"{pair.lower()}.csv")
        self.save_csv(df, destination)
        return destination

    def save_csv(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save a DataFrame to CSV, sorted by date."""
        output_path = output_path if isinstance(output_path, Path) else Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_df = df.copy()
        sorted_df = sorted_df.sort_values("date")
        sorted_df.to_csv(output_path, index=False)
        logger.info("Saved %d rows to %s", len(sorted_df), output_path)

    def _fetch_from_stooq(self, pair: str) -> pd.DataFrame:
        symbol = pair.lower()
        params = {"s": symbol, "i": "d"}
        response = requests.get(self._STOOQ_BASE, params=params, timeout=self.timeout)
        response.raise_for_status()
        logger.info("Fetched Stooq data for %s", pair)

        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            raise ValueError(f"Stooq returned no data for {pair}")

        normalized_columns = {col: str(col).replace("\ufeff", "").strip() for col in df.columns}
        df = df.rename(columns=normalized_columns)

        lowered_to_original = {str(col).casefold(): col for col in df.columns}
        stooq_column_aliases = {
            "date": ["date", "data"],
            "close": ["close", "zamkniecie"],
        }

        rename_map: dict[str, str] = {}
        for target, aliases in stooq_column_aliases.items():
            matched_column = next(
                (lowered_to_original[alias.casefold()] for alias in aliases if alias.casefold() in lowered_to_original),
                None,
            )
            if matched_column is not None:
                rename_map[matched_column] = target

        df = df.rename(columns=rename_map)
        missing_columns = [col for col in ("date", "close") if col not in df.columns]
        if missing_columns:
            available = ", ".join(map(str, df.columns))
            missing = ", ".join(missing_columns)
            raise ValueError(
                f"Stooq schema mismatch for {pair}: missing columns [{missing}] (available: [{available}])"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
        return df[["date", "close"]]

    def _fetch_from_fred(
        self,
        pair: str,
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> pd.DataFrame:
        series_id = self._FRED_SERIES.get(pair)
        if not series_id:
            raise ValueError(f"Unsupported pair for FRED: {pair}")

        params = {
            "file_type": "json",
            "series_id": series_id,
            "api_key": self.fred_api_key,
        }
        if start_date is not None:
            params["observation_start"] = self._to_iso_date(start_date)
        if end_date is not None:
            params["observation_end"] = self._to_iso_date(end_date)

        url = "https://api.stlouisfed.org/fred/series/observations"
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        observations = payload.get("observations", [])
        if not observations:
            raise ValueError(f"FRED returned no data for {pair}")

        records = []
        for obs in observations:
            value = obs.get("value")
            value = None if value in (".", None) else value
            records.append({"date": obs.get("date"), "close": value})

        df = pd.DataFrame.from_records(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
        return df[["date", "close"]]

    @staticmethod
    def _to_iso_date(value: DateLike) -> str:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid date value: {value}")
        return parsed.date().isoformat()

    @staticmethod
    def _filter_by_dates(
        df: pd.DataFrame,
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> pd.DataFrame:
        result = df.copy()
        if start_date is not None:
            start_ts = pd.to_datetime(start_date, errors="coerce")
            if pd.isna(start_ts):
                raise ValueError(f"Invalid start_date: {start_date}")
            result = result[result["date"] >= start_ts]
        if end_date is not None:
            end_ts = pd.to_datetime(end_date, errors="coerce")
            if pd.isna(end_ts):
                raise ValueError(f"Invalid end_date: {end_date}")
            result = result[result["date"] <= end_ts]
        return result.reset_index(drop=True)
