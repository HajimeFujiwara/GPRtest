"""Stooq/FRED から日次FXデータを取得するユーティリティ。

USDJPY と EURUSD を対象に、データ取得・整形・CSV保存までを
一貫して提供する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime]

SUPPORTED_PAIRS = ("USDJPY", "EURUSD")
COMMON_REQUIRED_COLUMNS = ("date", "close")


@dataclass(frozen=True)
class FXDataFetcher:
    """日次FXレートを Stooq または FRED から取得する。

    Parameters
    ----------
    source : Literal["stooq", "fred"]
        取得元データソース。
    data_dir : Path
        CSV保存先ディレクトリ。
    fred_api_key : Optional[str]
        FRED利用時のAPIキー（Stooq利用時は不要）。
    timeout : int
        HTTPリクエストのタイムアウト秒。
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

        resolved_data_dir = self.data_dir
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
        """指定した通貨ペアのFXデータを取得する。

        Parameters
        ----------
        pair : str
            通貨ペア（例: "USDJPY", "EURUSD"）。
        start_date : Optional[DateLike]
            取得開始日（含む）。None の場合はソース既定。
        end_date : Optional[DateLike]
            取得終了日（含む）。None の場合はソース既定。

        Returns
        -------
        pd.DataFrame
            ["date", "close"] 列を持つDataFrame。
            日付昇順・欠損行除去済み。
        """

        pair_key = self._normalize_pair(pair)
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
        """FXデータを取得してCSV保存する。

        Parameters
        ----------
        pair : str
            取得する通貨ペア。
        output_path : Optional[Path]
            出力先CSVパス。Noneなら `<data_dir>/<pair>.csv`（小文字）。
        start_date : Optional[DateLike]
            取得開始日（含む）。
        end_date : Optional[DateLike]
            取得終了日（含む）。

        Returns
        -------
        Path
            書き出したCSVファイルパス。
        """

        df = self.fetch_pair(pair=pair, start_date=start_date, end_date=end_date)
        pair_key = self._normalize_pair(pair)
        destination = output_path or (self.data_dir / f"{pair_key.lower()}.csv")
        self.save_csv(df, destination)
        return destination

    def save_csv(self, df: pd.DataFrame, output_path: Path) -> None:
        """DataFrameを日付昇順でCSV保存する。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_df = df.copy()
        sorted_df = sorted_df.sort_values("date")
        sorted_df.to_csv(output_path, index=False)
        logger.info("Saved %d rows to %s", len(sorted_df), output_path)

    def _fetch_from_stooq(self, pair: str) -> pd.DataFrame:
        """Stooqから通貨ペアデータを取得し標準形式に整形する。"""
        symbol = pair.lower()
        params = {"s": symbol, "i": "d"}
        response = requests.get(self._STOOQ_BASE, params=params, timeout=self.timeout)
        response.raise_for_status()
        logger.info("Fetched Stooq data for %s", pair)

        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            raise ValueError(f"Stooq returned no data for {pair}")

        aliases = {
            "date": ("date", "data"),
            "close": ("close", "zamkniecie"),
        }
        normalized = self._rename_columns_by_alias(df, aliases)
        self._ensure_required_columns(normalized, COMMON_REQUIRED_COLUMNS, source_name="Stooq", pair=pair)
        return self._coerce_and_clean_common_columns(normalized)

    def _fetch_from_fred(
        self,
        pair: str,
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> pd.DataFrame:
        """FREDから通貨ペアデータを取得し標準形式に整形する。"""
        series_id = self._FRED_SERIES.get(pair)
        if not series_id:
            raise ValueError(f"Unsupported pair for FRED: {pair}")

        params: dict[str, str] = {
            "file_type": "json",
            "series_id": series_id,
            "api_key": self.fred_api_key or "",
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

        records: list[dict[str, Any]] = []
        for obs in observations:
            value = obs.get("value")
            value = None if value in (".", None) else value
            records.append({"date": obs.get("date"), "close": value})

        df = pd.DataFrame.from_records(records)
        self._ensure_required_columns(df, COMMON_REQUIRED_COLUMNS, source_name="FRED", pair=pair)
        return self._coerce_and_clean_common_columns(df)

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        """通貨ペア文字列を正規化し、対応可否を検証する。"""
        pair_key = pair.strip().upper()
        if pair_key not in SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported pair: {pair_key}. supported={SUPPORTED_PAIRS}")
        return pair_key

    @staticmethod
    def _rename_columns_by_alias(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
        """列名を別名辞書で標準列名に寄せる。"""
        normalized_columns = {col: str(col).replace("\ufeff", "").strip() for col in df.columns}
        out = df.rename(columns=normalized_columns).copy()

        lower_to_original = {str(col).casefold(): col for col in out.columns}
        rename_map: dict[str, str] = {}
        for target, candidates in aliases.items():
            match = next((lower_to_original[c.casefold()] for c in candidates if c.casefold() in lower_to_original), None)
            if match is not None:
                rename_map[match] = target

        return out.rename(columns=rename_map)

    @staticmethod
    def _ensure_required_columns(
        df: pd.DataFrame,
        required_columns: tuple[str, ...],
        source_name: str,
        pair: str,
    ) -> None:
        """必須列の存在を検証し、欠落時は原因が分かる例外を送出する。"""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if not missing_columns:
            return

        available = ", ".join(map(str, df.columns))
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"{source_name} schema mismatch for {pair}: missing columns [{missing}] (available: [{available}])"
        )

    @staticmethod
    def _coerce_and_clean_common_columns(df: pd.DataFrame) -> pd.DataFrame:
        """date/close列の型変換と欠損除去を共通処理として実施する。"""
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False)
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out = out.dropna(subset=["date", "close"]).reset_index(drop=True)
        return out[["date", "close"]]

    @staticmethod
    def _to_iso_date(value: DateLike) -> str:
        """日付入力を YYYY-MM-DD 形式に変換する。"""
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
        """開始日・終了日でDataFrameをフィルタする。"""
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
