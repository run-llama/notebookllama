import requests
import time
import pandas as pd

from sqlalchemy import Engine, create_engine, Connection, Result
from typing import Optional, Dict, Any, List, Literal, Union, cast


class OtelTracesSqlEngine:
    def __init__(
        self,
        engine: Optional[Engine] = None,
        engine_url: Optional[str] = None,
        table_name: Optional[str] = None,
        service_name: Optional[str] = None,
    ):
        self.service_name: str = service_name or "service"
        self.table_name: str = table_name or "otel_traces"
        self._connection: Optional[Connection] = None
        if engine:
            self._engine: Engine = engine
        elif engine_url:
            self._engine = create_engine(url=engine_url)
        else:
            raise ValueError("One of engine or engine_setup_kwargs must be set")

    def _connect(self) -> None:
        self._connection = self._engine.connect()

    def _export(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = "http://localhost:16686/api/traces"
        params: Dict[str, Union[str, int]] = {
            "service": self.service_name,
            "start": start_time
            or int(time.time() * 1000000) - (24 * 60 * 60 * 1000000),
            "end": end_time or int(time.time() * 1000000),
            "limit": limit or 1000,
        }
        response = requests.get(url, params=params)
        print(response.json())
        return response.json()

    def _to_pandas(self, data: Dict[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        # Loop over each trace
        for trace in data.get("data", []):
            trace_id = trace.get("traceID")
            service_map = {
                pid: proc.get("serviceName")
                for pid, proc in trace.get("processes", {}).items()
            }

            for span in trace.get("spans", []):
                span_id = span.get("spanID")
                operation = span.get("operationName")
                start = span.get("startTime")
                duration = span.get("duration")
                process_id = span.get("processID")
                service = service_map.get(process_id, "")
                status = next(
                    (
                        tag.get("value")
                        for tag in span.get("tags", [])
                        if tag.get("key") == "otel.status_code"
                    ),
                    "",
                )
                parent_span_id = None
                if span.get("references"):
                    parent_span_id = span["references"][0].get("spanID")

                rows.append(
                    {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span_id,
                        "operation_name": operation,
                        "start_time": start,
                        "duration": duration,
                        "status_code": status,
                        "service_name": service,
                    }
                )

        return pd.DataFrame(rows)

    def _to_sql(
        self,
        dataframe: pd.DataFrame,
        if_exists_policy: Optional[Literal["fail", "replace", "append"]] = None,
    ) -> None:
        if not self._connection:
            self._connect()
        dataframe.to_sql(
            name=self.table_name,
            con=self._connection,
            if_exists=if_exists_policy or "append",
        )

    def to_sql_database(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
        if_exists_policy: Optional[Literal["fail", "replace", "append"]] = None,
    ) -> None:
        data = self._export(start_time=start_time, end_time=end_time, limit=limit)
        df = self._to_pandas(data=data)
        self._to_sql(dataframe=df, if_exists_policy=if_exists_policy)

    def execute(
        self,
        statement: Any,
        parameters: Optional[Any] = None,
        execution_options: Optional[Any] = None,
        return_pandas: bool = False,
    ) -> Union[Result, pd.DataFrame]:
        if not self._connection:
            self._connect()
        if not return_pandas:
            self._connection = cast(Connection, self._connection)
            return self._connection.execute(
                statement=statement,
                parameters=parameters,
                execution_options=execution_options,
            )
        return pd.read_sql(sql=statement, con=self._connection)

    def to_pandas(
        self,
    ) -> pd.DataFrame:
        if not self._connection:
            self._connect()
        return pd.read_sql_table(table_name=self.table_name, con=self._connection)

    def to_parquet(
        self,
        output_path: str,
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
        partition_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Export traces to Parquet format for efficient storage and querying.

        Args:
            output_path: Path to save parquet file/directory
            compression: Compression algorithm (default: snappy)
            partition_cols: Columns to partition by (e.g., ['service_name', 'date'])
        """
        try:
            df = self.to_pandas()
        except ValueError:
            df = pd.DataFrame()

        # Ensure start_time is datetime for better downstream usability
        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], unit="us")

        # Handle partition columns
        if partition_cols:
            # Derive date column only if explicitly requested
            if "date" in partition_cols:
                if "start_time" not in df.columns:
                    raise ValueError("Cannot derive 'date' column without 'start_time'")
                df["date"] = df["start_time"].dt.date

            # Validate partition columns
            missing_cols = [col for col in partition_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Invalid partition columns: {missing_cols}. "
                    f"Available columns: {list(df.columns)}"
                )

        df.to_parquet(
            output_path,
            compression=compression,
            partition_cols=partition_cols,
            index=False,
        )

    def disconnect(self) -> None:
        if not self._connection:
            raise ValueError("Engine was never connected!")
        self._engine.dispose(close=True)
