import pandas as pd
import pytest

from src.notebookllama.instrumentation import OtelTracesSqlEngine


def _create_sample_df():
    return pd.DataFrame(
        {
            "trace_id": ["abc"],
            "span_id": ["span1"],
            "parent_span_id": [None],
            "operation_name": ["op"],
            "start_time": [1700000000000000],  # microseconds
            "duration": [100],
            "status_code": ["OK"],
            "service_name": ["svc"],
        }
    )


def test_to_parquet_creates_file(tmp_path):
    engine = OtelTracesSqlEngine(engine_url="sqlite:///:memory:")
    engine._connect()

    df = _create_sample_df()
    engine._to_sql(df, if_exists_policy="replace")

    output_file = tmp_path / "traces.parquet"
    engine.to_parquet(str(output_file))

    assert output_file.exists()


def test_to_parquet_writes_correct_content(tmp_path):
    engine = OtelTracesSqlEngine(engine_url="sqlite:///:memory:")
    engine._connect()

    df = _create_sample_df()
    engine._to_sql(df, if_exists_policy="replace")

    output_file = tmp_path / "traces.parquet"
    engine.to_parquet(str(output_file))

    exported_df = pd.read_parquet(output_file)

    assert len(exported_df) == 1
    assert exported_df["trace_id"].iloc[0] == "abc"
    assert exported_df["service_name"].iloc[0] == "svc"


def test_to_parquet_with_partitioning(tmp_path):
    engine = OtelTracesSqlEngine(engine_url="sqlite:///:memory:")
    engine._connect()

    df = _create_sample_df()
    engine._to_sql(df, if_exists_policy="replace")

    output_dir = tmp_path / "partitioned"

    engine.to_parquet(
        str(output_dir),
        partition_cols=["service_name", "date"],
    )

    # derive expected date from start_time
    expected_date = pd.to_datetime(1700000000000000, unit="us").date()

    partition_path = output_dir / "service_name=svc" / f"date={expected_date}"

    assert partition_path.exists()
    assert any(partition_path.glob("*.parquet"))


def test_to_parquet_empty_table(tmp_path):
    engine = OtelTracesSqlEngine(engine_url="sqlite:///:memory:")
    engine._connect()

    # Create empty table explicitly
    empty_df = pd.DataFrame(
        columns=[
            "trace_id",
            "span_id",
            "parent_span_id",
            "operation_name",
            "start_time",
            "duration",
            "status_code",
            "service_name",
        ]
    )

    engine._to_sql(empty_df, if_exists_policy="replace")

    output_file = tmp_path / "empty.parquet"
    engine.to_parquet(str(output_file))

    assert output_file.exists()

    df = pd.read_parquet(output_file)
    assert df.empty


def test_to_parquet_invalid_partition_column(tmp_path):
    engine = OtelTracesSqlEngine(engine_url="sqlite:///:memory:")
    engine._connect()

    df = _create_sample_df()
    engine._to_sql(df, if_exists_policy="replace")

    with pytest.raises(ValueError):
        engine.to_parquet(
            str(tmp_path / "bad.parquet"),
            partition_cols=["does_not_exist"],
        )


def test_to_parquet_with_compression(tmp_path):
    engine = OtelTracesSqlEngine(engine_url="sqlite:///:memory:")
    engine._connect()

    df = _create_sample_df()
    engine._to_sql(df, if_exists_policy="replace")

    output_file = tmp_path / "compressed.parquet"

    engine.to_parquet(
        str(output_file),
        compression="gzip",
    )

    assert output_file.exists()
