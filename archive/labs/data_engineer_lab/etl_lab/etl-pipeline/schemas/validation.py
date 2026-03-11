"""Pandera schema definitions for each pipeline stage."""

import pandera as pa
from pandera import Column, Check

# ── Raw Schema ────────────────────────────────────────────────────────

RawTransactionSchema = pa.DataFrameSchema(
    columns={
        "Time": Column(float, Check.ge(0)),
        "Amount": Column(float, Check.ge(0)),
        "Class": Column(int, Check.isin([0, 1])),
        # V1-V28 PCA features: float, nullable
        **{f"V{i}": Column(float, nullable=True) for i in range(1, 29)},
    },
    strict=False,
    coerce=True,
)

# ── Prepared Schema (after cleaning + feature engineering) ────────────

PreparedSchema = pa.DataFrameSchema(
    columns={
        "time_seconds": Column(float, Check.ge(0)),
        "amount": Column(float, Check.ge(0)),
        "is_fraud": Column(int, Check.isin([0, 1])),
        "hour_of_day": Column(int, Check.in_range(0, 23)),
        "amount_bin": Column(str),
        "log_amount": Column(float),
    },
    strict=False,
    coerce=True,
)
