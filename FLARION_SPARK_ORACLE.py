from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import DataFrame, SparkSession, functions as F

spark = (
    SparkSession.builder.master("local[1]")
    .appName("flarion-regexp-extract-oracle")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("Spark:", spark.version)
print("Java:", spark.sparkContext._jvm.java.lang.System.getProperty("java.version"))


def show(name: str, build: Callable[[], DataFrame]) -> None:
    try:
        rows = build().collect()
        print(f"{name}: {rows!r}")
    except Exception as error:  # Spark wraps Catalyst/JVM errors in Python exceptions.
        message = str(error).splitlines()[0]
        print(f"{name}: ERROR: {message}")


base = spark.range(1)
show(
    "dataframe groups 0/1/2",
    lambda: base.select(
        F.regexp_extract(F.lit("100-200"), r"(\d+)-(\d+)", 0).alias("g0"),
        F.regexp_extract(F.lit("100-200"), r"(\d+)-(\d+)", 1).alias("g1"),
        F.regexp_extract(F.lit("100-200"), r"(\d+)-(\d+)", 2).alias("g2"),
    ),
)
show(
    "catalyst default idx",
    lambda: base.selectExpr("regexp_extract('100-200', '(\\\\d+)-(\\\\d+)') AS value"),
)
show(
    "no match",
    lambda: base.select(F.regexp_extract(F.lit("abc"), r"(\d+)", 1).alias("value")),
)
show(
    "optional group did not participate",
    lambda: base.select(F.regexp_extract(F.lit("aaaac"), r"(a+)(b)?(c)", 2).alias("value")),
)
show(
    "null propagation",
    lambda: base.select(
        F.regexp_extract(F.lit(None).cast("string"), r"(a)", 1).alias("subject_null"),
        F.expr("regexp_extract('a', CAST(NULL AS STRING), 1)").alias("pattern_null"),
        F.expr("regexp_extract('a', '(a)', CAST(NULL AS INT))").alias("idx_null"),
    ),
)
show(
    "row-varying Catalyst pattern and idx",
    lambda: spark.sql(
        r"""
        SELECT regexp_extract(subject, pattern, idx) AS value
        FROM VALUES
          ('100-200', '(\\d+)-(\\d+)', 2),
          ('abc', '([a-z]+)', 1),
          ('aaaac', '(a+)(b)?(c)', 2)
        AS rows(subject, pattern, idx)
        """
    ),
)
show(
    "implicit integer index casts",
    lambda: base.selectExpr(
        "regexp_extract('100-200', '(\\\\d+)-(\\\\d+)', CAST(1 AS BIGINT)) AS i64",
        "regexp_extract('100-200', '(\\\\d+)-(\\\\d+)', CAST(2 AS SMALLINT)) AS i16",
    ),
)
show(
    "implicit string index cast",
    lambda: base.selectExpr(
        "regexp_extract('100-200', '([0-9]+)-([0-9]+)', '1') AS value",
    ),
)
show(
    "implicit atomic string casts",
    lambda: base.selectExpr(
        "regexp_extract(CAST(12345 AS INT), '(\\\\d+)', 1) AS numeric_subject",
        "regexp_extract('12345', CAST(123 AS INT), 0) AS numeric_pattern",
        "regexp_extract(CAST(true AS BOOLEAN), '(true)', 1) AS boolean_subject",
    ),
)
show(
    "invalid idx but no match",
    lambda: base.select(F.regexp_extract(F.lit("abc"), r"(\d+)", 2).alias("value")),
)
show(
    "invalid idx after match",
    lambda: base.select(F.regexp_extract(F.lit("123"), r"(\d+)", 2).alias("value")),
)
show(
    "negative idx after match",
    lambda: base.select(F.regexp_extract(F.lit("123"), r"(\d+)", -1).alias("value")),
)
show(
    "invalid pattern",
    lambda: base.select(F.regexp_extract(F.lit("abc"), "(?l)", 0).alias("value")),
)

spark.stop()
