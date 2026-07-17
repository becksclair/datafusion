// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Int16Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::display::array_value_to_string;
use datafusion::prelude::SessionContext;
use datafusion_common::Result;
use datafusion_expr::{col, lit};
use datafusion_spark::expr_fn::regexp_extract;

#[tokio::test]
async fn regexp_extract_executes_through_dataframe_api() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("subject", DataType::Utf8, false),
        Field::new("pattern", DataType::Utf8, false),
        Field::new("idx64", DataType::Int64, false),
        Field::new("idx16", DataType::Int16, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["100-200", "abc", "aaaac"])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                r"(\d+)-(\d+)",
                "([a-z]+)",
                "(a+)(b)?(c)",
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![2, 1, 2])) as ArrayRef,
            Arc::new(Int16Array::from(vec![2, 1, 2])) as ArrayRef,
        ],
    )?;

    let result = SessionContext::new()
        .read_batch(batch)?
        .select(vec![
            regexp_extract(col("subject"), col("pattern"), Some(col("idx64")))
                .alias("from_i64"),
            regexp_extract(col("subject"), col("pattern"), Some(col("idx16")))
                .alias("from_i16"),
        ])?
        .collect()
        .await?;

    for column in 0..2 {
        let values = result[0].column(column).as_string::<i32>();
        assert_eq!(
            values.iter().collect::<Vec<_>>(),
            vec![Some("200"), Some("abc"), Some("")]
        );
    }
    Ok(())
}

#[tokio::test]
async fn regexp_extract_implicitly_casts_atomic_string_arguments() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("subject", DataType::Int32, false),
        Field::new("pattern", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![12345])) as ArrayRef,
            Arc::new(Int32Array::from(vec![123])) as ArrayRef,
        ],
    )?;

    let result = SessionContext::new()
        .read_batch(batch)?
        .select(vec![
            regexp_extract(col("subject"), col("pattern"), Some(lit(0))).alias("value"),
        ])?
        .collect()
        .await?;

    assert_eq!(array_value_to_string(result[0].column(0), 0)?, "123");
    Ok(())
}

#[tokio::test]
async fn regexp_extract_accepts_scalar_pattern_and_index_arguments() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "subject",
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(vec!["100-200", "300-400"])) as ArrayRef],
    )?;

    let result = SessionContext::new()
        .read_batch(batch)?
        .select(vec![
            regexp_extract(col("subject"), lit(r"(\d+)-(\d+)"), Some(lit("1")))
                .alias("from_string_idx"),
            regexp_extract(col("subject"), lit(r"(\d+)-(\d+)"), Some(lit(2)))
                .alias("from_int_idx"),
        ])?
        .collect()
        .await?;

    assert_eq!(
        result[0]
            .column(0)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some("100"), Some("300")]
    );
    assert_eq!(
        result[0]
            .column(1)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some("200"), Some("400")]
    );
    Ok(())
}
