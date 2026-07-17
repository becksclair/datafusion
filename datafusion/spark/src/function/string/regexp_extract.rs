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

use arrow::array::{
    Array, ArrayRef, AsArray, GenericStringBuilder, OffsetSizeTrait, StringViewBuilder,
};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::cast::as_int32_array;
use datafusion_common::types::{
    NativeType, logical_boolean, logical_date, logical_int32, logical_string,
};
use datafusion_common::{DataFusionError, Result, exec_err};
use datafusion_expr::function::Hint;
use datafusion_expr::{
    Coercion, ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use regex::{CaptureLocations, Regex};

/// Spark-compatible `regexp_extract` expression.
///
/// Returns group `idx` from the first regex match. Group zero is the complete
/// match; when `idx` is omitted Spark defaults it to one. A missing match or an
/// optional group that did not participate yields an empty string, while a NULL
/// input yields NULL.
///
/// Spark's exact pattern contract is `java.util.regex.Pattern`. This standalone
/// native path intentionally uses Rust `regex` for predictable linear-time
/// matching, so it does not claim full Java-regex compatibility. Apache
/// DataFusion Comet uses the same dual-engine boundary: Spark/JVM `doGenCode`
/// dispatch is the default exact path, while native Rust regex is an explicitly
/// incompatible performance path.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#regexp_extract>
/// <https://datafusion.apache.org/comet/user-guide/latest/compatibility/regex.html>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkRegexpExtract {
    signature: Signature,
}

impl Default for SparkRegexpExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkRegexpExtract {
    pub fn new() -> Self {
        let string = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_string()),
            vec![
                TypeSignatureClass::Numeric,
                TypeSignatureClass::Native(logical_boolean()),
                TypeSignatureClass::Native(logical_date()),
                TypeSignatureClass::Timestamp,
                TypeSignatureClass::Time,
                TypeSignatureClass::Interval,
                TypeSignatureClass::Duration,
                TypeSignatureClass::Binary,
            ],
            NativeType::String,
        );
        let int32 = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_int32()),
            vec![
                TypeSignatureClass::Integer,
                TypeSignatureClass::Native(logical_string()),
            ],
            NativeType::Int32,
        );

        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Coercible(vec![string.clone(), string.clone()]),
                    TypeSignature::Coercible(vec![string.clone(), string, int32]),
                ],
                Volatility::Immutable,
            )
            .with_parameter_names(vec![
                "str".to_string(),
                "regexp".to_string(),
                "idx".to_string(),
            ])
            .expect("valid parameter names"),
        }
    }
}

impl ScalarUDFImpl for SparkRegexpExtract {
    fn name(&self) -> &str {
        "regexp_extract"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion_common::internal_err!(
            "return_type should not be called for Spark regexp_extract"
        )
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|field| field.is_nullable());
        Ok(Arc::new(Field::new(
            self.name(),
            args.arg_fields[0].data_type().clone(),
            nullable,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(
            spark_regexp_extract,
            vec![Hint::Pad, Hint::AcceptsSingular, Hint::AcceptsSingular],
        )(&args.args)
    }
}

fn spark_regexp_extract(args: &[ArrayRef]) -> Result<ArrayRef> {
    if !matches!(args.len(), 2 | 3) {
        return exec_err!(
            "regexp_extract requires two or three arguments, got {}",
            args.len()
        );
    }

    match args[0].data_type() {
        DataType::Utf8 => regexp_extract_with_builder(
            args,
            GenericStringBuilder::<i32>::with_capacity(args[0].len(), 0),
        ),
        DataType::LargeUtf8 => regexp_extract_with_builder(
            args,
            GenericStringBuilder::<i64>::with_capacity(args[0].len(), 0),
        ),
        DataType::Utf8View => regexp_extract_with_builder(
            args,
            StringViewBuilder::with_capacity(args[0].len()),
        ),
        other => exec_err!(
            "regexp_extract does not support subject type {other:?}; expected Utf8, LargeUtf8, or Utf8View"
        ),
    }
}

fn regexp_extract_with_builder<B>(args: &[ArrayRef], mut builder: B) -> Result<ArrayRef>
where
    B: RegexpExtractBuilder,
{
    let idx_array = if args.len() == 3 {
        Some(as_int32_array(&args[2])?)
    } else {
        None
    };

    // Spark caches the last observed pattern in the expression. Mirroring that
    // policy avoids recompilation for the common literal / run-length case
    // without retaining an unbounded cache for a high-cardinality pattern column.
    let mut last_pattern = None::<&str>;
    let mut last_regex = None::<Regex>;
    let mut last_locations = None::<CaptureLocations>;

    for row in 0..args[0].len() {
        let Some(subject) = string_at(&args[0], row)? else {
            builder.append_null();
            continue;
        };
        let Some(pattern) = string_at(&args[1], row)? else {
            builder.append_null();
            continue;
        };
        let idx = match idx_array {
            Some(array) if array.is_null(singular_or_row(array.len(), row)) => {
                builder.append_null();
                continue;
            }
            Some(array) => array.value(singular_or_row(array.len(), row)),
            None => 1,
        };

        if last_pattern != Some(pattern) {
            let compiled = Regex::new(pattern).map_err(|error| {
                DataFusionError::Execution(format!(
                    "Invalid regexp pattern for regexp_extract: {pattern:?}: {error}"
                ))
            })?;
            last_locations = Some(compiled.capture_locations());
            last_pattern = Some(pattern);
            last_regex = Some(compiled);
        }

        let regex = last_regex
            .as_ref()
            .expect("a regex is compiled before it is used");

        if idx == 0 {
            builder
                .append_value(regex.find(subject).map_or("", |matched| matched.as_str()));
            continue;
        }

        let locations = last_locations
            .as_mut()
            .expect("capture locations are created with the regex");
        let Some(_) = regex.captures_read(locations, subject) else {
            // Spark does not validate idx until a match exists.
            builder.append_value("");
            continue;
        };

        let group_count = regex.captures_len() - 1;
        let group_index = validate_group_index(idx, group_count)?;
        builder.append_value(
            locations
                .get(group_index)
                .map(|(start, end)| &subject[start..end])
                .unwrap_or(""),
        );
    }

    Ok(builder.finish())
}

fn validate_group_index(group_index: i32, group_count: usize) -> Result<usize> {
    let Ok(group_index_usize) = usize::try_from(group_index) else {
        return exec_err!(
            "Invalid regexp group index {group_index} for regexp_extract: pattern has {group_count} capturing groups"
        );
    };

    if group_index_usize > group_count {
        return exec_err!(
            "Invalid regexp group index {group_index} for regexp_extract: pattern has {group_count} capturing groups"
        );
    }

    Ok(group_index_usize)
}

fn string_at(array: &ArrayRef, row: usize) -> Result<Option<&str>> {
    let row = singular_or_row(array.len(), row);
    match array.data_type() {
        DataType::Utf8 => {
            let array = array.as_string::<i32>();
            Ok((!array.is_null(row)).then(|| array.value(row)))
        }
        DataType::LargeUtf8 => {
            let array = array.as_string::<i64>();
            Ok((!array.is_null(row)).then(|| array.value(row)))
        }
        DataType::Utf8View => {
            let array = array.as_string_view();
            Ok((!array.is_null(row)).then(|| array.value(row)))
        }
        other => exec_err!(
            "regexp_extract does not support pattern type {other:?}; expected Utf8, LargeUtf8, or Utf8View"
        ),
    }
}

fn singular_or_row(array_len: usize, row: usize) -> usize {
    if array_len == 1 { 0 } else { row }
}

trait RegexpExtractBuilder {
    fn append_value(&mut self, value: &str);
    fn append_null(&mut self);
    fn finish(&mut self) -> ArrayRef;
}

impl<O: OffsetSizeTrait> RegexpExtractBuilder for GenericStringBuilder<O> {
    fn append_value(&mut self, value: &str) {
        GenericStringBuilder::append_value(self, value);
    }

    fn append_null(&mut self) {
        GenericStringBuilder::append_null(self);
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(GenericStringBuilder::finish(self))
    }
}

impl RegexpExtractBuilder for StringViewBuilder {
    fn append_value(&mut self, value: &str) {
        StringViewBuilder::append_value(self, value);
    }

    fn append_null(&mut self) {
        StringViewBuilder::append_null(self);
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(StringViewBuilder::finish(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::utils::test::test_scalar_function;
    use arrow::array::{Int32Array, LargeStringArray, StringArray, StringViewArray};
    use datafusion_common::ScalarValue;
    use datafusion_expr::{Expr, col, lit};

    macro_rules! test_regexp_extract_string_types {
        ($subject:expr, $pattern:expr, $idx:expr, $expected:expr) => {{
            test_scalar_function!(
                SparkRegexpExtract::new(),
                vec![
                    ColumnarValue::Scalar(ScalarValue::Utf8(
                        $subject.map(str::to_string)
                    )),
                    ColumnarValue::Scalar(ScalarValue::Utf8(
                        $pattern.map(str::to_string)
                    )),
                    ColumnarValue::Scalar(ScalarValue::Int32($idx)),
                ],
                Ok($expected.map(str::to_string)),
                String,
                DataType::Utf8,
                StringArray
            );
            test_scalar_function!(
                SparkRegexpExtract::new(),
                vec![
                    ColumnarValue::Scalar(ScalarValue::LargeUtf8(
                        $subject.map(str::to_string),
                    )),
                    ColumnarValue::Scalar(ScalarValue::LargeUtf8(
                        $pattern.map(str::to_string),
                    )),
                    ColumnarValue::Scalar(ScalarValue::Int32($idx)),
                ],
                Ok($expected.map(str::to_string)),
                String,
                DataType::LargeUtf8,
                LargeStringArray
            );
            test_scalar_function!(
                SparkRegexpExtract::new(),
                vec![
                    ColumnarValue::Scalar(ScalarValue::Utf8View(
                        $subject.map(str::to_string),
                    )),
                    ColumnarValue::Scalar(ScalarValue::Utf8View(
                        $pattern.map(str::to_string),
                    )),
                    ColumnarValue::Scalar(ScalarValue::Int32($idx)),
                ],
                Ok($expected.map(str::to_string)),
                String,
                DataType::Utf8View,
                StringViewArray
            );
        }};
    }

    #[test]
    fn extracts_first_match_and_groups() {
        test_regexp_extract_string_types!(
            Some("100-200"),
            Some(r"(\d+)-(\d+)"),
            Some(0),
            Some("100-200")
        );
        test_regexp_extract_string_types!(
            Some("100-200"),
            Some(r"(\d+)-(\d+)"),
            Some(1),
            Some("100")
        );
        test_regexp_extract_string_types!(
            Some("100-200"),
            Some(r"(\d+)-(\d+)"),
            Some(2),
            Some("200")
        );
        test_regexp_extract_string_types!(
            Some("abc123def456"),
            Some(r"(\d+)"),
            Some(1),
            Some("123")
        );
    }

    #[test]
    fn defaults_to_group_one() {
        test_scalar_function!(
            SparkRegexpExtract::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("100-200".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                    r"(\d+)-(\d+)".to_string(),
                ))),
            ],
            Ok(Some("100".to_string())),
            String,
            DataType::Utf8,
            StringArray
        );
    }

    #[test]
    fn no_match_and_unmatched_optional_group_are_empty() {
        test_regexp_extract_string_types!(
            Some("100-200"),
            Some("([a-z]+)"),
            Some(1),
            Some("")
        );
        test_regexp_extract_string_types!(
            Some("aaaac"),
            Some("(a+)(b)?(c)"),
            Some(2),
            Some("")
        );
    }

    #[test]
    fn nulls_propagate() {
        test_regexp_extract_string_types!(None, Some("(a)"), Some(1), None);
        test_regexp_extract_string_types!(Some("a"), None, Some(1), None);
        test_regexp_extract_string_types!(Some("a"), Some("(a)"), None, None);
    }

    #[test]
    fn supports_row_varying_pattern_and_index() -> Result<()> {
        let args = vec![
            Arc::new(StringArray::from(vec!["100-200", "abc", "aaaac"])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                r"(\d+)-(\d+)",
                "([a-z]+)",
                "(a+)(b)?(c)",
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![2, 1, 2])) as ArrayRef,
        ];

        let actual = spark_regexp_extract(&args)?;
        let actual = actual.as_string::<i32>();
        assert_eq!(
            actual.iter().collect::<Vec<_>>(),
            vec![Some("200"), Some("abc"), Some("")]
        );
        Ok(())
    }

    #[test]
    fn invalid_group_is_checked_only_after_a_match() -> Result<()> {
        let no_match = spark_regexp_extract(&[
            Arc::new(StringArray::from(vec!["abc"])) as ArrayRef,
            Arc::new(StringArray::from(vec![r"(\d+)"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![2])) as ArrayRef,
        ])?;
        assert_eq!(no_match.as_string::<i32>().value(0), "");

        for invalid_index in [2, -1] {
            let error = spark_regexp_extract(&[
                Arc::new(StringArray::from(vec!["123"])) as ArrayRef,
                Arc::new(StringArray::from(vec![r"(\d+)"])) as ArrayRef,
                Arc::new(Int32Array::from(vec![invalid_index])) as ArrayRef,
            ])
            .unwrap_err();
            assert!(error.to_string().contains("pattern has 1 capturing groups"));
        }
        Ok(())
    }

    #[test]
    fn registered_in_default_spark_functions() {
        assert!(
            crate::all_default_scalar_functions()
                .iter()
                .any(|function| function.name() == "regexp_extract")
        );
    }

    #[test]
    fn dataframe_expression_api_supports_optional_index() {
        for (idx, expected_arity) in [(None, 2), (Some(lit(2)), 3)] {
            let expression =
                crate::expr_fn::regexp_extract(col("subject"), col("pattern"), idx);
            let Expr::ScalarFunction(function) = expression else {
                panic!("regexp_extract expression should be a scalar function");
            };
            assert_eq!(function.name(), "regexp_extract");
            assert_eq!(function.args.len(), expected_arity);
        }
    }

    #[test]
    fn invalid_pattern_is_an_error() {
        let error = spark_regexp_extract(&[
            Arc::new(StringArray::from(vec!["abc"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["(?l)"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![0])) as ArrayRef,
        ])
        .unwrap_err();
        assert!(error.to_string().contains("Invalid regexp pattern"));
    }
}
