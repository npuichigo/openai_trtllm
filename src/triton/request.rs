#![allow(dead_code)]

use super::{InferTensorContents, ModelInferRequest};
use crate::triton::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

pub(crate) struct Builder {
    inner: anyhow::Result<ModelInferRequest>,
}

impl Builder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn build(self) -> anyhow::Result<ModelInferRequest> {
        self.inner
    }

    pub(crate) fn model_name<S>(self, model_name: S) -> Self
    where
        S: Into<String>,
    {
        self.and_then(|mut request| {
            request.model_name = model_name.into();
            Ok(request)
        })
    }

    fn model_version<S>(self, model_version: S) -> Self
    where
        S: Into<String>,
    {
        self.and_then(|mut request| {
            request.model_version = model_version.into();
            Ok(request)
        })
    }

    fn id<S>(self, id: S) -> Self
    where
        S: Into<String>,
    {
        self.and_then(|mut request| {
            request.id = id.into();
            Ok(request)
        })
    }

    pub(crate) fn input<S, V>(self, name: S, shape: V, data: InferTensorData) -> Self
    where
        S: Into<String>,
        V: Into<Vec<i64>>,
    {
        self.and_then(|mut request| {
            request.inputs.push(InferInputTensor {
                name: name.into(),
                shape: shape.into(),
                datatype: data.as_ref().into(),
                contents: Some(data.into()),
                ..Default::default()
            });
            Ok(request)
        })
    }

    pub(crate) fn output<S>(self, name: S) -> Self
    where
        S: Into<String>,
    {
        self.and_then(|mut request| {
            request.outputs.push(InferRequestedOutputTensor {
                name: name.into(),
                ..Default::default()
            });
            Ok(request)
        })
    }

    fn and_then<F>(self, f: F) -> Self
    where
        F: FnOnce(ModelInferRequest) -> anyhow::Result<ModelInferRequest>,
    {
        Self {
            inner: self.inner.and_then(f),
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            inner: Ok(ModelInferRequest::default()),
        }
    }
}

pub(crate) enum InferTensorData {
    Bool(Vec<bool>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
    FP32(Vec<f32>),
    FP64(Vec<f64>),
    Bytes(Vec<Vec<u8>>),
}

/// View `InferTensorData` as triton datatype
impl AsRef<str> for InferTensorData {
    fn as_ref(&self) -> &str {
        match self {
            InferTensorData::Bool(_) => "BOOL",
            InferTensorData::Int32(_) => "INT32",
            InferTensorData::Int64(_) => "INT64",
            InferTensorData::UInt32(_) => "UINT32",
            InferTensorData::UInt64(_) => "UINT64",
            InferTensorData::FP32(_) => "FP32",
            InferTensorData::FP64(_) => "FP64",
            InferTensorData::Bytes(_) => "BYTES",
        }
    }
}

impl From<InferTensorData> for InferTensorContents {
    fn from(data: InferTensorData) -> Self {
        match data {
            InferTensorData::Bool(data) => InferTensorContents {
                bool_contents: data,
                ..Default::default()
            },
            InferTensorData::Int32(data) => InferTensorContents {
                int_contents: data,
                ..Default::default()
            },
            InferTensorData::Int64(data) => InferTensorContents {
                int64_contents: data,
                ..Default::default()
            },
            InferTensorData::UInt32(data) => InferTensorContents {
                uint_contents: data,
                ..Default::default()
            },
            InferTensorData::UInt64(data) => InferTensorContents {
                uint64_contents: data,
                ..Default::default()
            },
            InferTensorData::FP32(data) => InferTensorContents {
                fp32_contents: data,
                ..Default::default()
            },
            InferTensorData::FP64(data) => InferTensorContents {
                fp64_contents: data,
                ..Default::default()
            },
            InferTensorData::Bytes(data) => InferTensorContents {
                bytes_contents: data,
                ..Default::default()
            },
        }
    }
}
