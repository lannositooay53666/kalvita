//! Kalvita runtime faults: fatal (uncatchable) failures vs typed,
//! catchable errors (`try`/`catch`/`throw`).
use crate::parser::Value;

/// Catchable typed error vs fatal (uncatchable) runtime failure.
/// `Fatal` aborts with `Runtime error: ...`; `Throw` carries a
/// `Value::Error` that the nearest enclosing `try` can catch.
#[derive(Debug, PartialEq, Clone)]
pub enum RuntimeFault {
    Fatal(String),
    Throw(Value),
}

impl From<String> for RuntimeFault {
    fn from(msg: String) -> Self {
        RuntimeFault::Fatal(msg)
    }
}

pub(crate) fn fatal_err(msg: impl Into<String>) -> RuntimeFault {
    RuntimeFault::Fatal(msg.into())
}

pub(crate) fn throw_err(error_type: &str, message: impl Into<String>) -> RuntimeFault {
    RuntimeFault::Throw(Value::Error {
        error_type: error_type.to_string(),
        message: message.into(),
    })
}

pub(crate) fn uncaught_message(err: &Value) -> String {
    match err {
        Value::Error { error_type, message } => format!("Uncaught {}: {}", error_type, message),
        other => format!("Uncaught error: {:?}", other),
    }
}
