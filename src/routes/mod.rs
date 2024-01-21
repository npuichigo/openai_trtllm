pub(crate) use chat::compat_chat_completions;
pub(crate) use completions::compat_completions;
pub(crate) use health_check::health_check;

mod chat;
mod completions;
mod health_check;
