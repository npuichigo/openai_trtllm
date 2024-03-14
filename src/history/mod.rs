use crate::routes::chat::ChatCompletionMessageParams;
use anyhow::bail;
use liquid::{ParserBuilder, Template};
use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;

const DEFAULT_TEMPLATE: &str = "{% for item in items %}\
{{ item.identity }}{% if item.name %} {{ item.name }}{% endif %}: {{ item.content }}
{% endfor %}\
ASSISTANT:";

#[derive(Clone)]
pub struct HistoryBuilder {
    history_template: Arc<Template>,
}

impl HistoryBuilder {
    pub fn new(template: &Option<String>, template_file: &Option<String>) -> anyhow::Result<Self> {
        if template.is_some() && template_file.is_some() {
            bail!("cannot set both history-template and history-template-file")
        }
        let mut _ref_holder = None;

        let template = match template_file {
            None => match template {
                None => DEFAULT_TEMPLATE,
                Some(cfg) => cfg.as_str(),
            },
            Some(filename) => {
                _ref_holder = Some(load_template_file(filename)?);
                _ref_holder.as_ref().unwrap().as_str()
            }
        };

        let history_template = Arc::new(ParserBuilder::with_stdlib().build()?.parse(template)?);

        Ok(HistoryBuilder { history_template })
    }

    pub fn build_history(
        &self,
        messages: &Vec<ChatCompletionMessageParams>,
    ) -> anyhow::Result<String> {
        let items: Vec<_> = messages.iter().map(|x| HistoryItem::new(x)).collect();
        let context = liquid::object!({"items": items});
        Ok(self.history_template.render(&context)?)
    }
}

fn load_template_file(file: &String) -> anyhow::Result<String> {
    let mut file = File::open(file)?;
    let mut result = String::new();
    file.read_to_string(&mut result)?;
    Ok(result)
}

#[derive(Serialize)]
struct HistoryItem {
    identity: String,
    content: String,
    name: Option<String>,
}

impl HistoryItem {
    pub fn new(message: &ChatCompletionMessageParams) -> Self {
        let (identity, content, name) = match message {
            ChatCompletionMessageParams::System { content, name } => {
                ("System".into(), content.clone(), name.clone())
            }
            ChatCompletionMessageParams::User { content, name } => {
                ("User".into(), content.clone(), name.clone())
            }
            ChatCompletionMessageParams::Assistant { content } => {
                ("Assistant".into(), content.clone(), None)
            }
            ChatCompletionMessageParams::Tool { content, .. } => {
                ("Tool".into(), content.clone(), None)
            }
        };

        HistoryItem {
            identity,
            content,
            name,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn test_default_template() {
        let template = None;
        let template_file = None;
        let builder = HistoryBuilder::new(&template, &template_file)
            .expect("default template should build correctly");

        let messages = vec![
            ChatCompletionMessageParams::System {
                content: "test system 1".into(),
                name: Some("system 1".into()),
            },
            ChatCompletionMessageParams::System {
                content: "test system 2".into(),
                name: None,
            },
            ChatCompletionMessageParams::Assistant {
                content: "test assistant 1".into(),
            },
            ChatCompletionMessageParams::Tool {
                content: "test tool 1".into(),
                tool_call_id: "tool_1".into(),
            },
            ChatCompletionMessageParams::User {
                content: "test user 1".into(),
                name: Some("user 1".into()),
            },
            ChatCompletionMessageParams::User {
                content: "test user 2".into(),
                name: None,
            },
        ];

        let result = builder
            .build_history(&messages)
            .expect("history should build correctly");

        let expected_result: String = "System system 1: test system 1
System: test system 2
Assistant: test assistant 1
Tool: test tool 1
User user 1: test user 1
User: test user 2
ASSISTANT:"
            .into();

        assert_eq!(expected_result, result)
    }

    #[test]
    pub fn test_template_file() {
        let template = None;
        let template_file = Some(format!(
            "{}/example/history_template.liquid",
            env!("CARGO_MANIFEST_DIR")
        ));
        let builder = HistoryBuilder::new(&template, &template_file)
            .expect("default template should build correctly");

        let messages = vec![
            ChatCompletionMessageParams::System {
                content: "test system 1".into(),
                name: Some("system 1".into()),
            },
            ChatCompletionMessageParams::System {
                content: "test system 2".into(),
                name: None,
            },
            ChatCompletionMessageParams::Assistant {
                content: "test assistant 1".into(),
            },
            ChatCompletionMessageParams::Tool {
                content: "test tool 1".into(),
                tool_call_id: "tool_1".into(),
            },
            ChatCompletionMessageParams::User {
                content: "test user 1".into(),
                name: Some("user 1".into()),
            },
            ChatCompletionMessageParams::User {
                content: "test user 2".into(),
                name: None,
            },
        ];

        let result = builder
            .build_history(&messages)
            .expect("history should build correctly");

        let expected_result: String = "System system 1: test system 1
System: test system 2
Assistant: test assistant 1
Tool: test tool 1
User user 1: test user 1
User: test user 2
ASSISTANT:"
            .into();

        assert_eq!(expected_result, result)
    }

    #[test]
    pub fn test_template_file_custom_roles() {
        let template = None;
        let template_file = Some(format!(
            "{}/example/history_template_custom_roles.liquid",
            env!("CARGO_MANIFEST_DIR")
        ));
        let builder = HistoryBuilder::new(&template, &template_file)
            .expect("default template should build correctly");

        let messages = vec![
            ChatCompletionMessageParams::System {
                content: "test system 1".into(),
                name: Some("system 1".into()),
            },
            ChatCompletionMessageParams::System {
                content: "test system 2".into(),
                name: None,
            },
            ChatCompletionMessageParams::Assistant {
                content: "test assistant 1".into(),
            },
            ChatCompletionMessageParams::Tool {
                content: "test tool 1".into(),
                tool_call_id: "tool_1".into(),
            },
            ChatCompletionMessageParams::User {
                content: "test user 1".into(),
                name: Some("user 1".into()),
            },
            ChatCompletionMessageParams::User {
                content: "test user 2".into(),
                name: None,
            },
        ];

        let result = builder
            .build_history(&messages)
            .expect("history should build correctly");

        let expected_result: String = "Robot system 1: test system 1
Robot: test system 2
Support: test assistant 1
Robot: test tool 1
Customer user 1: test user 1
Customer: test user 2
ASSISTANT:"
            .into();

        assert_eq!(expected_result, result)
    }

    #[test]
    pub fn test_validations() {
        let template = Some("abc".into());
        let template_file = Some("abc".into());
        match HistoryBuilder::new(&template, &template_file) {
            Ok(_) => {
                assert!(false, "expected err")
            }
            Err(e) => {
                assert_eq!(
                    "cannot set both history-template and history-template-file",
                    e.to_string()
                )
            }
        };
    }
}
