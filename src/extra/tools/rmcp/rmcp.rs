use std::marker::PhantomData;

use async_trait::async_trait;
use rmcp::handler::server::router::tool::{AsyncTool, ToolBase};
use rmcp::service::MaybeSend;

use crate::core::tools::{ToolError, ToolHandler};
use crate::core::types::{TextContent, ToolDefinition, ToolResult, ToolResultContent};

/// Wrapper that adapts a type implementing [`ToolBase`] + [`AsyncTool<S>`] into
/// akasha's [`ToolHandler`].
pub struct RmcpTool<T, S = ()> {
    definition: ToolDefinition,
    service: S,
    phantom: PhantomData<fn() -> T>,
}

impl<T: ToolBase, S> RmcpTool<T, S> {
    pub fn new(service: S) -> Self {
        Self {
            definition: ToolDefinition {
                name: T::name().into_owned(),
                description: T::description().map(|d| d.into_owned()).unwrap_or_default(),
                parameters: T::input_schema()
                    .map(|schema| serde_json::Value::Object(schema.as_ref().clone()))
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            },
            service,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<T, S> ToolHandler for RmcpTool<T, S>
where
    T: AsyncTool<S> + 'static,
    S: MaybeSend + Send + Sync + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        _cancel: tokio::sync::watch::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let param: T::Parameter = serde_json::from_value(params).map_err(|e| ToolError::Validation(e.to_string()))?;

        let result = T::invoke(&self.service, param).await.map_err(|e| {
            let error_data: rmcp::ErrorData = e.into();
            ToolError::Execution(error_data.message.to_string())
        })?;

        let content = serde_json::to_string(&result)
            .map(|s| vec![ToolResultContent::Text(TextContent { content: s })])
            .unwrap_or_else(|_| {
                vec![ToolResultContent::Text(TextContent {
                    content: "tool returned non-serializable result".to_string(),
                })]
            });

        Ok(ToolResult { tool_call_id: None, content, is_error: false })
    }
}

impl<T> From<T> for Box<dyn ToolHandler>
where
    T: AsyncTool<()> + 'static,
{
    fn from(_tool: T) -> Self {
        Box::new(RmcpTool::<T, ()>::new(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::ErrorData;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use std::borrow::Cow;
    use tokio::sync::watch;

    fn cancel_rx() -> watch::Receiver<bool> {
        watch::channel(false).1
    }

    // --- Mock tools ---

    struct DoubleTool;

    #[derive(Deserialize, JsonSchema, Default)]
    struct DoubleParams {
        input: i64,
    }

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
    struct DoubleOutput {
        result: i64,
    }

    impl ToolBase for DoubleTool {
        type Parameter = DoubleParams;
        type Output = DoubleOutput;
        type Error = ErrorData;

        fn name() -> Cow<'static, str> {
            "double".into()
        }

        fn description() -> Option<Cow<'static, str>> {
            Some("Doubles the input".into())
        }
    }

    impl AsyncTool<()> for DoubleTool {
        async fn invoke(_service: &(), param: Self::Parameter) -> Result<Self::Output, Self::Error> {
            Ok(DoubleOutput { result: param.input * 2 })
        }
    }

    struct NoDescTool;

    impl ToolBase for NoDescTool {
        type Parameter = ();
        type Output = ();
        type Error = ErrorData;

        fn name() -> Cow<'static, str> {
            "no_desc".into()
        }
    }

    impl AsyncTool<()> for NoDescTool {
        async fn invoke(_service: &(), _param: Self::Parameter) -> Result<Self::Output, Self::Error> {
            Ok(())
        }
    }

    struct FailTool;

    impl ToolBase for FailTool {
        type Parameter = ();
        type Output = ();
        type Error = ErrorData;

        fn name() -> Cow<'static, str> {
            "fail".into()
        }

        fn description() -> Option<Cow<'static, str>> {
            Some("Always fails".into())
        }
    }

    impl AsyncTool<()> for FailTool {
        async fn invoke(_service: &(), _param: Self::Parameter) -> Result<Self::Output, Self::Error> {
            Err(ErrorData::internal_error("something went wrong", None))
        }
    }

    // --- RmcpTool::new / definition tests ---

    #[test]
    fn new_stores_definition() {
        let tool = RmcpTool::<DoubleTool, ()>::new(());
        let def = tool.definition();
        assert_eq!(def.name, "double");
        assert_eq!(def.description, "Doubles the input");
    }

    #[test]
    fn definition_returns_clone() {
        let tool = RmcpTool::<DoubleTool, ()>::new(());
        let def1 = tool.definition();
        let def2 = tool.definition();
        assert_eq!(def1.name, def2.name);
    }

    // --- execute tests ---

    #[tokio::test]
    async fn execute_happy_path() {
        let tool = RmcpTool::<DoubleTool, ()>::new(());
        let result = tool.execute(cancel_rx(), serde_json::json!({"input": 21})).await.unwrap();

        assert!(!result.is_error);
        assert!(result.tool_call_id.is_none());
        assert_eq!(result.content.len(), 1);

        match &result.content[0] {
            ToolResultContent::Text(tc) => {
                let output: DoubleOutput = serde_json::from_str(&tc.content).unwrap();
                assert_eq!(output.result, 42);
            }
            _ => panic!("expected Text content"),
        }
    }

    #[tokio::test]
    async fn execute_returns_validation_error_on_bad_params() {
        let tool = RmcpTool::<DoubleTool, ()>::new(());
        let err = tool.execute(cancel_rx(), serde_json::json!({"input": "not_a_number"})).await.unwrap_err();

        match err {
            ToolError::Validation(msg) => assert!(!msg.is_empty()),
            other => panic!("expected Validation error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn execute_returns_execution_error_on_tool_failure() {
        let tool = RmcpTool::<FailTool, ()>::new(());
        let err = tool.execute(cancel_rx(), serde_json::json!(null)).await.unwrap_err();

        match err {
            ToolError::Execution(msg) => assert!(msg.contains("something went wrong")),
            other => panic!("expected Execution error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn execute_unit_tool_with_null_params() {
        let tool = RmcpTool::<NoDescTool, ()>::new(());
        let result = tool.execute(cancel_rx(), serde_json::json!(null)).await.unwrap();
        assert!(!result.is_error);
    }

    // --- From<T> conversion tests ---

    #[test]
    fn from_conversion_produces_correct_definition() {
        let boxed: Box<dyn ToolHandler> = DoubleTool.into();
        assert_eq!(boxed.definition().name, "double");
    }

    #[tokio::test]
    async fn from_boxed_tool_can_execute() {
        let boxed: Box<dyn ToolHandler> = DoubleTool.into();
        let result = boxed.execute(cancel_rx(), serde_json::json!({"input": 5})).await.unwrap();

        match &result.content[0] {
            ToolResultContent::Text(tc) => {
                let output: DoubleOutput = serde_json::from_str(&tc.content).unwrap();
                assert_eq!(output.result, 10);
            }
            _ => panic!("expected Text content"),
        }
    }
}
