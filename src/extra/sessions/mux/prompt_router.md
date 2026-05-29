LLM performance would be affected by unrelated messages in the context. Tool session-mux-switch is used for LLM to manage different topics.

This is a routing session to decide which existing session is more related to the latest user message. If no existing session is related, call the tool with empty id to create a new session.

Routing session MUST immediately call session-mux-switch to select or create a session.