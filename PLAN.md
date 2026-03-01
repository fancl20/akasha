# Akasha: LLM Context Management using Zettelkasten

A Go library for managing LLM context using Zettelkasten principles with Lossless Context Management (LCM) patterns, built on Temporal workflows for state and error handling.

## Core Principles

### 1. Immutable Notes
- Every note is a separate markdown file with a unique ID as filename
- Once created, notes are **immutable** - they never change
- Notes can only be linked, referenced, or summarized into new notes

### 2. Sessions as Mutable Work-in-Progress
- Sessions are temporary, mutable containers for ongoing work
- A session represents a conversation or task (including subagent interactions)
- When a session ends, it calls an llm to segment it by topic into notes and becomes immutable

### 3. DAG-Based Summarization (LCM)
- Summaries are NEW notes that link back to original notes
- Nothing is lost - original notes remain accessible via inline links
- Enables "drill-down" from summaries to original details

---

## Architecture Overview

### Application Layer
Uses akasha as a library with the following interface:

#### Session
Session is not concurrent safe - use it in single thread

- `Sessions.List()` - Get all existing sessions
- `Sessions.Create()` - Create new session
- `Session.ID()` - Get the id of session which will become note id after Close()
- `Session.Write([]byte)` - Append message to session
- `Session.ReaderAt([]byte, offset)` - Read session context by bytes
- `Session.Close()` - End the session, mark immutable and await summarization.
- `Session.Summarize()` - Summarize the note for an closed session and return the ID of the summarization note.

When reaching soft threshold, the application will create a new session and close the existing one and trigger summarization. It will still concat the 2 session as context until the summarization finished and replace the old session with the summarization. If hard threshold reached application should wait until summarization finished to prevent more session being created.

Summarization will contain links to the original context, segemented by topics.

The application should maintain dedicated session(s) as index for retrieving notes.

#### MCP
MCP can only read the notes. New notes must be created through session. Search is intentionally restricted as part of the Zettelkasten principle to enable nature degragation of the information.

- `read(note_id, offset, limit)` - Read a immutable note
- `search(query)` - Expore from the index session

### akasha Library

#### Workflows
- Session Workflow
- Explore Workflow

#### Notes
Notes can be stored in different backend (e.g InMemory, Filesystem...)

- `Note.Create(session)` - Create new immutable note from session
- `Note.Get(id)` - Get the note from storage

#### LLM
- Provider Interface (generic, pluggable)

---

## References

- [Zettelkasten Method](https://zettelkasten.de/) - Note-taking methodology
- [LCM: Lossless Context Management](https://papers.voltropy.com/LCM) - DAG-based context preservation
- [Temporal.io](https://temporal.io/) - Workflow orchestration for Go
