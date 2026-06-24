//! Agent Skills support — implements the [Agent Skills specification](https://agentskills.io/specification).
//!
//! Each skill is a directory containing a `SKILL.md` (YAML frontmatter + a
//! Markdown body of instructions). [`register`] discovers skills under a
//! directory and exposes each one to the parent agent as a tool that runs as a
//! subagent, restricted to the tools named in its `allowed-tools` frontmatter.

pub mod config;
pub mod skill;

pub use config::{SkillConfig, SkillError};
pub use skill::{SkillTool, register};
