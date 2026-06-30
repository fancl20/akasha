//! Parsing, validation, and discovery for [Agent Skills](https://agentskills.io/specification).
//!
//! A skill is a directory containing a `SKILL.md` file: YAML frontmatter
//! delimited by `---` fences, followed by a Markdown body of instructions.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Parsed `SKILL.md` frontmatter.
///
/// Mirrors the Agent Skills specification. `name` and `description` are
/// required; the rest are optional. `allowed_tools` is deserialized from the
/// `allowed-tools` field, which may be a space-separated scalar or a YAML
/// sequence.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillFrontmatter {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub compatibility: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    #[serde(default, rename = "allowed-tools", deserialize_with = "de_allowed_tools")]
    pub allowed_tools: Vec<String>,
    /// Path (relative to the skill dir) to a JSON file holding the skill's tool
    /// schema, mirroring MCP: an object with optional `inputSchema` and
    /// `outputSchema` subschemas (each a JSON-Schema object). `inputSchema`
    /// describes the args the parent passes (absent ⇒ the default
    /// `{input: string}`); `outputSchema` describes the result the skill must
    /// return via the `skill_submit` tool (absent ⇒ free text). File-only — inline
    /// schemas are not supported.
    #[serde(default)]
    pub schema: Option<String>,
}

/// A skill parsed from disk: its frontmatter, its directory, and the instruction
/// body. [`discover`] parses the whole `SKILL.md` eagerly, so activating a skill
/// needs no further file reads.
#[derive(Debug, Clone)]
pub struct ParsedSkill {
    pub frontmatter: SkillFrontmatter,
    /// The skill's own directory (its name must equal `frontmatter.name`).
    pub dir: PathBuf,
    /// The Markdown instruction body from `SKILL.md` (everything after the
    /// frontmatter fences).
    pub body: String,
}

/// Where to look for skills. Discovery walks `dir` recursively: any directory
/// (at any depth) that directly contains a `SKILL.md` is registered as one
/// skill, so layouts like `skills/search/tavily/SKILL.md` work as well as flat
/// ones like `skills/tavily/SKILL.md`.
#[derive(Debug, Clone)]
pub struct SkillConfig {
    pub dir: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("invalid skill '{name}': {reason}")]
    Invalid { name: String, reason: String },
}

/// Deserialize `allowed-tools` from either a space-separated scalar
/// (`"a b c"`) or a YAML sequence (`["a", "b"]`). Absent or null ⇒ empty.
fn de_allowed_tools<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Seq(Vec<String>),
        Scalar(String),
    }

    match Option::<Raw>::deserialize(deserializer)? {
        None => Ok(Vec::new()),
        Some(Raw::Seq(v)) => Ok(v),
        Some(Raw::Scalar(s)) => Ok(s.split_whitespace().map(String::from).collect()),
    }
}

/// Splits `SKILL.md` content into `(frontmatter_yaml, body)`. Returns `None`
/// when there is no opening `---` fence or no matching closing fence.
fn split_frontmatter(content: &str) -> Option<(String, String)> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    let lines: Vec<&str> = content.lines().collect();

    if lines.first()?.trim_end() != "---" {
        return None;
    }

    // The closing fence is the next line equal to "---".
    let close = lines.iter().skip(1).position(|l| l.trim_end() == "---")? + 1;
    let yaml = lines[1..close].join("\n");
    let body = lines[(close + 1).min(lines.len())..].join("\n");
    Some((yaml, body))
}

/// Parses a `SKILL.md`'s content into validated frontmatter and the raw body.
pub fn parse(content: &str) -> Result<(SkillFrontmatter, String), SkillError> {
    let (yaml, body) = split_frontmatter(content).ok_or_else(|| {
        SkillError::Parse("missing YAML frontmatter: expected leading and closing '---' fences".to_string())
    })?;

    let frontmatter: SkillFrontmatter =
        serde_yml::from_str(&yaml).map_err(|e| SkillError::Parse(format!("invalid frontmatter: {e}")))?;

    validate(&frontmatter)?;
    Ok((frontmatter, body))
}

fn validate(fm: &SkillFrontmatter) -> Result<(), SkillError> {
    let desc_len = fm.description.chars().count();
    if desc_len == 0 || desc_len > 1024 {
        return Err(SkillError::Invalid {
            name: fm.name.clone(),
            reason: format!("description must be 1-1024 characters, got {desc_len}"),
        });
    }
    if let Some(compat) = &fm.compatibility {
        let len = compat.chars().count();
        if len > 500 {
            return Err(SkillError::Invalid {
                name: fm.name.clone(),
                reason: format!("compatibility must be at most 500 characters, got {len}"),
            });
        }
    }
    Ok(())
}

/// Recursively scans `dir` for skills and returns the valid ones. A skill is any
/// directory (at any depth) that directly contains a `SKILL.md`; discovery
/// descends into ordinary directories but stops at a skill's own directory, so a
/// skill's bundled assets are never mistaken for further skills. A missing `dir`
/// is not an error — it yields an empty list. Malformed skills are skipped with a
/// warning on stderr so one bad skill cannot prevent the rest from loading.
pub fn discover(dir: &Path) -> Result<Vec<ParsedSkill>, SkillError> {
    let mut out = Vec::new();
    match std::fs::read_dir(dir) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(SkillError::Io(e)),
    }
    discover_into(dir, &mut out)?;
    Ok(out)
}

/// Recursive step of [`discover`]. If `dir` directly owns a `SKILL.md`, it is a
/// skill: register it and stop (its contents are skill assets, not further
/// skills). Otherwise descend into each subdirectory, sorted for deterministic
/// registration order.
fn discover_into(dir: &Path, out: &mut Vec<ParsedSkill>) -> Result<(), SkillError> {
    if dir.join("SKILL.md").exists() {
        register_one(dir, out);
        return Ok(());
    }

    let mut subdirs: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(SkillError::Io(e)),
    };
    subdirs.sort();

    for subdir in subdirs {
        discover_into(&subdir, out)?;
    }
    Ok(())
}

/// Parses and validates a single skill directory (which must contain a
/// `SKILL.md`), pushing it onto `out` or logging a skip warning on failure.
fn register_one(dir_path: &Path, out: &mut Vec<ParsedSkill>) {
    let dir_name = match dir_path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return,
    };
    let skill_md = dir_path.join("SKILL.md");

    let content = match std::fs::read_to_string(&skill_md) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skill] skipping {}: cannot read SKILL.md ({e})", dir_path.display());
            return;
        }
    };

    let (frontmatter, body) = match parse(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[skill] skipping {}: {e}", dir_path.display());
            return;
        }
    };

    if frontmatter.name != dir_name {
        eprintln!(
            "[skill] skipping {}: frontmatter name '{}' must match directory '{dir_name}'",
            dir_path.display(),
            frontmatter.name
        );
        return;
    }

    out.push(ParsedSkill { frontmatter, dir: dir_path.to_path_buf(), body });
}

/// The default argument schema a skill tool exposes when it declares no
/// `inputSchema`: a single natural-language `input` string.
pub(crate) fn default_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The task to perform with this skill, in natural language."
            }
        },
        "required": ["input"]
    })
}

impl ParsedSkill {
    /// Resolves the skill's `schema` frontmatter field — a path (relative to
    /// [`dir`](ParsedSkill::dir)) to a JSON file mirroring the MCP Tool shape
    /// (`inputSchema`/`outputSchema` peer JSON-Schemas) — into the
    /// `(input, output)` pair a tool surfaces via its `schema()` method (the
    /// `ToolHandler`/`Subagent` trait shape). The input falls back to
    /// [`default_input_schema`] when no `inputSchema` is declared (including
    /// when there is no `schema` file at all); the output is `None` unless an
    /// `outputSchema` is declared. The file and each present subschema must be
    /// JSON objects.
    pub fn resolve_schema(&self) -> Result<(serde_json::Value, Option<serde_json::Value>), SkillError> {
        let Some(path) = self.frontmatter.schema.as_deref() else {
            return Ok((default_input_schema(), None));
        };
        let resolved = self.dir.join(path);
        let text = std::fs::read_to_string(&resolved)
            .map_err(|e| SkillError::Parse(format!("cannot read schema '{path}': {e}")))?;
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| SkillError::Parse(format!("invalid schema '{path}': {e}")))?;
        if !value.is_object() {
            return Err(SkillError::Invalid {
                name: "schema".to_string(),
                reason: "schema must be a JSON object".to_string(),
            });
        }
        let input = take_object(&value, "inputSchema")?.unwrap_or_else(default_input_schema);
        let output = take_object(&value, "outputSchema")?;
        Ok((input, output))
    }
}

/// Extracts `key` from `parent` as an object schema, or `None` if absent.
fn take_object(parent: &serde_json::Value, key: &str) -> Result<Option<serde_json::Value>, SkillError> {
    match parent.get(key) {
        None => Ok(None),
        Some(v) if v.is_object() => Ok(Some(v.clone())),
        Some(_) => Err(SkillError::Invalid { name: key.to_string(), reason: format!("'{key}' must be a JSON object") }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_frontmatter() {
        let (fm, body) = parse("---\nname: my-skill\ndescription: does a thing\n---\nbody text\n").unwrap();
        assert_eq!(fm.name, "my-skill");
        assert_eq!(fm.description, "does a thing");
        assert!(fm.allowed_tools.is_empty());
        assert!(fm.metadata.is_empty());
        assert_eq!(body, "body text");
    }

    #[test]
    fn parses_folded_description() {
        // Folded `>` scalar — a single trailing newline is folded to a space by YAML.
        let src = "---\nname: fold\ndescription: >\n  one line\n  another line\n---\n";
        let (fm, _) = parse(src).unwrap();
        assert_eq!(fm.description, "one line another line\n");
    }

    #[test]
    fn parses_allowed_tools_as_scalar() {
        let (fm, _) = parse("---\nname: s\ndescription: d\nallowed-tools: \"a b c\"\n---\n").unwrap();
        assert_eq!(fm.allowed_tools, vec!["a", "b", "c"]);
    }

    #[test]
    fn parses_allowed_tools_as_sequence() {
        let (fm, _) = parse("---\nname: s\ndescription: d\nallowed-tools:\n  - a\n  - c\n---\n").unwrap();
        assert_eq!(fm.allowed_tools, vec!["a", "c"]);
    }

    #[test]
    fn parses_metadata_map() {
        let (fm, _) =
            parse("---\nname: s\ndescription: d\nmetadata:\n  author: alice\n  version: \"2\"\n---\n").unwrap();
        assert_eq!(fm.metadata["author"], "alice");
        assert_eq!(fm.metadata["version"], "2");
    }

    #[test]
    fn rejects_missing_closing_fence() {
        let err = parse("---\nname: s\ndescription: d\nbody without close").unwrap_err();
        assert!(matches!(err, SkillError::Parse(_)));
    }

    #[test]
    fn rejects_missing_name() {
        let err = parse("---\ndescription: d\n---\n").unwrap_err();
        assert!(matches!(err, SkillError::Parse(_)));
    }

    #[test]
    fn rejects_empty_description() {
        let err = parse("---\nname: s\ndescription: \"\"\n---\n").unwrap_err();
        assert!(matches!(err, SkillError::Invalid { .. }));
    }

    #[test]
    fn discover_missing_dir_is_empty() {
        let skills = discover(Path::new("/nonexistent/akasha-skill-dir-xyz")).unwrap();
        assert!(skills.is_empty());
    }

    /// Removes the wrapped directory tree on drop (best-effort).
    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Writes a `SKILL.md` into `root/.../segments`, creating parents as needed.
    fn write_skill(root: &Path, segments: &[&str]) {
        let dir = segments.iter().fold(root.to_path_buf(), |acc, s| acc.join(s));
        std::fs::create_dir_all(&dir).unwrap();
        let name = *segments.last().unwrap();
        std::fs::write(dir.join("SKILL.md"), format!("---\nname: {name}\ndescription: d\n---\nbody\n")).unwrap();
    }

    #[test]
    fn discover_finds_nested_skills() {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!("akasha-skill-nest-{}-{n}", std::process::id()));
        // Flat skill, a two-deep skill, and a three-deep skill.
        write_skill(&root, &["flat"]);
        write_skill(&root, &["search", "tavily"]);
        write_skill(&root, &["search", "web", "bing"]);
        let guard = TempDir(root.clone());

        let skills = discover(&root).unwrap();
        let mut names: Vec<&str> = skills.iter().map(|s| s.frontmatter.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["bing", "flat", "tavily"]);
        // `dir` is the skill's own directory at every depth.
        assert!(skills.iter().any(|s| s.dir == root.join("search").join("tavily")));
        drop(guard);
    }

    #[test]
    fn discover_does_not_descend_into_a_skill_dir() {
        // A nested dir *inside* a skill's own directory must not be treated as a
        // further skill, even if it has a SKILL.md.
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!("akasha-skill-opaque-{}-{n}", std::process::id()));
        write_skill(&root, &["alpha"]);
        write_skill(&root, &["alpha", "nested"]); // asset inside alpha, not a skill
        let guard = TempDir(root.clone());

        let skills = discover(&root).unwrap();
        let names: Vec<&str> = skills.iter().map(|s| s.frontmatter.name.as_str()).collect();
        assert_eq!(names, vec!["alpha"], "alpha's bundled dir is opaque");
        drop(guard);
    }

    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Writes `contents` to `schema.json` in a fresh temp dir and returns the dir.
    fn schema_dir(contents: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("akasha-skill-cfg-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("schema.json"), contents).unwrap();
        dir
    }

    #[test]
    fn parses_schema_path() {
        let (fm, _) = parse("---\nname: s\ndescription: d\nschema: schema.json\n---\n").unwrap();
        assert_eq!(fm.schema.as_deref(), Some("schema.json"));
    }

    /// Builds a `ParsedSkill` rooted at `dir` with a frontmatter that references
    /// `schema` (or none) — the minimal fixture for exercising `resolve_schema`.
    fn skill_with_schema(schema: Option<&str>, dir: PathBuf) -> ParsedSkill {
        let fm_src = match schema {
            Some(p) => format!("---\nname: s\ndescription: d\nschema: {p}\n---\n"),
            None => "---\nname: s\ndescription: d\n---\n".to_string(),
        };
        let (frontmatter, body) = parse(&fm_src).unwrap();
        ParsedSkill { frontmatter, dir, body }
    }

    #[test]
    fn resolve_schema_extracts_input_output() {
        let dir = schema_dir(
            r#"{"inputSchema":{"type":"object","properties":{"x":{"type":"integer"}}},"outputSchema":{"type":"object","required":["y"]}}"#,
        );
        let (input, output) = skill_with_schema(Some("schema.json"), dir.clone()).resolve_schema().unwrap();
        assert_eq!(input["properties"]["x"]["type"], "integer");
        assert_eq!(output.unwrap()["required"][0], "y");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_schema_allows_input_only() {
        let dir = schema_dir(r#"{"inputSchema":{"type":"object","required":["a"]}}"#);
        let (input, output) = skill_with_schema(Some("schema.json"), dir.clone()).resolve_schema().unwrap();
        assert_eq!(input["required"][0], "a");
        assert!(output.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_schema_rejects_non_object_file() {
        let dir = schema_dir("[1, 2, 3]");
        let err = skill_with_schema(Some("schema.json"), dir.clone()).resolve_schema().unwrap_err();
        assert!(matches!(err, SkillError::Invalid { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_schema_rejects_non_object_subschema() {
        let dir = schema_dir(r#"{"inputSchema":"not-an-object"}"#);
        let err = skill_with_schema(Some("schema.json"), dir.clone()).resolve_schema().unwrap_err();
        assert!(matches!(err, SkillError::Invalid { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_schema_missing_file_is_parse_error() {
        let skill = skill_with_schema(Some("nope.json"), Path::new("/nonexistent-akasha-skill-xyz").to_path_buf());
        let err = skill.resolve_schema().unwrap_err();
        assert!(matches!(err, SkillError::Parse(_)));
    }

    #[test]
    fn resolve_schema_defaults_when_no_schema_file() {
        // No `schema` field: the default `{input: string}` shape is returned and the
        // filesystem is never touched (the nonexistent dir causes no error).
        let skill = skill_with_schema(None, Path::new("/nonexistent-akasha-skill-xyz").to_path_buf());
        let (input, output) = skill.resolve_schema().unwrap();
        assert_eq!(input, default_input_schema());
        assert!(output.is_none());
    }
}
