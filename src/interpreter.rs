//! Kalvita tree-walk interpreter: modules, execution, builtins.
use crate::error::{RuntimeFault, fatal_err, throw_err, uncaught_message};
use crate::parser::{
    AssignTarget, BinaryOperator, CatchClause, Header, Parser, Program, Statement, UnaryOperator,
    Value,
};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
enum Flow {
    Normal,
    Break,
    Continue,
    Return(Value),
}

const MAX_LOOP_ITERS: usize = 1_000_000;
const MAX_CALL_DEPTH: usize = 32;

/// RAII decrement for the shared call-depth counter so early `?`
/// returns (including catchable throws) never leak depth.
struct DepthGuard {
    depth: Rc<Cell<usize>>,
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        self.depth.set(self.depth.get().saturating_sub(1));
    }
}

fn enter_call(rt: &ModuleRuntime) -> Result<DepthGuard, RuntimeFault> {    let depth = rt.call_depth.get() + 1;
    rt.call_depth.set(depth);
    if depth > MAX_CALL_DEPTH {
        rt.call_depth.set(depth - 1);
        return Err(fatal_err(
            "maximum call depth exceeded (possible infinite recursion)",
        ));
    }
    Ok(DepthGuard {
        depth: Rc::clone(&rt.call_depth),
    })
}

/// Bind call args to `arg.<param>`: positional args first, then declared
/// defaults (resolved in the caller scope). Missing required params are a
/// catchable `ValueError`; extras stay available via the `arg` array.
#[allow(clippy::too_many_arguments)]
fn bind_params(
    params: &[String],
    defaults: &HashMap<String, Value>,
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
    function: &str,
    local_env: &mut HashMap<String, Value>,
) -> Result<(), RuntimeFault> {
    for (i, param) in params.iter().enumerate() {
        if let Some(arg) = args.get(i) {
            let value = resolve_value(arg, environment, types, rt, cur)?;
            local_env.insert(format!("arg.{}", param), value);
        } else if let Some(default) = defaults.get(param) {
            let value = resolve_value(default, environment, types, rt, cur)?;
            local_env.insert(format!("arg.{}", param), value);
        } else {
            return Err(throw_err(
                "ValueError",
                format!("{}() missing required argument '{}'", function, param),
            ));
        }
    }
    Ok(())
}

pub const ENTRY_ALIAS: &str = "__main__";

/// A loaded module: file-private top-level locals plus exported globals.
/// Globals are read live from here on every `var.x` / `alias:var.x` access.
#[derive(Debug, Clone)]
struct LoadedModule {
    path: PathBuf,
    top_locals: HashMap<String, Value>,
    top_local_types: HashMap<String, String>,
    globals: HashMap<String, Value>,
    global_names: HashSet<String>,
    global_types: HashMap<String, String>,
    program: Option<Program>,
    onstart_executed: bool,
}

impl LoadedModule {
    fn empty() -> Self {
        Self {
            path: PathBuf::new(),
            top_locals: HashMap::new(),
            top_local_types: HashMap::new(),
            globals: HashMap::new(),
            global_names: HashSet::new(),
            global_types: HashMap::new(),
            program: None,
            onstart_executed: false,
        }
    }
}

/// Module loader + live global store. Threaded through the whole
/// interpreter (`rt`) alongside the current file alias (`cur`).
#[derive(Debug)]
pub struct ModuleRuntime {
    modules: HashMap<String, LoadedModule>,
    canonical_to_alias: HashMap<PathBuf, String>,
    loading: Vec<PathBuf>,
    call_depth: Rc<Cell<usize>>,
}

impl Default for ModuleRuntime {
    fn default() -> Self {
        Self {
            modules: HashMap::new(),
            canonical_to_alias: HashMap::new(),
            loading: Vec::new(),
            call_depth: Rc::new(Cell::new(0)),
        }
    }
}

impl ModuleRuntime {
    pub fn new(_base_dir: PathBuf) -> Self {
        Self::default()
    }

    fn ensure_module(&mut self, alias: &str) {
        self.modules
            .entry(alias.to_string())
            .or_insert_with(LoadedModule::empty);
    }

    fn is_global(&self, alias: &str, short_name: &str) -> bool {
        self.modules
            .get(alias)
            .map(|m| m.global_names.contains(short_name))
            .unwrap_or(false)
    }

    /// Live read of a global; falls back to `None` when absent.
    fn read_global(&self, alias: &str, short_name: &str) -> Option<Value> {
        self.modules.get(alias).and_then(|m| {
            m.globals.get(&format!("var.{}", short_name)).cloned()
        })
    }

    /// Merged call base for a module: globals first, top-level locals
    /// overlay (locals shadow on conflict).
    fn module_merged_env(&self, alias: &str) -> HashMap<String, Value> {
        let mut env = HashMap::new();
        if let Some(m) = self.modules.get(alias) {
            for (k, v) in &m.globals {
                env.insert(k.clone(), v.clone());
            }
            for (k, v) in &m.top_locals {
                env.insert(k.clone(), v.clone());
            }
        }
        env
    }

    /// Declared types merged the same way as values (locals shadow).
    fn module_merged_types(&self, alias: &str) -> HashMap<String, String> {
        let mut types = HashMap::new();
        if let Some(m) = self.modules.get(alias) {
            for name in &m.global_names {
                if let Some(ty) = m.global_types.get(name) {
                    types.insert(format!("var.{}", name), ty.clone());
                }
            }
            for (k, v) in &m.top_local_types {
                types.insert(k.clone(), v.clone());
            }
        }
        types
    }

    fn lookup_type(&self, alias: &str, key: &str) -> Option<String> {
        let short = key.strip_prefix("var.")?;
        if self
            .modules
            .get(alias)
            .map(|m| m.global_names.contains(short))
            .unwrap_or(false)
        {
            self.modules
                .get(alias)
                .and_then(|m| m.global_types.get(short).cloned())
        } else {
            None
        }
    }

    /// Snapshot file-scope locals after top-level execution: every
    /// `var.*` entry that is not a declared global.
    fn snapshot_top_locals(
        &mut self,
        alias: &str,
        env: &HashMap<String, Value>,
        types: &HashMap<String, String>,
    ) {
        self.ensure_module(alias);
        let global_names = self.modules[alias].global_names.clone();
        let module = self.modules.get_mut(alias).unwrap();
        module.top_locals.clear();
        module.top_local_types.clear();
        for (k, v) in env {
            if let Some(short) = k.strip_prefix("var.") {
                if !global_names.contains(short) {
                    module.top_locals.insert(k.clone(), v.clone());
                    if let Some(ty) = types.get(k) {
                        module.top_local_types.insert(k.clone(), ty.clone());
                    }
                }
            }
        }
    }
}

fn is_valid_alias(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn alias_from_path(path: &Path) -> Result<String, String> {
    match path.file_stem().and_then(|s| s.to_str()) {
        Some(stem) if is_valid_alias(stem) => Ok(stem.to_string()),
        _ => Err(format!(
            "Module error: cannot derive module alias from '{}': file stem must match [A-Za-z_][A-Za-z0-9_]*",
            path.display()
        )),
    }
}

/// Run a file with module loading: `declare()` paths resolve against the
/// entry file's directory, load once (cached), cycles rejected.
pub fn run_file(entry: &Path) -> Result<(), String> {
    let canonical = fs::canonicalize(entry)
        .map_err(|e| format!("Cannot load '{}': {}", entry.display(), e))?;
    let base = canonical
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let source = fs::read_to_string(&canonical)
        .map_err(|e| format!("Cannot read '{}': {}", canonical.display(), e))?;
    run_source(&source, &base, &canonical)
}

/// Parse + load + execute a source string. `base_dir` anchors relative
/// `declare()` paths; `label` names the entry in error messages.
pub fn run_source(source: &str, base_dir: &Path, label: &Path) -> Result<(), String> {
    let program = Parser::parse(source)
        .map_err(|e| format!("Parse error in '{}': {}", label.display(), e))?;
    let mut rt = ModuleRuntime::new(base_dir.to_path_buf());
    // Seed the loading stack with the entry itself (when resolvable) so
    // self-imports and entry-involved cycles report instead of recursing.
    let canonical_label = fs::canonicalize(label).ok();
    if let Some(ref canonical) = canonical_label {
        rt.loading.push(canonical.clone());
    }
    let result = execute_with_runtime(&program, &mut rt, ENTRY_ALIAS, base_dir);
    if canonical_label.is_some() {
        rt.loading.pop();
    }
    result
}

/// Persistent session for the REPL: one runtime + one file scope kept
/// alive across inputs. Each chunk is wrapped as top-level statements so
/// `var` decls, functions, loops, and even `declare()` persist.
pub struct Repl {
    rt: ModuleRuntime,
    env: HashMap<String, Value>,
    types: HashMap<String, String>,
    base_dir: PathBuf,
}

impl Repl {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            rt: ModuleRuntime::default(),
            env: HashMap::new(),
            types: HashMap::new(),
            base_dir,
        }
    }

    pub fn run_chunk(&mut self, chunk: &str) -> Result<(), String> {
        let wrapped = format!("[SCRIPTTYPE KALVITA VERSION 1]\n{}\n", chunk);
        let program = Parser::parse(&wrapped)?;
        for stmt in &program.statements {
            if let Statement::Declare { path, run } = stmt {
                declare_module(&mut self.rt, &self.base_dir.clone(), path, *run)?;
            }
        }
        self.rt.ensure_module(ENTRY_ALIAS);
        for stmt in &program.statements {
            if matches!(stmt, Statement::Declare { .. }) {
                continue;
            }
            match execute_statement(stmt, &mut self.env, &mut self.types, &mut self.rt, ENTRY_ALIAS) {
                Ok(Flow::Normal) => {}
                Ok(Flow::Break) => return Err("break outside loop".to_string()),
                Ok(Flow::Continue) => return Err("continue outside loop".to_string()),
                Ok(Flow::Return(_)) => {
                    return Err("return can only be used inside a function body".to_string())
                }
                Err(RuntimeFault::Fatal(msg)) => return Err(msg),
                Err(RuntimeFault::Throw(err)) => return Err(uncaught_message(&err)),
            }
        }
        self.rt.snapshot_top_locals(ENTRY_ALIAS, &self.env, &self.types);
        Ok(())
    }
}

fn execute_with_runtime(
    program: &Program,
    rt: &mut ModuleRuntime,
    alias: &str,
    importer_dir: &Path,
) -> Result<(), String> {
    // Phase 1: load declares depth-first (cached, cycles rejected).
    for stmt in &program.statements {
        if let Statement::Declare { path, run } = stmt {
            declare_module(rt, importer_dir, path, *run)?;
        }
    }
    // Phase 2: run own top-level (declares are no-ops here).
    rt.ensure_module(alias);
    let mut environment: HashMap<String, Value> = HashMap::new();
    let mut types: HashMap<String, String> = HashMap::new();
    for statement in &program.statements {
        if matches!(statement, Statement::Declare { .. }) {
            continue;
        }
        match execute_statement(statement, &mut environment, &mut types, rt, alias) {
            Ok(Flow::Normal) => {}
            Ok(Flow::Break) => return Err("break outside loop".to_string()),
            Ok(Flow::Continue) => return Err("continue outside loop".to_string()),
            Ok(Flow::Return(_)) => {
                return Err("return can only be used inside a function body".to_string())
            }
            Err(RuntimeFault::Fatal(msg)) => return Err(msg),
            Err(RuntimeFault::Throw(err)) => return Err(uncaught_message(&err)),
        }
    }
    rt.snapshot_top_locals(alias, &environment, &types);
    Ok(())
}

/// Find the enclosing project root (nearest dir holding `kal.toml`).
fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut dir = if start.is_file() {
        start.parent().map(Path::to_path_buf)
    } else {
        Some(start.to_path_buf())
    };
    while let Some(current) = dir {
        if current.join("kal.toml").is_file() {
            return Some(current);
        }
        dir = current.parent().map(Path::to_path_buf);
    }
    None
}

/// Resolve + load one `declare()` import. Cached by canonical path;
/// a later `run=true` upgrades a previously `run=false` load.
fn declare_module(
    rt: &mut ModuleRuntime,
    importer_dir: &Path,
    raw: &str,
    run: bool,
) -> Result<(), String> {
    let joined = if raw.starts_with("@/") {
        match find_project_root(importer_dir) {
            Some(root) => root.join(&raw[2..]),
            None => {
                return Err(format!(
                    "Module error: '@/...' needs a kal.toml project root (imported from '{}')",
                    importer_dir.display()
                ))
            }
        }
    } else if raw.starts_with('/') {
        PathBuf::from(raw)
    } else {
        importer_dir.join(raw)
    };
    let canonical = fs::canonicalize(&joined).map_err(|e| {
        format!(
            "Module error: cannot load '{}' (imported from '{}'): {}",
            raw,
            importer_dir.display(),
            e
        )
    })?;
    if rt.loading.contains(&canonical) {
        let mut chain: Vec<String> = rt
            .loading
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        chain.push(canonical.display().to_string());
        return Err(format!("Module error: circular import: {}", chain.join(" -> ")));
    }
    if let Some(existing) = rt.canonical_to_alias.get(&canonical).cloned() {
        if run && !rt.modules[&existing].onstart_executed {
            run_module_onstart(rt, &existing)?;
        }
        return Ok(());
    }
    let alias = alias_from_path(&canonical)?;
    if rt.modules.contains_key(&alias) {
        return Err(format!(
            "Module error: duplicate module alias '{}' from '{}'",
            alias,
            canonical.display()
        ));
    }
    load_module(rt, canonical, alias, run)
}

fn load_module(
    rt: &mut ModuleRuntime,
    canonical: PathBuf,
    alias: String,
    run: bool,
) -> Result<(), String> {
    let source = fs::read_to_string(&canonical)
        .map_err(|e| format!("Module error: cannot read '{}': {}", canonical.display(), e))?;
    let program = Parser::parse(&source)
        .map_err(|e| format!("Module error: '{}': parse error: {}", canonical.display(), e))?;
    let dep_dir = canonical
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    rt.loading.push(canonical.clone());
    let mut module = LoadedModule::empty();
    module.path = canonical.clone();
    module.program = Some(program.clone());
    rt.modules.insert(alias.clone(), module);
    rt.canonical_to_alias.insert(canonical.clone(), alias.clone());
    // Nested declares first (depth-first).
    let nested: Vec<(String, bool)> = program
        .statements
        .iter()
        .filter_map(|s| match s {
            Statement::Declare { path, run } => Some((path.clone(), *run)),
            _ => None,
        })
        .collect();
    for (path, run) in nested {
        declare_module(rt, &dep_dir, &path, run)?;
    }
    // Own top-level (declares skipped; events deferred to
    // run_module_onstart so `run=false` truly skips them).
    let mut env: HashMap<String, Value> = HashMap::new();
    let mut env_types: HashMap<String, String> = HashMap::new();
    for stmt in &program.statements {
        if matches!(
            stmt,
            Statement::Declare { .. } | Statement::Event { .. }
        ) {
            continue;
        }
        match execute_statement(stmt, &mut env, &mut env_types, rt, &alias) {
            Ok(Flow::Normal) => {}
            Ok(Flow::Break) => {
                rt.loading.pop();
                return Err(format!(
                    "Module error: '{}': break outside loop",
                    canonical.display()
                ));
            }
            Ok(Flow::Continue) => {
                rt.loading.pop();
                return Err(format!(
                    "Module error: '{}': continue outside loop",
                    canonical.display()
                ));
            }
            Ok(Flow::Return(_)) => {
                rt.loading.pop();
                return Err(format!(
                    "Module error: '{}': return can only be used inside a function body",
                    canonical.display()
                ));
            }
            Err(RuntimeFault::Fatal(msg)) => {
                rt.loading.pop();
                return Err(format!("Module error: '{}': {}", canonical.display(), msg));
            }
            Err(RuntimeFault::Throw(err)) => {
                rt.loading.pop();
                return Err(format!(
                    "Module error: '{}': {}",
                    canonical.display(),
                    uncaught_message(&err)
                ));
            }
        }
    }
    rt.snapshot_top_locals(&alias, &env, &env_types);
    if run {
        run_module_onstart(rt, &alias)?;
    }
    rt.loading.pop();
    Ok(())
}

/// Execute a module's `kal.OnStart` bodies once against its live tables.
fn run_module_onstart(rt: &mut ModuleRuntime, alias: &str) -> Result<(), String> {
    let already = rt
        .modules
        .get(alias)
        .map(|m| m.onstart_executed)
        .unwrap_or(true);
    if already {
        return Ok(());
    }
    let program = rt
        .modules
        .get(alias)
        .and_then(|m| m.program.clone())
        .unwrap_or(Program {
            header: Header {
                script_type: String::new(),
                language: String::new(),
                version: 0,
            },
            statements: Vec::new(),
        });
    let events: Vec<Statement> = program
        .statements
        .iter()
        .filter(|s| matches!(s, Statement::Event { .. }))
        .cloned()
        .collect();
    let path_display = rt
        .modules
        .get(alias)
        .map(|m| m.path.display().to_string())
        .unwrap_or_default();
    let mut env = rt.module_merged_env(alias);
    let mut env_types = rt.module_merged_types(alias);
    for event in &events {
        match execute_statement(event, &mut env, &mut env_types, rt, alias) {
            Ok(_) => {}
            Err(RuntimeFault::Fatal(msg)) => {
                return Err(format!("Module error: '{}': {}", path_display, msg))
            }
            Err(RuntimeFault::Throw(err)) => {
                return Err(format!(
                    "Module error: '{}': {}",
                    path_display,
                    uncaught_message(&err)
                ))
            }
        }
    }
    // Write back any globals touched by OnStart, refresh locals snapshot.
    // (GlobalDecl and global assignment already write the live store;
    // this only refreshes the locals snapshot. Types stay as declared.)
    if let Some(module) = rt.modules.get_mut(alias) {
        for name in module.global_names.clone() {
            let key = format!("var.{}", name);
            if let Some(value) = env.get(&key).cloned() {
                module.globals.insert(key, value);
            }
        }
    }
    rt.snapshot_top_locals(alias, &env, &env_types);
    if let Some(module) = rt.modules.get_mut(alias) {
        module.onstart_executed = true;
    }
    Ok(())
}

fn execute_block(
    statements: &[Statement],
    environment: &mut HashMap<String, Value>,
    types: &mut HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Flow, RuntimeFault> {
    for stmt in statements {
        match execute_statement(stmt, environment, types, rt, cur)? {
            Flow::Normal => {}
            other => return Ok(other),
        }
    }
    Ok(Flow::Normal)
}

/// Display a value as a string for `str.From` / `file.Write`.
fn stringify_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Number(n) => n.to_string(),
        Value::Logic(b) => b.to_string(),
        other => format_value(other.clone()),
    }
}

fn resolve_args(
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Vec<Value>, RuntimeFault> {
    args.iter()
        .map(|arg| resolve_value(arg, environment, types, rt, cur))
        .collect()
}

fn expect_string(value: &Value, what: &str) -> Result<String, RuntimeFault> {
    match value {
        Value::String(s) => Ok(s.clone()),
        _ => Err(throw_err(
            "TypeError",
            format!("{} expects a string, got {:?}", what, value),
        )),
    }
}

fn expect_number(value: &Value, what: &str) -> Result<f64, RuntimeFault> {
    match value {
        Value::Number(n) => Ok(*n),
        _ => Err(throw_err(
            "TypeError",
            format!("{} expects a number, got {:?}", what, value),
        )),
    }
}

fn expect_array(value: &Value, what: &str) -> Result<Vec<Value>, RuntimeFault> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        _ => Err(throw_err(
            "TypeError",
            format!("{} expects an array, got {:?}", what, value),
        )),
    }
}

fn expect_int(value: &Value, what: &str) -> Result<i64, RuntimeFault> {
    match value {
        Value::Number(n) => as_i64(*n).map_err(|_| {
            throw_err(
                "ValueError",
                format!("{} expects an integer, got {}", what, n),
            )
        }),
        _ => Err(throw_err(
            "TypeError",
            format!("{} expects a number, got {:?}", what, value),
        )),
    }
}

fn invoke_math(
    function: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Option<Value>, RuntimeFault> {
    let resolved = resolve_args(args, environment, types, rt, cur)?;
    match function {
        "Sin" | "Cos" | "Tan" | "Sqrt" | "Floor" | "Ceil" | "Abs" => {
            let n = match resolved.as_slice() {
                [v] => expect_number(v, function)?,
                _ => {
                    return Err(throw_err(
                        "ValueError",
                        format!("{} expects exactly one argument", function),
                    ))
                }
            };
            let result = match function {
                "Sin" => n.sin(),
                "Cos" => n.cos(),
                "Tan" => n.tan(),
                "Sqrt" => {
                    if n < 0.0 {
                        return Err(throw_err("ValueError", "Sqrt of negative number"));
                    }
                    n.sqrt()
                }
                "Floor" => n.floor(),
                "Ceil" => n.ceil(),
                _ => n.abs(),
            };
            Ok(Some(Value::Number(result)))
        }
        "Pow" => match resolved.as_slice() {
            [a, b] => Ok(Some(Value::Number(
                expect_number(a, "Pow")?.powf(expect_number(b, "Pow")?),
            ))),
            _ => Err(throw_err("ValueError", "Pow expects exactly two arguments")),
        },
        "Min" | "Max" => {
            if resolved.is_empty() {
                return Err(throw_err(
                    "ValueError",
                    format!("{} expects at least one argument", function),
                ));
            }
            let mut numbers = Vec::with_capacity(resolved.len());
            for v in &resolved {
                numbers.push(expect_number(v, function)?);
            }
            let best = numbers.into_iter().fold(None, |acc: Option<f64>, n| {
                Some(match acc {
                    None => n,
                    Some(m) if function == "Min" => m.min(n),
                    Some(m) => m.max(n),
                })
            });
            Ok(Some(Value::Number(best.unwrap_or(0.0))))
        }
        "Clamp" => match resolved.as_slice() {
            [x, lo, hi] => {
                let (x, lo, hi) = (
                    expect_number(x, "Clamp")?,
                    expect_number(lo, "Clamp")?,
                    expect_number(hi, "Clamp")?,
                );
                Ok(Some(Value::Number(x.clamp(lo.min(hi), lo.max(hi)))))
            }
            _ => Err(throw_err(
                "ValueError",
                "Clamp expects (value, low, high)",
            )),
        },
        "Random" => {
            if !resolved.is_empty() {
                return Err(throw_err("ValueError", "Random expects no arguments"));
            }
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs()))
                .unwrap_or(0x9E3779B97F4A7C15);
            // xorshift64* — no external crates needed.
            let mut x = nanos | 1;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let out = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64
                / (u64::MAX >> 11) as f64;
            Ok(Some(Value::Number(out)))
        }
        _ => Err(throw_err(
            "NameError",
            format!("Unknown math function: {}", function),
        )),
    }
}

fn invoke_str(
    function: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Option<Value>, RuntimeFault> {
    let resolved = resolve_args(args, environment, types, rt, cur)?;
    match function {
        "Len" => match resolved.as_slice() {
            [v] => Ok(Some(Value::Number(expect_string(v, "Len")?.chars().count() as f64))),
            _ => Err(throw_err("ValueError", "Len expects exactly one string")),
        },
        "Upper" | "Lower" => match resolved.as_slice() {
            [v] => {
                let s = expect_string(v, function)?;
                Ok(Some(Value::String(if function == "Upper" {
                    s.to_uppercase()
                } else {
                    s.to_lowercase()
                })))
            }
            _ => Err(throw_err(
                "ValueError",
                format!("{} expects exactly one string", function),
            )),
        },
        "Split" => match resolved.as_slice() {
            [s, sep] => {
                let (s, sep) = (expect_string(s, "Split")?, expect_string(sep, "Split")?);
                Ok(Some(Value::Array(
                    s.split(sep.as_str())
                        .map(|p| Value::String(p.to_string()))
                        .collect(),
                )))
            }
            _ => Err(throw_err("ValueError", "Split expects (string, separator)")),
        },
        "Join" => match resolved.as_slice() {
            [arr, sep] => {
                let (items, sep) = (expect_array(arr, "Join")?, expect_string(sep, "Join")?);
                let mut parts = Vec::with_capacity(items.len());
                for item in &items {
                    parts.push(stringify_value(item));
                }
                Ok(Some(Value::String(parts.join(sep.as_str()))))
            }
            _ => Err(throw_err("ValueError", "Join expects (array, separator)")),
        },
        "Contains" => match resolved.as_slice() {
            [s, sub] => {
                let (s, sub) = (expect_string(s, "Contains")?, expect_string(sub, "Contains")?);
                Ok(Some(Value::Logic(s.contains(sub.as_str()))))
            }
            _ => Err(throw_err(
                "ValueError",
                "Contains expects (string, substring)",
            )),
        },
        "Replace" => match resolved.as_slice() {
            [s, old, new] => {
                let (s, old, new) = (
                    expect_string(s, "Replace")?,
                    expect_string(old, "Replace")?,
                    expect_string(new, "Replace")?,
                );
                Ok(Some(Value::String(s.replace(old.as_str(), new.as_str()))))
            }
            _ => Err(throw_err(
                "ValueError",
                "Replace expects (string, old, new)",
            )),
        },
        "Trim" => match resolved.as_slice() {
            [v] => Ok(Some(Value::String(
                expect_string(v, "Trim")?.trim().to_string(),
            ))),
            _ => Err(throw_err("ValueError", "Trim expects exactly one string")),
        },
        "Sub" => match resolved.as_slice() {
            [s, start] => {
                let (s, start) = (expect_string(s, "Sub")?, expect_int(start, "Sub")?);
                substring(s, start, None).map(Value::String).map(Some)
            }
            [s, start, len] => {
                let (s, start, len) = (
                    expect_string(s, "Sub")?,
                    expect_int(start, "Sub")?,
                    expect_int(len, "Sub")?,
                );
                substring(s, start, Some(len)).map(Value::String).map(Some)
            }
            _ => Err(throw_err("ValueError", "Sub expects (string, start[, length])")),
        },
        "From" => match resolved.as_slice() {
            [v] => Ok(Some(Value::String(stringify_value(v)))),
            _ => Err(throw_err("ValueError", "From expects exactly one value")),
        },
        "ToNum" => match resolved.as_slice() {
            [v] => {
                let s = expect_string(v, "ToNum")?;
                s.trim().parse::<f64>().map(Value::Number).map(Some).map_err(|_| {
                    throw_err("ValueError", format!("cannot convert to number: '{}'", s))
                })
            }
            _ => Err(throw_err("ValueError", "ToNum expects exactly one string")),
        },
        _ => Err(throw_err(
            "NameError",
            format!("Unknown str function: {}", function),
        )),
    }
}

fn substring(s: String, start: i64, len: Option<i64>) -> Result<String, RuntimeFault> {
    if start < 0 {
        return Err(throw_err("ValueError", "Sub start must be >= 0"));
    }
    if let Some(len) = len {
        if len < 0 {
            return Err(throw_err("ValueError", "Sub length must be >= 0"));
        }
    }
    let chars: Vec<char> = s.chars().collect();
    let start = (start as usize).min(chars.len());
    let end = match len {
        Some(len) => (start + len as usize).min(chars.len()),
        None => chars.len(),
    };
    Ok(chars[start..end].iter().collect())
}

fn invoke_arr(
    function: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Option<Value>, RuntimeFault> {
    let resolved = resolve_args(args, environment, types, rt, cur)?;
    match function {
        "Len" => match resolved.as_slice() {
            [v] => Ok(Some(Value::Number(expect_array(v, "Len")?.len() as f64))),
            _ => Err(throw_err("ValueError", "Len expects exactly one array")),
        },
        "Push" => match resolved.as_slice() {
            [arr, value] => {
                let mut items = expect_array(arr, "Push")?;
                items.push(value.clone());
                Ok(Some(Value::Array(items)))
            }
            _ => Err(throw_err("ValueError", "Push expects (array, value)")),
        },
        "Pop" => match resolved.as_slice() {
            [arr] => {
                let mut items = expect_array(arr, "Pop")?;
                if items.is_empty() {
                    return Err(throw_err("IndexError", "Pop from empty array"));
                }
                items.pop();
                Ok(Some(Value::Array(items)))
            }
            _ => Err(throw_err("ValueError", "Pop expects exactly one array")),
        },
        "Reverse" => match resolved.as_slice() {
            [arr] => {
                let mut items = expect_array(arr, "Reverse")?;
                items.reverse();
                Ok(Some(Value::Array(items)))
            }
            _ => Err(throw_err("ValueError", "Reverse expects exactly one array")),
        },
        "Sort" => match resolved.as_slice() {
            [arr] => {
                let mut items = expect_array(arr, "Sort")?;
                if items.iter().all(|v| matches!(v, Value::Number(_))) {
                    items.sort_by(|a, b| match (a, b) {
                        (Value::Number(x), Value::Number(y)) => {
                            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        _ => std::cmp::Ordering::Equal,
                    });
                } else if items.iter().all(|v| matches!(v, Value::String(_))) {
                    items.sort_by(|a, b| match (a, b) {
                        (Value::String(x), Value::String(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    });
                } else {
                    return Err(throw_err(
                        "TypeError",
                        "Sort needs all numbers or all strings",
                    ));
                }
                Ok(Some(Value::Array(items)))
            }
            _ => Err(throw_err("ValueError", "Sort expects exactly one array")),
        },
        "Join" => match resolved.as_slice() {
            [arr, sep] => {
                let (items, sep) = (expect_array(arr, "Join")?, expect_string(sep, "Join")?);
                let mut parts = Vec::with_capacity(items.len());
                for item in &items {
                    parts.push(stringify_value(item));
                }
                Ok(Some(Value::String(parts.join(sep.as_str()))))
            }
            _ => Err(throw_err("ValueError", "Join expects (array, separator)")),
        },
        "Keys" => match resolved.as_slice() {
            [Value::Object(map)] => {
                let mut keys: Vec<Value> = map.keys().cloned().map(Value::String).collect();
                keys.sort_by(|a, b| match (a, b) {
                    (Value::String(x), Value::String(y)) => x.cmp(y),
                    _ => std::cmp::Ordering::Equal,
                });
                Ok(Some(Value::Array(keys)))
            }
            [other] => Err(throw_err(
                "TypeError",
                format!("Keys expects an object, got {:?}", other),
            )),
            _ => Err(throw_err("ValueError", "Keys expects exactly one object")),
        },
        "Has" => match resolved.as_slice() {
            [Value::Array(items), needle] => Ok(Some(Value::Logic(items.contains(needle)))),
            [Value::Object(map), Value::String(key)] => {
                Ok(Some(Value::Logic(map.contains_key(key))))
            }
            [container, _] => Err(throw_err(
                "TypeError",
                format!("Has expects (array, value) or (object, key), got {:?}", container),
            )),
            _ => Err(throw_err("ValueError", "Has expects two arguments")),
        },
        "Get" => match resolved.as_slice() {
            [arr, idx] => {
                let (items, idx) = (expect_array(arr, "Get")?, expect_int(idx, "Get")?);
                if idx < 0 {
                    return Err(throw_err("IndexError", format!("Index out of bounds: {}", idx)));
                }
                items.get(idx as usize).cloned().ok_or_else(|| {
                    throw_err("IndexError", format!("Index out of bounds: {}", idx))
                }).map(Some)
            }
            _ => Err(throw_err("ValueError", "Get expects (array, index)")),
        },
        "Slice" => match resolved.as_slice() {
            [arr, start] => {
                let (items, start) = (expect_array(arr, "Slice")?, expect_int(start, "Slice")?);
                slice_array(items, start, None).map(Value::Array).map(Some)
            }
            [arr, start, len] => {
                let (items, start, len) = (
                    expect_array(arr, "Slice")?,
                    expect_int(start, "Slice")?,
                    expect_int(len, "Slice")?,
                );
                slice_array(items, start, Some(len)).map(Value::Array).map(Some)
            }
            _ => Err(throw_err("ValueError", "Slice expects (array, start[, length])")),
        },
        _ => Err(throw_err(
            "NameError",
            format!("Unknown arr function: {}", function),
        )),
    }
}

fn slice_array(
    items: Vec<Value>,
    start: i64,
    len: Option<i64>,
) -> Result<Vec<Value>, RuntimeFault> {
    if start < 0 {
        return Err(throw_err("ValueError", "Slice start must be >= 0"));
    }
    if let Some(len) = len {
        if len < 0 {
            return Err(throw_err("ValueError", "Slice length must be >= 0"));
        }
    }
    let start = (start as usize).min(items.len());
    let end = match len {
        Some(len) => (start + len as usize).min(items.len()),
        None => items.len(),
    };
    Ok(items[start..end].to_vec())
}

fn invoke_function(
    object: Option<&str>,
    module: Option<&str>,
    function: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Option<Value>, RuntimeFault> {
    if let Some(alias) = module {
        // Cross-module call `alias:foo(args)`: resolve in the callee
        // module's live globals, run against its merged tables so the
        // callee sees its own top-level locals + globals (never the
        // caller's scope). `arg.*`/`pass` stay call-local as usual.
        if object.is_some() {
            return Err(fatal_err("module call cannot have an object"));
        }
        if !rt.modules.contains_key(alias) {
            return Err(throw_err(
                "NameError",
                format!("Unknown module: {}", alias),
            ));
        }
        let callee = rt
            .read_global(alias, function)
            .ok_or_else(|| {
                throw_err(
                    "NameError",
                    format!("Unknown variable: {}:var.{}", alias, function),
                )
            })?;
        match callee {
            Value::Function {
                params,
                defaults,
                body,
            } => {
                let _guard = enter_call(rt)?;
                let mut local_env = rt.module_merged_env(alias);
                let mut local_types = rt.module_merged_types(alias);
                let argument_array = Value::Array(
                    args.iter()
                        .map(|arg| resolve_value(arg, environment, types, rt, cur))
                        .collect::<Result<Vec<_>, RuntimeFault>>()?,
                );
                local_env.insert("pass".to_string(), Value::Null);
                local_env.insert("arg".to_string(), argument_array.clone());
                local_env.insert("args".to_string(), argument_array);
                bind_params(&params, &defaults, args, environment, types, rt, cur, function, &mut local_env)?;
                let mut result = None;
                for stmt in body {
                    match execute_statement(&stmt, &mut local_env, &mut local_types, rt, alias)? {
                        Flow::Normal => {}
                        Flow::Break => {
                            return Err(fatal_err("break outside loop"));
                        }
                        Flow::Continue => {
                            return Err(fatal_err("continue outside loop"));
                        }
                        Flow::Return(value) => {
                            result = Some(value);
                            break;
                        }
                    }
                }
                Ok(result)
            }
            _ => Err(throw_err(
                "TypeError",
                format!("{}:{} is not callable", alias, function),
            )),
        }
    } else {
    match object {
        Some(obj) if obj == "con" && function == "Print" => {
            let mut rendered = Vec::new();
            for arg in args {
                let value = resolve_value(arg, environment, types, rt, cur)?;
                rendered.push(format_value(value));
            }
            println!("{}", rendered.join(" "));
            Ok(None)
        }
        Some(obj) if obj == "con" && function == "Input" => {
            let prompt = match args {
                [] => String::new(),
                [arg] => match resolve_value(arg, environment, types, rt, cur)? {
                    Value::String(s) => s,
                    other => format_value(other),
                },
                _ => {
                    return Err(throw_err(
                        "ValueError",
                        "Input expects zero or one argument",
                    ))
                }
            };
            if !prompt.is_empty() {
                print!("{}", prompt);
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
            }
            let mut line = String::new();
            std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut line)
                .map_err(|e| throw_err("IOError", format!("Input failed: {}", e)))?;
            Ok(Some(Value::String(
                line.trim_end_matches(&['\n', '\r'][..]).to_string(),
            )))
        }
        Some(obj) if obj == "math" => invoke_math(function, args, environment, types, rt, cur),
        Some(obj) if obj == "str" => invoke_str(function, args, environment, types, rt, cur),
        Some(obj) if obj == "arr" => invoke_arr(function, args, environment, types, rt, cur),
        Some(obj) if obj == "time" => {
            if !args.is_empty() {
                return Err(throw_err("ValueError", "Now expects no arguments"));
            }
            match function {
                "Now" => {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    let millis = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_err(|e| throw_err("IOError", format!("clock failed: {}", e)))?
                        .as_millis();
                    #[allow(clippy::cast_precision_loss)]
                    Ok(Some(Value::Number(millis as f64)))
                }
                _ => Err(throw_err(
                    "NameError",
                    format!("Unknown time function: {}", function),
                )),
            }
        }
        Some(obj) if obj == "file" => {
            let resolved: Vec<Value> = args
                .iter()
                .map(|arg| resolve_value(arg, environment, types, rt, cur))
                .collect::<Result<Vec<_>, RuntimeFault>>()?;
            match function {
                "Read" => {
                    let path = match resolved.as_slice() {
                        [Value::String(path)] => path.clone(),
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Read expects exactly one path string",
                            ))
                        }
                    };
                    fs::read_to_string(&path)
                        .map(Value::String)
                        .map(Some)
                        .map_err(|e| throw_err("IOError", format!("cannot read '{}': {}", path, e)))
                }
                "Write" => {
                    let (path, content) = match resolved.as_slice() {
                        [Value::String(path), content] => (path.clone(), stringify_value(content)),
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Write expects a path string and content",
                            ))
                        }
                    };
                    fs::write(&path, content)
                        .map(|()| Some(Value::Null))
                        .map_err(|e| throw_err("IOError", format!("cannot write '{}': {}", path, e)))
                }
                _ => Err(throw_err(
                    "NameError",
                    format!("Unknown file function: {}", function),
                )),
            }
        }
        _ => {
            let passed_value = match object {
                Some(obj) => resolve_value(&Value::Variable(obj.to_string()), environment, types, rt, cur)
                    .or_else(|_| lookup_scoped(obj, environment, rt, cur).ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", obj))))?,
                None => Value::Null,
            };

            if function.eq_ignore_ascii_case("EqualsCaseSensitive")
                || function.eq_ignore_ascii_case("caseSensitiveEquals")
                || function.eq_ignore_ascii_case("CaseSensitiveEquals")
            {
                let target = match object {
                    Some(obj) => resolve_value(&Value::Variable(obj.to_string()), environment, types, rt, cur)
                        .or_else(|_| lookup_scoped(obj, environment, rt, cur).ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", obj))))?,
                    None => return Err(throw_err("ValueError", "case-sensitive equality requires a target value")),
                };

                let rhs = match args {
                    [value] => resolve_value(value, environment, types, rt, cur)?,
                    _ => return Err(throw_err("ValueError", "case-sensitive equality expects exactly one argument")),
                };

                match (target, rhs) {
                    (Value::String(lhs), Value::String(rhs)) => return Ok(Some(Value::Logic(lhs == rhs))),
                    _ => return Err(throw_err("TypeError", "case-sensitive equality requires two strings")),
                }
            }

            let callee = resolve_value(&Value::Variable(function.to_string()), environment, types, rt, cur)
                .or_else(|_| {
                    lookup_scoped(function, environment, rt, cur).ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", function)))
                })?;

            match callee {
                Value::Function {
                    params,
                    defaults,
                    body,
                } => {
                    let _guard = enter_call(rt)?;
                    let mut local_env = environment.clone();
                    let mut local_types = types.clone();
                    let argument_array = Value::Array(
                        args.iter()
                            .map(|arg| resolve_value(arg, environment, types, rt, cur))
                            .collect::<Result<Vec<_>, RuntimeFault>>()?,
                    );
                    local_env.insert("pass".to_string(), passed_value);
                    local_env.insert("arg".to_string(), argument_array.clone());
                    local_env.insert("args".to_string(), argument_array);
                    bind_params(&params, &defaults, args, environment, types, rt, cur, function, &mut local_env)?;

                    let mut result = None;
                    for stmt in body {
                        match execute_statement(&stmt, &mut local_env, &mut local_types, rt, cur)? {
                            Flow::Normal => {}
                            Flow::Break => {
                                return Err(fatal_err("break outside loop"));
                            }
                            Flow::Continue => {
                                return Err(fatal_err("continue outside loop"));
                            }
                            Flow::Return(value) => {
                                result = Some(value);
                                break;
                            }
                        }
                    }
                    Ok(result)
                }
                _ => Err(throw_err("TypeError", format!("{} is not callable", function))),
            }
        }
    }
    }
}

/// Enforce a declared slot type. `null` passes every type; anything else
/// must match the discriminant. Mismatches are catchable `TypeError`s.
fn check_type(slot: &str, type_name: &str, value: &Value) -> Result<(), RuntimeFault> {
    if matches!(value, Value::Null) {
        return Ok(());
    }
    let ok = match (type_name, value) {
        ("string", Value::String(_)) => true,
        ("number", Value::Number(_)) => true,
        ("logic", Value::Logic(_)) => true,
        ("null", Value::Null) => true,
        ("array", Value::Array(_)) => true,
        ("object", Value::Object(_)) => true,
        ("function", Value::Function { .. }) => true,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(throw_err(
            "TypeError",
            format!(
                "expected {} for var.{}, got {}",
                type_name,
                slot,
                value_type_name(value)
            ),
        ))
    }
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Logic(_) => "logic",
        Value::Variable(_) => "variable",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
        Value::Index { .. } => "index",
        Value::FunctionCall { .. } => "function-call",
        Value::ModuleVar { .. } => "module-var",
        Value::Binary { .. } => "expression",
        Value::Property { .. } => "property",
        Value::Unary { .. } => "expression",
        Value::Function { .. } => "function",
        Value::Error { .. } => "error",
    }
}

fn execute_statement(
    statement: &Statement,
    environment: &mut HashMap<String, Value>,
    types: &mut HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Flow, RuntimeFault> {
    match statement {
        Statement::VariableDecl { name, type_name, value } => {
            // Redeclare-as-assign: `var local x <type> = v` overwrites `var.x`
            // if it already exists, otherwise creates it. Redeclare resets
            // the recorded type; mismatched values are a catchable TypeError
            // (`null` always passes).
            let resolved = match value {
                Value::Function { .. } => value.clone(),
                _ => resolve_value(value, environment, types, rt, cur)?,
            };
            check_type(name, type_name, &resolved)?;
            environment.insert(format!("var.{}", name), resolved);
            types.insert(format!("var.{}", name), type_name.clone());
            Ok(Flow::Normal)
        }
        Statement::FunctionCall { object, module, function, args } => {
            let _ = invoke_function(object.as_deref(), module.as_deref(), function, args, environment, types, rt, cur)?;
            Ok(Flow::Normal)
        }
        Statement::GlobalDecl { name, type_name, value } => {
            // `var global x <type> = v` writes the live module table (and
            // the local scope). Redeclare overwrites + resets the type.
            // Allowed in any block so module functions can mutate globals.
            let resolved = match value {
                Value::Function { .. } => value.clone(),
                _ => resolve_value(value, environment, types, rt, cur)?,
            };
            check_type(name, type_name, &resolved)?;
            environment.insert(format!("var.{}", name), resolved.clone());
            types.insert(format!("var.{}", name), type_name.clone());
            rt.ensure_module(cur);
            let module = rt.modules.get_mut(cur).unwrap();
            module.globals.insert(format!("var.{}", name), resolved);
            module.global_names.insert(name.clone());
            module.global_types.insert(name.clone(), type_name.clone());
            Ok(Flow::Normal)
        }
        Statement::Declare { .. } => {
            // Handled by the loader before execution; no-op here.
            Ok(Flow::Normal)
        }
        Statement::Assign { target, op, value } => {
            let new_value = resolve_value(value, environment, types, rt, cur)?;
            let final_value = match op {
                None => {
                    // Plain `=` requires the target to exist (declare first).
                    read_assign_target(target, environment, types, rt, cur)?;
                    new_value
                }
                Some(binop) => {
                    let current = read_assign_target(target, environment, types, rt, cur)?;
                    evaluate_binary(current, new_value, binop)?
                }
            };
            write_assign_target(target, final_value, environment, types, rt, cur)?;
            Ok(Flow::Normal)
        }
        Statement::Switch {
            scrutinee,
            cases,
            default,
        } => {
            let subject = resolve_value(scrutinee, environment, types, rt, cur)?;
            for (case_value, body) in cases {
                let expected = resolve_value(case_value, environment, types, rt, cur)?;
                if values_strict_equal(&subject, &expected) {
                    return execute_block(body, environment, types, rt, cur);
                }
            }
            if let Some(default_body) = default {
                return execute_block(default_body, environment, types, rt, cur);
            }
            Ok(Flow::Normal)
        }
        Statement::Return { value } => {
            let resolved = resolve_value(value, environment, types, rt, cur)?;
            Ok(Flow::Return(resolved))
        }
        Statement::Break => Ok(Flow::Break),
        Statement::Continue => Ok(Flow::Continue),
        Statement::Throw { error_type, message } => {
            let resolved_msg = resolve_value(message, environment, types, rt, cur)?;
            let text = match resolved_msg {
                Value::String(s) => s,
                other => format_value(other),
            };
            Err(throw_err(error_type, text))
        }
        Statement::Try { body, catches } => execute_try(body, catches, environment, types, rt, cur),
        Statement::Event { object, name, body } => {
            if object == "kal" && name == "OnStart" {
                match execute_block(body, environment, types, rt, cur)? {
                    Flow::Normal => Ok(Flow::Normal),
                    Flow::Break => Err(fatal_err("break outside loop")),
                    Flow::Continue => Err(fatal_err("continue outside loop")),
                    Flow::Return(_) => {
                        Err(fatal_err("return can only be used inside a function body"))
                    }
                }
            } else {
                Ok(Flow::Normal)
            }
        }
        Statement::If {
            condition,
            then_branch,
            else_if_branches,
            else_branch,
        } => {
            let condition_value = resolve_value(condition, environment, types, rt, cur)?;
            if is_truthy(&condition_value) {
                return execute_block(then_branch, environment, types, rt, cur);
            }

            for (else_if_condition, else_if_body) in else_if_branches {
                let branch_value = resolve_value(else_if_condition, environment, types, rt, cur)?;
                if is_truthy(&branch_value) {
                    return execute_block(else_if_body, environment, types, rt, cur);
                }
            }

            if let Some(else_body) = else_branch {
                return execute_block(else_body, environment, types, rt, cur);
            }
            Ok(Flow::Normal)
        }
        Statement::While { condition, body } => {
            for _ in 0..MAX_LOOP_ITERS {
                let condition_value = resolve_value(condition, environment, types, rt, cur)?;
                if !is_truthy(&condition_value) {
                    return Ok(Flow::Normal);
                }
                match execute_block(body, environment, types, rt, cur)? {
                    Flow::Normal => {}
                    Flow::Break => return Ok(Flow::Normal),
                    Flow::Continue => {}
                    Flow::Return(value) => return Ok(Flow::Return(value)),
                }
            }
            Err(fatal_err("possible infinite loop: while exceeded iteration limit"))
        }
        Statement::ForIn { var, iterable, body } => {
            // `for (i in 0..10)` iterates 0..=9 (end-exclusive). Bounds
            // truncate to i64; empty when start >= end.
            if let Value::Binary {
                left,
                op: BinaryOperator::Range,
                right,
            } = iterable
            {
                let start = match resolve_value(left, environment, types, rt, cur)? {
                    Value::Number(n) => as_i64(n)?,
                    other => {
                        return Err(throw_err(
                            "TypeError",
                            format!("range bounds must be numbers, got {:?}", other),
                        ))
                    }
                };
                let end = match resolve_value(right, environment, types, rt, cur)? {
                    Value::Number(n) => as_i64(n)?,
                    other => {
                        return Err(throw_err(
                            "TypeError",
                            format!("range bounds must be numbers, got {:?}", other),
                        ))
                    }
                };
                let key = format!("var.{}", var);
                let saved = environment.get(&key).cloned();
                let saved_global = if rt.is_global(cur, var) {
                    rt.read_global(cur, var)
                } else {
                    None
                };
                let shadows_global = rt.is_global(cur, var);
                let mut i = start;
                while i < end {
                    let item = Value::Number(i as f64);
                    environment.insert(key.clone(), item.clone());
                    if shadows_global {
                        rt.ensure_module(cur);
                        rt.modules
                            .get_mut(cur)
                            .unwrap()
                            .globals
                            .insert(key.clone(), item);
                    }
                    match execute_block(body, environment, types, rt, cur)? {
                        Flow::Normal => {}
                        Flow::Break => break,
                        Flow::Continue => {}
                        Flow::Return(value) => {
                            restore_saved(environment, &key, saved);
                            restore_global(rt, cur, &key, saved_global);
                            return Ok(Flow::Return(value));
                        }
                    }
                    i += 1;
                }
                restore_saved(environment, &key, saved);
                restore_global(rt, cur, &key, saved_global);
                return Ok(Flow::Normal);
            }
            let resolved_iterable = resolve_value(iterable, environment, types, rt, cur)?;
            let items: Vec<Value> = match resolved_iterable {
                Value::Array(items) => items,
                Value::String(s) => s.chars().map(|ch| Value::String(ch.to_string())).collect(),
                other => {
                    return Err(throw_err(
                        "TypeError",
                        format!("for-in requires an array or string, got {:?}", other),
                    ))
                }
            };
            // `for (x in ...)` binds `var.x` each iteration (var. prefix;
            // arg. stays function-only). Restore outer value afterwards.
            // When the loop name shadows a `var global`, the live store
            // slot is shadowed too and restored after the loop.
            let key = format!("var.{}", var);
            let saved = environment.get(&key).cloned();
            let saved_global = if rt.is_global(cur, var) {
                rt.read_global(cur, var)
            } else {
                None
            };
            let shadows_global = rt.is_global(cur, var);
            for item in items {
                environment.insert(key.clone(), item.clone());
                if shadows_global {
                    rt.ensure_module(cur);
                    rt.modules
                        .get_mut(cur)
                        .unwrap()
                        .globals
                        .insert(key.clone(), item);
                }
                match execute_block(body, environment, types, rt, cur)? {
                    Flow::Normal => {}
                    Flow::Break => break,
                    Flow::Continue => {}
                    Flow::Return(value) => {
                        restore_saved(environment, &key, saved);
                        restore_global(rt, cur, &key, saved_global);
                        return Ok(Flow::Return(value));
                    }
                }
            }
            restore_saved(environment, &key, saved);
            restore_global(rt, cur, &key, saved_global);
            Ok(Flow::Normal)
        }
    }
}

fn execute_try(
    body: &[Statement],
    catches: &[CatchClause],
    environment: &mut HashMap<String, Value>,
    types: &mut HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Flow, RuntimeFault> {
    match execute_block(body, environment, types, rt, cur) {
        Ok(flow) => Ok(flow),
        Err(RuntimeFault::Throw(err_value)) => {
            let thrown_type = match &err_value {
                Value::Error { error_type, .. } => error_type.clone(),
                _ => "Error".to_string(),
            };
            let handler = catches
                .iter()
                .find(|c| c.error_type == thrown_type || c.error_type == "Error");
            match handler {
                Some(catch) => {
                    // Bind `var.err` for the handler (var. prefix; arg. untouched).
                    // Shadow the live global slot too when `err` is global.
                    let key = "var.err".to_string();
                    let saved = environment.get(&key).cloned();
                    let saved_global = if rt.is_global(cur, "err") {
                        rt.read_global(cur, "err")
                    } else {
                        None
                    };
                    let shadows_global = rt.is_global(cur, "err");
                    environment.insert(key.clone(), err_value.clone());
                    if shadows_global {
                        rt.ensure_module(cur);
                        rt.modules
                            .get_mut(cur)
                            .unwrap()
                            .globals
                            .insert(key.clone(), err_value);
                    }
                    let result = execute_block(&catch.body, environment, types, rt, cur);
                    restore_saved(environment, &key, saved);
                    restore_global(rt, cur, &key, saved_global);
                    result
                }
                None => Err(RuntimeFault::Throw(err_value)),
            }
        }
        Err(other) => Err(other),
    }
}

fn restore_saved(
    environment: &mut HashMap<String, Value>,
    key: &str,
    saved: Option<Value>,
) {
    match saved {
        Some(value) => {
            environment.insert(key.to_string(), value);
        }
        None => {
            environment.remove(key);
        }
    }
}

fn restore_global(rt: &mut ModuleRuntime, alias: &str, key: &str, saved: Option<Value>) {
    if !rt.modules.contains_key(alias) {
        return;
    }
    let module = rt.modules.get_mut(alias).unwrap();
    match saved {
        Some(value) => {
            module.globals.insert(key.to_string(), value);
        }
        None => {
            module.globals.remove(key);
        }
    }
}

/// Strict (`===`) equality for `switch/case`: case-sensitive strings,
/// no cross-type coercion.
fn values_strict_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Logic(x), Value::Logic(y)) => x == y,
        (Value::Null, Value::Null) => true,
        (Value::Array(x), Value::Array(y)) => x == y,
        (Value::Object(x), Value::Object(y)) => x == y,
        (Value::Error { error_type: t1, message: m1 }, Value::Error { error_type: t2, message: m2 }) => {
            t1 == t2 && m1 == m2
        }
        _ => false,
    }
}

/// Read the current value at an assignment target. Missing names,
/// properties, or indices are catchable errors.
fn read_assign_target(
    target: &AssignTarget,
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Value, RuntimeFault> {
    match target {
        AssignTarget::Var(name) => lookup_scoped(name, environment, rt, cur).ok_or_else(|| {
            throw_err(
                "NameError",
                format!("Cannot assign to undeclared variable: var.{}", name),
            )
        }),
        AssignTarget::Property { target: inner, key } => {
            let container = read_assign_target(inner, environment, types, rt, cur)?;
            match container {
                Value::Object(map) => map.get(key).cloned().ok_or_else(|| {
                    throw_err("NameError", format!("Unknown property: {}", key))
                }),
                other => Err(throw_err(
                    "TypeError",
                    format!("Property access requires an object, got {:?}", other),
                )),
            }
        }
        AssignTarget::Index { target: inner, index } => {
            let container = read_assign_target(inner, environment, types, rt, cur)?;
            let idx_value = resolve_value(index, environment, types, rt, cur)?;
            let idx = match idx_value {
                Value::Number(n)
                    if n.is_finite() && n.fract() == 0.0 && n >= 0.0 =>
                {
                    n as usize
                }
                _ => {
                    return Err(throw_err(
                        "TypeError",
                        "index must be a non-negative integer",
                    ))
                }
            };
            match container {
                Value::Array(items) => items.get(idx).cloned().ok_or_else(|| {
                    throw_err("IndexError", format!("Index out of bounds: {}", idx))
                }),
                Value::String(value) => value
                    .chars()
                    .nth(idx)
                    .map(|ch| Value::String(ch.to_string()))
                    .ok_or_else(|| {
                        throw_err("IndexError", format!("Index out of bounds: {}", idx))
                    }),
                other => Err(throw_err(
                    "TypeError",
                    format!("Index requires an array or string, got {:?}", other),
                )),
            }
        }
    }
}

/// Write a value at an assignment target. Containers are cloned,
/// mutated, and written back to the root `var.*` (live store included).
fn write_assign_target(
    target: &AssignTarget,
    new_value: Value,
    environment: &mut HashMap<String, Value>,
    types: &mut HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<(), RuntimeFault> {
    match target {
        AssignTarget::Var(name) => {
            let key = format!("var.{}", name);
            if lookup_scoped(name, environment, rt, cur).is_none() {
                return Err(throw_err(
                    "NameError",
                    format!("Cannot assign to undeclared variable: var.{}", name),
                ));
            }
            // Enforce the slot's declared type (if any).
            if let Some(declared) = types
                .get(&key)
                .cloned()
                .or_else(|| rt.lookup_type(cur, &key))
            {
                check_type(name, &declared, &new_value)?;
            }
            environment.insert(key.clone(), new_value.clone());
            if rt.is_global(cur, name) {
                rt.ensure_module(cur);
                rt.modules
                    .get_mut(cur)
                    .unwrap()
                    .globals
                    .insert(key, new_value);
            }
            Ok(())
        }
        AssignTarget::Property { target: inner, key } => {
            let mut container = read_assign_target(inner, environment, types, rt, cur)?;
            match &mut container {
                Value::Object(map) => {
                    map.insert(key.clone(), new_value);
                }
                other => {
                    return Err(throw_err(
                        "TypeError",
                        format!("Property access requires an object, got {:?}", other),
                    ))
                }
            }
            write_assign_target(inner, container, environment, types, rt, cur)
        }
        AssignTarget::Index { target: inner, index } => {
            let idx_value = resolve_value(index, environment, types, rt, cur)?;
            let idx = match idx_value {
                Value::Number(n) if n.is_finite() && n.fract() == 0.0 && n >= 0.0 => n as usize,
                _ => {
                    return Err(throw_err(
                        "TypeError",
                        "index must be a non-negative integer",
                    ))
                }
            };
            let mut container = read_assign_target(inner, environment, types, rt, cur)?;
            match &mut container {
                Value::Array(items) => {
                    if idx >= items.len() {
                        return Err(throw_err(
                            "IndexError",
                            format!("Index out of bounds: {}", idx),
                        ));
                    }
                    items[idx] = new_value;
                }
                Value::String(_) => {
                    return Err(throw_err(
                        "TypeError",
                        "cannot assign into string characters",
                    ))
                }
                other => {
                    return Err(throw_err(
                        "TypeError",
                        format!("Index requires an array or string, got {:?}", other),
                    ))
                }
            }
            write_assign_target(inner, container, environment, types, rt, cur)
        }
    }
}

/// Own-file `var.<short>` read: the live global store wins when the name
/// was declared `var global`, otherwise the local scope.
fn lookup_scoped(
    short: &str,
    environment: &HashMap<String, Value>,
    rt: &ModuleRuntime,
    cur: &str,
) -> Option<Value> {
    if rt.is_global(cur, short) {
        if let Some(value) = rt.read_global(cur, short) {
            return Some(value);
        }
    }
    environment.get(&format!("var.{}", short)).cloned()
}

fn resolve_variable_name(
    name: &str,
    environment: &HashMap<String, Value>,
    rt: &ModuleRuntime,
    cur: &str,
) -> Option<Value> {
    if name == "pass" {
        return environment.get("pass").cloned();
    }

    if name == "arg" || name == "args" {
        return environment.get(name).cloned();
    }

    if let Some(short) = name.strip_prefix("var.") {
        return lookup_scoped(short, environment, rt, cur);
    }

    if name.starts_with("arg.") {
        return environment.get(name).cloned();
    }

    None
}

fn resolve_value(
    value: &Value,
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Value, RuntimeFault> {
    match value {
        Value::Variable(name) => resolve_variable_name(name, environment, rt, cur)
            .ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", name))),
        Value::ModuleVar { alias, name } => {
            let key = format!("var.{}", name);
            if !rt.modules.contains_key(alias) {
                return Err(throw_err("NameError", format!("Unknown module: {}", alias)));
            }
            rt.modules[alias].globals.get(&key).cloned().ok_or_else(|| {
                throw_err(
                    "NameError",
                    format!("Unknown variable: {}:var.{}", alias, name),
                )
            })
        }
        Value::FunctionCall { object, module, function, args } => {
            let result = invoke_function(object.as_deref(), module.as_deref(), function, args, environment, types, rt, cur)?;
            match result {
                Some(value) => Ok(value),
                None => Ok(Value::Null),
            }
        }
        Value::Array(items) => Ok(Value::Array(
            items
                .iter()
                .map(|item| resolve_value(item, environment, types, rt, cur))
                .collect::<Result<Vec<_>, RuntimeFault>>()?,
        )),
        Value::Object(map) => {
            let mut resolved = HashMap::new();
            for (key, item) in map {
                resolved.insert(key.clone(), resolve_value(item, environment, types, rt, cur)?);
            }
            Ok(Value::Object(resolved))
        }
        Value::Property { target, key } => {
            let target_value = resolve_value(target, environment, types, rt, cur)?;
            match target_value {
                Value::Object(map) => map.get(key).cloned().ok_or_else(|| throw_err("NameError", format!("Unknown property: {}", key))),
                Value::Error { error_type, message } => match key.as_str() {
                    "type" => Ok(Value::String(error_type)),
                    "message" => Ok(Value::String(message)),
                    _ => Err(throw_err("NameError", format!("Unknown property: {}", key))),
                },
                Value::Null => Ok(Value::Null),
                other => Err(throw_err("TypeError", format!("Property access requires an object, got {:?}", other))),
            }
        }
        Value::Index { target, index } => {
            let target_value = resolve_value(target, environment, types, rt, cur)?;
            let index_value = resolve_value(index, environment, types, rt, cur)?;
            let idx = match index_value {
                Value::Number(n) => {
                    if !n.is_finite() || n.fract() != 0.0 || n < 0.0 {
                        return Err(throw_err(
                            "TypeError",
                            format!("index must be a non-negative integer, got {}", n),
                        ));
                    }
                    n as usize
                }
                other => {
                    return Err(throw_err(
                        "TypeError",
                        format!("Index requires an array or string with a numeric index, got {:?}", other),
                    ))
                }
            };
            match target_value {
                Value::Array(items) => items.get(idx).cloned().ok_or_else(|| {
                    throw_err("IndexError", format!("Index out of bounds: {}", idx))
                }),
                Value::String(value) => {
                    let ch = value
                        .chars()
                        .nth(idx)
                        .ok_or_else(|| throw_err("IndexError", format!("Index out of bounds: {}", idx)))?;
                    Ok(Value::String(ch.to_string()))
                }
                other => Err(throw_err(
                    "TypeError",
                    format!("Index requires an array or string, got {:?}", other),
                )),
            }
        }
        Value::Binary { left, op, right } => {
            let left_value = resolve_value(left, environment, types, rt, cur)?;
            let right_value = resolve_value(right, environment, types, rt, cur)?;
            evaluate_binary(left_value, right_value, op)
        }
        Value::Unary { op, value } => {
            let inner = resolve_value(value, environment, types, rt, cur)?;
            evaluate_unary(inner, op)
        }
        _ => Ok(value.clone()),
    }
}

fn evaluate_unary(value: Value, op: &UnaryOperator) -> Result<Value, RuntimeFault> {
    match op {
        UnaryOperator::Not => Ok(Value::Logic(!is_truthy(&value))),
        UnaryOperator::BitNot => match value {
            Value::Number(n) => Ok(Value::Number((!as_i64(n)?) as f64)),
            Value::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::Number(n) => out.push(Value::Number((!as_i64(n)?) as f64)),
                        other => {
                            return Err(throw_err(
                                "TypeError",
                                format!("~ requires numbers, got {:?}", other),
                            ))
                        }
                    }
                }
                Ok(Value::Array(out))
            }
            other => Err(throw_err(
                "TypeError",
                format!("~ requires a number, got {:?}", other),
            )),
        },
    }
}

/// Truncate f64 toward zero to i64 for bitwise ops. Non-finite and
/// out-of-range values are catchable errors, not silent garbage.
fn as_i64(n: f64) -> Result<i64, RuntimeFault> {
    if !n.is_finite() {
        return Err(throw_err(
            "ValueError",
            format!("bitwise requires a finite integer, got {}", n),
        ));
    }
    if n < i64::MIN as f64 || n >= 9.223372036854776e18 {
        return Err(throw_err(
            "ValueError",
            format!("bitwise operand out of i64 range: {}", n),
        ));
    }
    Ok(n.trunc() as i64)
}

fn check_shift_amount(n: f64) -> Result<u32, RuntimeFault> {
    let amount = as_i64(n)?;
    if amount < 0 || amount >= 64 {
        return Err(throw_err(
            "ValueError",
            format!("shift amount must be 0..64, got {}", n),
        ));
    }
    Ok(amount as u32)
}

fn evaluate_binary(left: Value, right: Value, op: &BinaryOperator) -> Result<Value, RuntimeFault> {
    match op {
        BinaryOperator::Add => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
            (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            (Value::String(a), Value::Number(b)) => Ok(Value::String(format!("{}{}", a, b))),
            (Value::Number(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            _ => Err(throw_err("TypeError", "Addition requires numbers or strings")),
        },
        BinaryOperator::Subtract => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a - b)),
            _ => Err(throw_err("TypeError", "Subtraction requires numbers")),
        },
        BinaryOperator::Multiply => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a * b)),
            _ => Err(throw_err("TypeError", "Multiplication requires numbers")),
        },
        BinaryOperator::Divide => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) if b != 0.0 => Ok(Value::Number(a / b)),
            _ => Err(throw_err("DivZero", "Division requires non-zero numbers")),
        },
        BinaryOperator::BitAnd | BinaryOperator::BitOr | BinaryOperator::BitXor => {
            match (left, right) {
                (Value::Array(left_items), Value::Array(right_items)) => {
                    apply_array_array_op(left_items, right_items, op)
                }
                (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
                (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
                (Value::Number(a), Value::Number(b)) => {
                    let (x, y) = (as_i64(a)?, as_i64(b)?);
                    let result = match op {
                        BinaryOperator::BitAnd => x & y,
                        BinaryOperator::BitOr => x | y,
                        _ => x ^ y,
                    };
                    Ok(Value::Number(result as f64))
                }
                _ => Err(throw_err("TypeError", "bitwise requires numbers")),
            }
        }
        BinaryOperator::Shl | BinaryOperator::Shr => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => {
                apply_array_array_op(left_items, right_items, op)
            }
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) => {
                let x = as_i64(a)?;
                let amount = check_shift_amount(b)?;
                let result = match op {
                    BinaryOperator::Shl => x.wrapping_shl(amount),
                    _ => x.wrapping_shr(amount),
                };
                Ok(Value::Number(result as f64))
            }
            _ => Err(throw_err("TypeError", "shifts require numbers")),
        },
        BinaryOperator::Range => Err(throw_err(
            "TypeError",
            "ranges (..) can only be iterated with for-in",
        )),
        BinaryOperator::Equal => match (&left, &right) {
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a.eq_ignore_ascii_case(b))),
            (Value::String(_), Value::Null) => Ok(Value::Logic(false)),
            (Value::Null, Value::String(_)) => Ok(Value::Logic(false)),
            _ => Ok(Value::Logic(left == right)),
        },
        BinaryOperator::NotEqual => match (&left, &right) {
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(!a.eq_ignore_ascii_case(b))),
            _ => Ok(Value::Logic(left != right)),
        },
        BinaryOperator::StrictEqual => Ok(Value::Logic(left == right)),
        BinaryOperator::StrictNotEqual => Ok(Value::Logic(left != right)),
        BinaryOperator::And => Ok(Value::Logic(is_truthy(&left) && is_truthy(&right))),
        BinaryOperator::Or => Ok(Value::Logic(is_truthy(&left) || is_truthy(&right))),
        BinaryOperator::Greater => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a > b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a > b)),
            _ => Err(throw_err("TypeError", "Greater-than requires comparable values")),
        },
        BinaryOperator::Less => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a < b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a < b)),
            _ => Err(throw_err("TypeError", "Less-than requires comparable values")),
        },
        BinaryOperator::GreaterEqual => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a >= b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a >= b)),
            _ => Err(throw_err("TypeError", "Greater-or-equal requires comparable values")),
        },
        BinaryOperator::LessEqual => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a <= b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a <= b)),
            _ => Err(throw_err("TypeError", "Less-or-equal requires comparable values")),
        },
    }
}

fn apply_array_scalar_op(items: Vec<Value>, scalar: Value, op: &BinaryOperator) -> Result<Value, RuntimeFault> {
    // Bitwise ops broadcast over truncated ints; arithmetic over f64.
    if matches!(
        op,
        BinaryOperator::BitAnd
            | BinaryOperator::BitOr
            | BinaryOperator::BitXor
            | BinaryOperator::Shl
            | BinaryOperator::Shr
    ) {
        let is_shift = matches!(op, BinaryOperator::Shl | BinaryOperator::Shr);
        let scalar_int = as_i64(match scalar {
            Value::Number(value) => value,
            _ => {
                return Err(throw_err(
                    "TypeError",
                    format!("Array math requires a numeric scalar, got {:?}", scalar),
                ))
            }
        })?;
        let shift_amount = if is_shift {
            check_shift_amount(scalar_int as f64)?
        } else {
            0
        };
        let mut result = Vec::with_capacity(items.len());
        for item in items {
            let current = match item {
                Value::Number(value) => as_i64(value)?,
                _ => return Err(throw_err("TypeError", "Array math only works on numeric arrays")),
            };
            let transformed = match op {
                BinaryOperator::BitAnd => current & scalar_int,
                BinaryOperator::BitOr => current | scalar_int,
                BinaryOperator::BitXor => current ^ scalar_int,
                BinaryOperator::Shl => current.wrapping_shl(shift_amount),
                BinaryOperator::Shr => current.wrapping_shr(shift_amount),
                _ => return Err(throw_err("TypeError", "Unsupported array operation")),
            };
            result.push(Value::Number(transformed as f64));
        }
        return Ok(Value::Array(result));
    }
    let scalar_number = match scalar {
        Value::Number(value) => value,
        _ => return Err(throw_err("TypeError", format!("Array math requires a numeric scalar, got {:?}", scalar))),
    };

    let mut result = Vec::new();
    for item in items {
        let current = match item {
            Value::Number(value) => value,
            _ => return Err(throw_err("TypeError", "Array math only works on numeric arrays")),
        };

        let transformed = match op {
            BinaryOperator::Add => current + scalar_number,
            BinaryOperator::Subtract => current - scalar_number,
            BinaryOperator::Multiply => current * scalar_number,
            BinaryOperator::Divide if scalar_number != 0.0 => current / scalar_number,
            BinaryOperator::Divide => return Err(throw_err("DivZero", "Division by zero in array math")),
            _ => return Err(throw_err("TypeError", "Unsupported array operation")),
        };
        result.push(Value::Number(transformed));
    }

    Ok(Value::Array(result))
}

fn apply_array_array_op(left_items: Vec<Value>, right_items: Vec<Value>, op: &BinaryOperator) -> Result<Value, RuntimeFault> {
    let max_len = left_items.len().max(right_items.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let left_value = left_items.get(i % left_items.len()).cloned().unwrap_or_else(|| Value::Number(0.0));
        let right_value = right_items.get(i % right_items.len()).cloned().unwrap_or_else(|| Value::Number(0.0));

        let next = match evaluate_binary(left_value, right_value, op)? {
            Value::Number(value) => Value::Number(value),
            other => other,
        };
        result.push(next);
    }

    Ok(Value::Array(result))
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Logic(value) => *value,
        Value::Number(value) => *value != 0.0,
        Value::String(value) => !value.is_empty(),
        Value::Null => false,
        Value::Error { .. } => true,
        _ => true,
    }
}

fn format_value(value: Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v,
        Value::Logic(v) => v.to_string(),
        Value::Variable(v) => v,
        Value::Error { error_type, message } => format!("{}: {}", error_type, message),
        Value::Array(items) => {
            let rendered: Vec<String> = items.into_iter().map(format_value).collect();
            format!("[{}]", rendered.join(", "))
        }
        Value::FunctionCall { .. } => "<function-call>".to_string(),
        Value::ModuleVar { .. } => "<module-var>".to_string(),
        Value::Index { .. } => "<index>".to_string(),
        Value::Binary { .. } => "<expression>".to_string(),
        Value::Property { .. } => "<property>".to_string(),
        Value::Object(object) => {
            let rendered: Vec<String> = object
                .iter()
                .map(|(key, value)| format!("{}: {}", key, format_value(value.clone())))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
        Value::Unary { .. } => "<expression>".to_string(),
        Value::Function { .. } => "<function>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{Lexer, Token};

    const TEST_ALIAS: &str = "__test__";

    fn eval(value: &Value, env: &HashMap<String, Value>) -> Result<Value, RuntimeFault> {
        let mut rt = ModuleRuntime::default();
        resolve_value(value, env, &HashMap::new(), &mut rt, TEST_ALIAS)
    }

    fn call(
        object: Option<&str>,
        function: &str,
        args: &[Value],
        env: &HashMap<String, Value>,
    ) -> Result<Option<Value>, RuntimeFault> {
        let mut rt = ModuleRuntime::default();
        invoke_function(object, None, function, args, env, &HashMap::new(), &mut rt, TEST_ALIAS)
    }

    fn run_stmt(stmt: &Statement, env: &mut HashMap<String, Value>) {
        execute_statement(
            stmt,
            env,
            &mut HashMap::new(),
            &mut ModuleRuntime::default(),
            TEST_ALIAS,
        )
        .unwrap();
    }

    fn try_stmt(
        stmt: &Statement,
        env: &mut HashMap<String, Value>,
    ) -> Result<Flow, RuntimeFault> {
        execute_statement(
            stmt,
            env,
            &mut HashMap::new(),
            &mut ModuleRuntime::default(),
            TEST_ALIAS,
        )
    }

    #[test]
    fn parses_header_and_kal_event() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local wow string = \"Hello world\"\n    con.Print(wow)\n}\n";

        let program = Parser::parse(source).unwrap();
        assert_eq!(program.header.version, 1);
        assert_eq!(program.header.script_type, "SCRIPTTYPE");
        assert_eq!(program.header.language, "KALVITA");
        assert!(matches!(
            &program.statements[0],
            Statement::Event { object, name, .. } if object == "kal" && name == "OnStart"
        ));
    }

    #[test]
    fn parses_null_variable_and_literal() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local empty null = null\n    con.Print(empty)\n}\n";

        let program = Parser::parse(source).unwrap();
        assert!(matches!(
            &program.statements[0],
            Statement::Event { object, name, .. } if object == "kal" && name == "OnStart"
        ));
    }

    #[test]
    fn parses_function_as_variable_and_call() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local add function = (a, b) {\n        con.Print(a, b)\n    }\n    add(\"hi\", 42)\n}\n";

        let program = Parser::parse(source).unwrap();
        assert!(matches!(
            &program.statements[0],
            Statement::Event { object, name, .. } if object == "kal" && name == "OnStart"
        ));
    }

    #[test]
    fn parses_math_and_if_else_if_else_blocks() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local score number = 7\n    var local active logic = true\n    if (score > 5) {\n        con.Print(\"big\")\n    } elseif (active == true) {\n        con.Print(\"active\")\n    } else {\n        con.Print(\"small\")\n    }\n}\n";

        let program = Parser::parse(source).unwrap();
        assert!(matches!(
            &program.statements[0],
            Statement::Event { object, name, .. } if object == "kal" && name == "OnStart"
        ));

        let if_stmt = match &program.statements[0] {
            Statement::Event { body, .. } => body.iter().find(|stmt| matches!(stmt, Statement::If { .. })).unwrap(),
            _ => panic!("expected event body"),
        };

        assert!(matches!(
            if_stmt,
            Statement::If { else_if_branches, .. } if !else_if_branches.is_empty()
        ));
    }

    #[test]
    fn redeclaring_variable_updates_value() {
        let mut environment = HashMap::new();

        let first = Statement::VariableDecl {
            name: "score".to_string(),
            type_name: "number".to_string(),
            value: Value::Number(10.0),
        };
        let second = Statement::VariableDecl {
            name: "score".to_string(),
            type_name: "number".to_string(),
            value: Value::Number(15.0),
        };

        run_stmt(&first, &mut environment);
        run_stmt(&second, &mut environment);

        assert_eq!(environment.get("var.score"), Some(&Value::Number(15.0)));
    }

    #[test]
    fn function_call_used_as_value_returns_result() {
        let mut environment = HashMap::new();
        environment.insert(
            "var.add".to_string(),
            Value::Function {
                params: vec!["a".to_string(), "b".to_string()],
                defaults: HashMap::new(),
                body: vec![Statement::Return {
                    value: Box::new(Value::Binary {
                        left: Box::new(Value::Variable("arg.a".to_string())),
                        op: BinaryOperator::Add,
                        right: Box::new(Value::Variable("arg.b".to_string())),
                    }),
                }],
            },
        );

        let result = eval(
            &Value::FunctionCall {
                object: None,
                module: None,
                function: "add".to_string(),
                args: vec![Value::Number(15.0), Value::Number(2.0)],
            },
            &environment,
        )
        .unwrap();

        assert_eq!(result, Value::Number(17.0));
    }

    #[test]
    fn namespaced_var_and_arg_access_parse_and_resolve() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local score number = 15\n    var local add function = (a, b) {\n        return(arg.a + arg.b)\n    }\n    con.Print(var.score)\n    con.Print(add(15 + 2))\n}\n";

        let program = Parser::parse(source).unwrap();
        assert!(matches!(
            &program.statements[0],
            Statement::Event { object, name, .. } if object == "kal" && name == "OnStart"
        ));

        let mut environment = HashMap::new();
        environment.insert("var.score".to_string(), Value::Number(15.0));
        environment.insert("arg.a".to_string(), Value::Number(7.0));
        environment.insert("arg.b".to_string(), Value::Number(2.0));

        let resolved = eval(&Value::Variable("var.score".to_string()), &environment).unwrap();
        assert_eq!(resolved, Value::Number(15.0));
        let arg_a = eval(&Value::Variable("arg.a".to_string()), &environment).unwrap();
        assert_eq!(arg_a, Value::Number(7.0));
    }

    #[test]
    fn naked_variables_are_rejected() {
        let mut environment = HashMap::new();
        environment.insert("var.score".to_string(), Value::Number(15.0));

        let result = eval(&Value::Variable("score".to_string()), &environment);
        assert!(result.is_err());
    }

    #[test]
    fn logic_gate_operations_work() {
        let mut environment = HashMap::new();
        environment.insert("var.active".to_string(), Value::Logic(true));
        environment.insert("var.ready".to_string(), Value::Logic(true));
        environment.insert("var.blocked".to_string(), Value::Logic(false));

        let and_result = eval(
            &Value::Binary {
                left: Box::new(Value::Variable("var.active".to_string())),
                op: BinaryOperator::And,
                right: Box::new(Value::Variable("var.ready".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(and_result, Value::Logic(true));

        let or_result = eval(
            &Value::Binary {
                left: Box::new(Value::Variable("var.blocked".to_string())),
                op: BinaryOperator::Or,
                right: Box::new(Value::Variable("var.active".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(or_result, Value::Logic(true));

        let not_result = eval(
            &Value::Unary {
                op: UnaryOperator::Not,
                value: Box::new(Value::Variable("var.blocked".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(not_result, Value::Logic(true));
    }

    #[test]
    fn trigonometric_functions_work() {
        let environment = HashMap::new();

        let sin_result = call(Some("math"), "Sin", &[Value::Number(0.0)], &environment).unwrap().unwrap();
        match sin_result {
            Value::Number(value) => assert!((value - 0.0).abs() < 1e-9),
            _ => panic!("Sin should return a number"),
        }

        let cos_result = call(Some("math"), "Cos", &[Value::Number(0.0)], &environment).unwrap().unwrap();
        match cos_result {
            Value::Number(value) => assert!((value - 1.0).abs() < 1e-9),
            _ => panic!("Cos should return a number"),
        }

        let tan_result = call(Some("math"), "Tan", &[Value::Number(0.0)], &environment).unwrap().unwrap();
        match tan_result {
            Value::Number(value) => assert!((value - 0.0).abs() < 1e-9),
            _ => panic!("Tan should return a number"),
        }
    }

    #[test]
    fn object_values_and_pass_work() {
        let mut environment = HashMap::new();
        environment.insert(
            "var.enemy".to_string(),
            Value::Object(HashMap::from([
                ("health".to_string(), Value::Number(40.0)),
                ("damage".to_string(), Value::Number(5.0)),
                ("speed".to_string(), Value::Number(10.0)),
            ])),
        );
        environment.insert(
            "var.attack".to_string(),
            Value::Function {
                params: vec![],
                defaults: HashMap::new(),
                body: vec![Statement::Return {
                    value: Box::new(Value::Property {
                        target: Box::new(Value::Variable("pass".to_string())),
                        key: "health".to_string(),
                    }),
                }],
            },
        );

        let result = call(Some("enemy"), "attack", &[], &environment).unwrap().unwrap();
        assert_eq!(result, Value::Number(40.0));

        let null_result = call(None, "attack", &[], &environment).unwrap().unwrap();
        assert_eq!(null_result, Value::Null);
    }

    #[test]
    fn namespaced_function_calls_parse_as_method_calls() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local enemy object = { health: 40, damage: 5 }\n    con.Print(math.Sin(0))\n    enemy.attack()\n}\n";

        let program = Parser::parse(source).unwrap();
        assert!(matches!(
            &program.statements[0],
            Statement::Event { object, name, .. } if object == "kal" && name == "OnStart"
        ));
    }

    #[test]
    fn case_insensitive_and_sensitive_string_equality_work() {
        let mut environment = HashMap::new();
        environment.insert("var.name".to_string(), Value::String("Hello".to_string()));

        let insensitive_equal = eval(
            &Value::Binary {
                left: Box::new(Value::Variable("var.name".to_string())),
                op: BinaryOperator::Equal,
                right: Box::new(Value::String("hello".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(insensitive_equal, Value::Logic(true));

        let sensitive_equal = eval(
            &Value::Binary {
                left: Box::new(Value::Variable("var.name".to_string())),
                op: BinaryOperator::StrictEqual,
                right: Box::new(Value::String("hello".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(sensitive_equal, Value::Logic(false));

        let insensitive_not_equal = eval(
            &Value::Binary {
                left: Box::new(Value::Variable("var.name".to_string())),
                op: BinaryOperator::NotEqual,
                right: Box::new(Value::String("world".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(insensitive_not_equal, Value::Logic(true));

        let sensitive_not_equal = eval(
            &Value::Binary {
                left: Box::new(Value::Variable("var.name".to_string())),
                op: BinaryOperator::StrictNotEqual,
                right: Box::new(Value::String("Hello".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(sensitive_not_equal, Value::Logic(false));
    }

    #[test]
    fn function_arguments_are_stored_as_array() {
        let mut environment = HashMap::new();
        environment.insert("var.score".to_string(), Value::Number(42.0));
        environment.insert(
            "var.doSomething".to_string(),
            Value::Function {
                params: vec!["a".to_string()],
                defaults: HashMap::new(),
                body: vec![Statement::Return {
                    value: Box::new(Value::Variable("arg".to_string())),
                }],
            },
        );

        let result = eval(
            &Value::FunctionCall {
                object: None,
                module: None,
                function: "doSomething".to_string(),
                args: vec![
                    Value::String("arg1".to_string()),
                    Value::Variable("var.score".to_string()),
                    Value::Number(3.0),
                    Value::Array(vec![
                        Value::String("you can".to_string()),
                        Value::String("also have".to_string()),
                    ]),
                ],
            },
            &environment,
        )
        .unwrap();

        assert_eq!(
            result,
            Value::Array(vec![
                Value::String("arg1".to_string()),
                Value::Number(42.0),
                Value::Number(3.0),
                Value::Array(vec![
                    Value::String("you can".to_string()),
                    Value::String("also have".to_string()),
                ]),
            ])
        );
    }

    #[test]
    fn array_numeric_math_broadcasts_and_string_arrays_error() {
        let nums = Value::Array(vec![
            Value::Number(12.0),
            Value::Number(64.0),
            Value::Number(9.0),
            Value::Number(747.0),
        ]);

        let result = evaluate_binary(nums.clone(), Value::Number(10.0), &BinaryOperator::Add).unwrap();
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Number(22.0),
                Value::Number(74.0),
                Value::Number(19.0),
                Value::Number(757.0),
            ])
        );

        let longer = Value::Array(vec![
            Value::Number(12.0),
            Value::Number(42.0),
            Value::Number(6.0),
            Value::Number(2.0),
        ]);
        let shorter = Value::Array(vec![
            Value::Number(11.0),
            Value::Number(6.0),
        ]);
        let repeated = evaluate_binary(longer, shorter, &BinaryOperator::Add).unwrap();
        assert_eq!(
            repeated,
            Value::Array(vec![
                Value::Number(23.0),
                Value::Number(48.0),
                Value::Number(17.0),
                Value::Number(8.0),
            ])
        );

        let strings = Value::Array(vec![
            Value::String("apple".to_string()),
            Value::String("banana".to_string()),
        ]);
        assert!(evaluate_binary(strings, Value::Number(10.0), &BinaryOperator::Add).is_err());
    }

    #[test]
    fn array_indexing_and_comparison_work() {
        let mut environment = HashMap::new();
        environment.insert("var.items".to_string(), Value::Array(vec![
            Value::String("apple".to_string()),
            Value::String("banana".to_string()),
            Value::String("orange".to_string()),
        ]));

        let first = eval(&Value::Index {
            target: Box::new(Value::Variable("var.items".to_string())),
            index: Box::new(Value::Number(0.0)),
        }, &environment).unwrap();
        assert_eq!(first, Value::String("apple".to_string()));

        let compare = evaluate_binary(
            Value::Number(15.0),
            Value::Number(10.0),
            &BinaryOperator::Greater,
        ).unwrap();
        assert_eq!(compare, Value::Logic(true));
    }

    #[test]
    fn bitwise_truth_table_and_shifts() {
        let cases = vec![
            ("6 & 3", 2.0),
            ("6 | 3", 7.0),
            ("6 ^ 3", 5.0),
            ("~0", -1.0),
            ("~5", -6.0),
            ("1 << 10", 1024.0),
            ("(0 - 8) >> 2", -2.0),
            ("7.9 & 3.1", 3.0),
            ("2 + 1 << 2", 12.0),
            ("15 ^ 1 ^ 1", 15.0),
        ];
        for (source, expected) in cases {
            let full = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local r number = {}\n}}\n",
                source
            );
            let program = Parser::parse(&full).unwrap();
            let body = match &program.statements[0] {
                Statement::Event { body, .. } => body.clone(),
                _ => panic!("expected event"),
            };
            let mut env = HashMap::new();
            run_stmt(&body[0], &mut env);
            assert_eq!(
                env.get("var.r"),
                Some(&Value::Number(expected)),
                "failed for {}",
                source
            );
        }
    }

    #[test]
    fn bitwise_errors_are_catchable() {
        let env = HashMap::new();
        // Non-number operand.
        assert!(matches!(
            eval(
                &Value::Binary {
                    left: Box::new(Value::String("a".to_string())),
                    op: BinaryOperator::BitAnd,
                    right: Box::new(Value::Number(1.0)),
                },
                &env,
            ),
            Err(RuntimeFault::Throw(_))
        ));
        // Negative shift.
        assert!(matches!(
            eval(
                &Value::Binary {
                    left: Box::new(Value::Number(1.0)),
                    op: BinaryOperator::Shl,
                    right: Box::new(Value::Number(-1.0)),
                },
                &env,
            ),
            Err(RuntimeFault::Throw(_))
        ));
        // Bitwise broadcasts over arrays.
        let broadcast = eval(
            &Value::Binary {
                left: Box::new(Value::Array(vec![
                    Value::Number(12.0),
                    Value::Number(10.0),
                ])),
                op: BinaryOperator::BitAnd,
                right: Box::new(Value::Number(10.0)),
            },
            &env,
        )
        .unwrap();
        assert_eq!(
            broadcast,
            Value::Array(vec![Value::Number(8.0), Value::Number(10.0)])
        );
    }

    #[test]
    fn strict_equality_tokens_lex_and_parse() {
        let mut lexer = Lexer::new("a === b !== c");
        let tokens = lexer.tokenize();
        assert!(tokens.contains(&Token::StrictEqual));
        assert!(tokens.contains(&Token::StrictNotEqual));
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local r logic = \"Hi\" === \"hi\"\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        run_stmt(&body[0], &mut env);
        assert_eq!(env.get("var.r"), Some(&Value::Logic(false)));
    }

    #[test]
    fn assignment_and_compound_ops() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local x number = 10\n    var.x = 15\n    var.x += 5\n    var.x *= 2\n    var local pair array = [1, 2, 3]\n    var.pair[0] = 99\n    var local hero object = { health: 40 }\n    var.hero.health = 50\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        for stmt in &body {
            run_stmt(stmt, &mut env);
        }
        assert_eq!(env.get("var.x"), Some(&Value::Number(40.0)));
        assert_eq!(
            env.get("var.pair"),
            Some(&Value::Array(vec![
                Value::Number(99.0),
                Value::Number(2.0),
                Value::Number(3.0),
            ]))
        );
        match env.get("var.hero") {
            Some(Value::Object(map)) => {
                assert_eq!(map.get("health"), Some(&Value::Number(50.0)))
            }
            other => panic!("expected hero object, got {:?}", other),
        }
    }

    #[test]
    fn assign_to_undeclared_is_name_error() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var.nope = 1\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        assert!(matches!(
            try_stmt(&body[0], &mut env),
            Err(RuntimeFault::Throw(Value::Error { error_type, .. })) if error_type == "NameError"
        ));
    }

    #[test]
    fn float_index_is_type_error() {
        let mut rt = ModuleRuntime::default();
        let mut env = HashMap::new();
        env.insert(
            "var.items".to_string(),
            Value::Array(vec![Value::Number(1.0)]),
        );
        let err = resolve_value(
            &Value::Index {
                target: Box::new(Value::Variable("var.items".to_string())),
                index: Box::new(Value::Number(1.9)),
            },
            &env,
            &HashMap::new(),
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "TypeError"
        ));
    }

    #[test]
    fn switch_first_match_and_default() {
        for (value, expected) in [
            (Value::Number(2.0), "two"),
            (Value::Number(9.0), "other"),
            (Value::String("Hi".to_string()), "other"),
        ] {
            let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    switch (var.v) {\n        case (1) { var local r string = \"one\" }\n        case (2) { var local r string = \"two\" }\n        default { var local r string = \"other\" }\n    }\n}\n";
            let program = Parser::parse(source).unwrap();
            let body = match &program.statements[0] {
                Statement::Event { body, .. } => body.clone(),
                _ => panic!("expected event"),
            };
            let mut env = HashMap::new();
            env.insert("var.v".to_string(), value);
            for stmt in &body {
                run_stmt(stmt, &mut env);
            }
            assert_eq!(
                env.get("var.r"),
                Some(&Value::String(expected.to_string()))
            );
        }
    }

    #[test]
    fn range_for_iterates_and_range_value_errors() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local total number = 0\n    for (i in 0..5) {\n        var local total number = var.total + var.i\n    }\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        for stmt in &body {
            run_stmt(stmt, &mut env);
        }
        assert_eq!(env.get("var.total"), Some(&Value::Number(10.0)));
        // Bare range as a value is a TypeError outside for-in.
        let env = HashMap::new();
        assert!(matches!(
            eval(
                &Value::Binary {
                    left: Box::new(Value::Number(0.0)),
                    op: BinaryOperator::Range,
                    right: Box::new(Value::Number(3.0)),
                },
                &env,
            ),
            Err(RuntimeFault::Throw(_))
        ));
    }

    #[test]
    fn string_interpolation_concats() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local name string = \"Nova\"\n    var local msg string = \"hi ${var.name}!\"\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        for stmt in &body {
            run_stmt(stmt, &mut env);
        }
        assert_eq!(
            env.get("var.msg"),
            Some(&Value::String("hi Nova!".to_string()))
        );
    }

    #[test]
    fn default_params_fill_and_missing_required_errors() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local greet function = (name = \"World\") {\n        return(arg.name)\n    }\n    var local a string = greet()\n    var local b string = greet(\"Nova\")\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        for stmt in &body {
            run_stmt(stmt, &mut env);
        }
        assert_eq!(
            env.get("var.a"),
            Some(&Value::String("World".to_string()))
        );
        assert_eq!(env.get("var.b"), Some(&Value::String("Nova".to_string())));
        // Missing required param.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local f function = (x) {\n        return(arg.x)\n    }\n    var local r number = f()\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        let mut failed = false;
        for stmt in &body {
            if try_stmt(stmt, &mut env).is_err() {
                failed = true;
            }
        }
        assert!(failed);
    }

    #[test]
    fn infinite_recursion_is_fatal() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local loop function = () {\n        return(loop())\n    }\n    var local r number = loop()\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        let mut saw_fatal = false;
        for stmt in &body {
            if let Err(RuntimeFault::Fatal(_)) = try_stmt(stmt, &mut env) {
                saw_fatal = true;
            }
        }
        assert!(saw_fatal);
    }

    #[test]
    fn stdlib_spot_checks() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local a string = str.Upper(\"hey\")\n    var local b array = str.Split(\"x,y\", \",\")\n    var local c number = arr.Len(var.b)\n    var local d number = math.Sqrt(16)\n    var local e number = math.Clamp(99, 0, 10)\n    var local f array = arr.Sort([3, 1, 2])\n    var local g string = arr.Join(var.f, \"-\")\n    var local h logic = str.Contains(\"hello\", \"ell\")\n}\n";
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut env = HashMap::new();
        for stmt in &body {
            run_stmt(stmt, &mut env);
        }
        assert_eq!(env.get("var.a"), Some(&Value::String("HEY".to_string())));
        assert_eq!(env.get("var.c"), Some(&Value::Number(2.0)));
        assert_eq!(env.get("var.d"), Some(&Value::Number(4.0)));
        assert_eq!(env.get("var.e"), Some(&Value::Number(10.0)));
        assert_eq!(
            env.get("var.f"),
            Some(&Value::Array(vec![
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(3.0),
            ]))
        );
        assert_eq!(
            env.get("var.g"),
            Some(&Value::String("1-2-3".to_string()))
        );
        assert_eq!(env.get("var.h"), Some(&Value::Logic(true)));
    }

    #[test]
    fn parses_global_decl_and_declare() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\ndeclare(\"./math.kal\")\ndeclare(\"./quiet.kal\", false)\nvar global share number = 53\nkal.OnStart {\n    con.Print(var.share)\n}\n";

        let program = Parser::parse(source).unwrap();
        assert!(matches!(
            &program.statements[0],
            Statement::Declare { path, run } if path == "./math.kal" && *run
        ));
        assert!(matches!(
            &program.statements[1],
            Statement::Declare { path, run } if path == "./quiet.kal" && !*run
        ));
        assert!(matches!(
            &program.statements[2],
            Statement::GlobalDecl { name, .. } if name == "share"
        ));
    }

    #[test]
    fn parses_module_access_and_call() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    con.Print(math:var.share)\n    con.Print(math:double(21))\n}\n";

        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event body"),
        };
        assert!(matches!(
            &body[0],
            Statement::FunctionCall { object, function, args, .. }
                if object.as_deref() == Some("con")
                    && function == "Print"
                    && matches!(&args[0], Value::ModuleVar { alias, name } if alias == "math" && name == "share")
        ));
        assert!(matches!(
            &body[1],
            Statement::FunctionCall { object, function, args, .. }
                if object.as_deref() == Some("con")
                    && matches!(&args[0], Value::FunctionCall { module, function, .. } if module.as_deref() == Some("math") && function == "double")
        ));
    }

    #[test]
    fn declare_must_be_top_level() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    declare(\"./math.kal\")\n}\n";
        assert!(Parser::parse(source).is_err());
    }

    #[test]
    fn catch_must_be_inside_try_still_rejected() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    catch (Error) {\n        con.Print(\"x\")\n    }\n}\n";
        assert!(Parser::parse(source).is_err());
    }

    fn module_test_runtime() -> ModuleRuntime {
        let mut rt = ModuleRuntime::default();
        rt.modules.insert(
            "math".to_string(),
            LoadedModule {
                path: PathBuf::from("math.kal"),
                top_locals: HashMap::new(),
                top_local_types: HashMap::new(),
                globals: HashMap::from([
                    ("var.share".to_string(), Value::Number(53.0)),
                    (
                        "var.double".to_string(),
                        Value::Function {
                            params: vec!["x".to_string()],
                            defaults: HashMap::new(),
                            body: vec![Statement::Return {
                                value: Box::new(Value::Binary {
                                    left: Box::new(Value::Variable("arg.x".to_string())),
                                    op: BinaryOperator::Multiply,
                                    right: Box::new(Value::Number(2.0)),
                                }),
                            }],
                        },
                    ),
                ]),
                global_names: HashSet::from(["share".to_string(), "double".to_string()]),
                global_types: HashMap::from([
                    ("share".to_string(), "number".to_string()),
                    ("double".to_string(), "function".to_string()),
                ]),
                program: None,
                onstart_executed: true,
            },
        );
        rt
    }

    #[test]
    fn module_var_reads_live_globals() {
        let mut rt = module_test_runtime();
        let env = HashMap::new();
        let value = resolve_value(
            &Value::ModuleVar {
                alias: "math".to_string(),
                name: "share".to_string(),
            },
            &env,
            &HashMap::new(),
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap();
        assert_eq!(value, Value::Number(53.0));
    }

    #[test]
    fn unknown_module_is_name_error() {
        let mut rt = ModuleRuntime::default();
        let env = HashMap::new();
        let err = resolve_value(
            &Value::ModuleVar {
                alias: "nope".to_string(),
                name: "share".to_string(),
            },
            &env,
            &HashMap::new(),
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "NameError"
        ));
    }

    #[test]
    fn module_function_call_runs_against_module_tables() {
        let mut rt = module_test_runtime();
        let env = HashMap::new();
        let value = resolve_value(
            &Value::FunctionCall {
                object: None,
                module: Some("math".to_string()),
                function: "double".to_string(),
                args: vec![Value::Number(21.0)],
            },
            &env,
            &HashMap::new(),
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap();
        assert_eq!(value, Value::Number(42.0));
    }

    #[test]
    fn global_decl_writes_live_store_and_local_reads_win() {
        let mut rt = ModuleRuntime::default();
        let mut env = HashMap::new();
        let mut types = HashMap::new();
        let decl = Statement::GlobalDecl {
            name: "share".to_string(),
            type_name: "number".to_string(),
            value: Value::Number(53.0),
        };
        execute_statement(&decl, &mut env, &mut types, &mut rt, TEST_ALIAS).unwrap();
        // Live store read.
        let via_store = resolve_value(
            &Value::ModuleVar {
                alias: TEST_ALIAS.to_string(),
                name: "share".to_string(),
            },
            &HashMap::new(),
            &HashMap::new(),
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap();
        assert_eq!(via_store, Value::Number(53.0));
        // Plain var read prefers the live global too.
        let via_plain = resolve_value(
            &Value::Variable("var.share".to_string()),
            &env,
            &types,
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap();
        assert_eq!(via_plain, Value::Number(53.0));
        // A later global redeclare updates both views.
        let redecl = Statement::GlobalDecl {
            name: "share".to_string(),
            type_name: "number".to_string(),
            value: Value::Number(99.0),
        };
        execute_statement(&redecl, &mut env, &mut types, &mut rt, TEST_ALIAS).unwrap();
        let updated = resolve_value(
            &Value::Variable("var.share".to_string()),
            &env,
            &types,
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap();
        assert_eq!(updated, Value::Number(99.0));
    }

    #[test]
    fn unknown_type_names_are_rejected() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local x banana = 1\n}\n";
        assert!(Parser::parse(source).is_err());
    }

    fn run_with_types(source: &str) -> (HashMap<String, Value>, Option<String>) {
        let program = Parser::parse(source).unwrap();
        let body = match &program.statements[0] {
            Statement::Event { body, .. } => body.clone(),
            _ => panic!("expected event"),
        };
        let mut rt = ModuleRuntime::default();
        let mut env = HashMap::new();
        let mut types = HashMap::new();
        for stmt in &body {
            match execute_statement(stmt, &mut env, &mut types, &mut rt, TEST_ALIAS) {
                Ok(_) => {}
                Err(RuntimeFault::Fatal(msg)) => return (env, Some(format!("fatal: {}", msg))),
                Err(RuntimeFault::Throw(Value::Error { error_type, message })) => {
                    return (env, Some(format!("{}: {}", error_type, message)))
                }
                Err(RuntimeFault::Throw(other)) => return (env, Some(format!("throw: {:?}", other))),
            }
        }
        (env, None)
    }

    #[test]
    fn mismatched_declaration_is_type_error() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local n number = \"huh\"\n}\n";
        let (_, err) = run_with_types(source);
        let err = err.expect("should fail");
        assert!(err.starts_with("TypeError"), "got: {}", err);
        // Same via try/catch in-language.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    try {\n        var local n number = \"huh\"\n        catch (TypeError) {\n            var local caught string = var.err.message\n        }\n    }\n    con.Print(var.caught)\n}\n";
        let (env, err) = run_with_types(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert!(matches!(env.get("var.caught"), Some(Value::String(_))));
    }

    #[test]
    fn all_seven_types_accept_and_reject() {
        let good = vec![
            ("string", "\"s\""),
            ("number", "1"),
            ("logic", "true"),
            ("null", "null"),
            ("array", "[1]"),
            ("object", "{ a: 1 }"),
            ("function", "(x) { return(arg.x) }"),
        ];
        for (ty, expr) in good {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local v {} = {}\n}}\n",
                ty, expr
            );
            let (_, err) = run_with_types(&source);
            assert!(err.is_none(), "{} = {} failed: {:?}", ty, expr, err);
        }
        let bad = vec![
            ("string", "1"),
            ("number", "\"s\""),
            ("logic", "1"),
            ("array", "{ a: 1 }"),
            ("object", "[1]"),
            ("function", "1"),
        ];
        for (ty, expr) in bad {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local v {} = {}\n}}\n",
                ty, expr
            );
            let (_, err) = run_with_types(&source);
            let err = err.expect("should fail");
            assert!(err.starts_with("TypeError"), "{} = {}: got {}", ty, expr, err);
        }
    }

    #[test]
    fn null_passes_every_type_and_redeclare_resets() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local s string = null\n    var local n number = 1\n    var local n string = \"now a string\"\n    var local n number = 2\n}\n";
        let (env, err) = run_with_types(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(env.get("var.s"), Some(&Value::Null));
        assert_eq!(env.get("var.n"), Some(&Value::Number(2.0)));
    }

    #[test]
    fn assign_and_compound_respect_slot_types() {
        // Plain assign of wrong type fails; compound result is checked too.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local x number = 1\n    var.x = \"s\"\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.unwrap().starts_with("TypeError"));
        // Index assign into a number slot fails (root must stay consistent).
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local xs array = [1]\n    var local xs number = 5\n}\n";
        let (env, err) = run_with_types(source);
        assert!(err.is_none(), "redeclare resets: {:?}", err);
        assert_eq!(env.get("var.xs"), Some(&Value::Number(5.0)));
    }
}
