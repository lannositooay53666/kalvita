//! Kalvita tree-walk interpreter: modules, execution, builtins.
use crate::error::{RuntimeFault, fatal_err, throw_err, uncaught_message};
use crate::parser::{
    AssignTarget, BinaryOperator, CatchClause, GuiKind, Header, Parser, Program, Statement,
    UnaryOperator, Value,
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

/// A widget registry entry. Entries are pure data until `Run` — no
/// display touch, so declaration/attach/position/event code is
/// headless-testable. `handlers` maps event name (`OnStart`, `OnExit`,
/// `OnClick`) to callbacks in registration order; blocks and
/// `OnClick(fn)` share the list.
#[derive(Debug, Clone)]
struct GuiEntry {
    kind: GuiKind,
    title: String,
    parent: Option<u64>,
    children: Vec<u64>,
    pos: Option<(i32, i32)>,
    handlers: HashMap<String, Vec<Value>>,
    closed: bool,
}

impl GuiEntry {
    fn window(title: String) -> Self {
        Self {
            kind: GuiKind::Window,
            title,
            parent: None,
            children: Vec::new(),
            pos: None,
            handlers: HashMap::new(),
            closed: false,
        }
    }

    fn button(text: String) -> Self {
        Self {
            kind: GuiKind::Button,
            title: text,
            parent: None,
            children: Vec::new(),
            pos: None,
            handlers: HashMap::new(),
            closed: false,
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
    db_next: u64,
    db_conns: HashMap<u64, Rc<rusqlite::Connection>>,
    gui_next: u64,
    gui: HashMap<u64, GuiEntry>,
    gui_running: bool,
}

impl Default for ModuleRuntime {
    fn default() -> Self {
        Self {
            modules: HashMap::new(),
            canonical_to_alias: HashMap::new(),
            loading: Vec::new(),
            call_depth: Rc::new(Cell::new(0)),
            db_next: 1,
            db_conns: HashMap::new(),
            gui_next: 1,
            gui: HashMap::new(),
            gui_running: false,
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
        "Lines" => match resolved.as_slice() {
            [v] => Ok(Some(Value::Array(
                expect_string(v, "Lines")?
                    .lines()
                    .map(|l| Value::String(l.to_string()))
                    .collect(),
            ))),
            _ => Err(throw_err("ValueError", "Lines expects exactly one string")),
        },
        "Chars" => match resolved.as_slice() {
            [v] => Ok(Some(Value::Array(
                expect_string(v, "Chars")?
                    .chars()
                    .map(|c| Value::String(c.to_string()))
                    .collect(),
            ))),
            _ => Err(throw_err("ValueError", "Chars expects exactly one string")),
        },
        "StartsWith" | "EndsWith" => match resolved.as_slice() {
            [s, affix] => {
                let (s, affix) = (expect_string(s, function)?, expect_string(affix, function)?);
                Ok(Some(Value::Logic(if function == "StartsWith" {
                    s.starts_with(affix.as_str())
                } else {
                    s.ends_with(affix.as_str())
                })))
            }
            _ => Err(throw_err(
                "ValueError",
                format!("{} expects (string, affix)", function),
            )),
        },
        "TrimPrefix" | "TrimSuffix" => match resolved.as_slice() {
            [s, affix] => {
                let (s, affix) = (expect_string(s, function)?, expect_string(affix, function)?);
                Ok(Some(Value::String(
                    if function == "TrimPrefix" {
                        s.strip_prefix(affix.as_str()).unwrap_or(&s).to_string()
                    } else {
                        s.strip_suffix(affix.as_str()).unwrap_or(&s).to_string()
                    },
                )))
            }
            _ => Err(throw_err(
                "ValueError",
                format!("{} expects (string, affix)", function),
            )),
        },
        "Repeat" => match resolved.as_slice() {
            [s, n] => {
                let (s, n) = (expect_string(s, "Repeat")?, expect_int(n, "Repeat")?);
                if n < 0 {
                    return Err(throw_err("ValueError", "Repeat count must be >= 0"));
                }
                const MAX_REPEAT_CHARS: u64 = 10_000_000;
                let chars = s.chars().count() as u64;
                if chars.saturating_mul(n as u64) > MAX_REPEAT_CHARS {
                    return Err(throw_err(
                        "ValueError",
                        "Repeat result would exceed 10M characters",
                    ));
                }
                Ok(Some(Value::String(s.repeat(n as usize))))
            }
            _ => Err(throw_err("ValueError", "Repeat expects (string, count)")),
        },
        "ParseInt" => match resolved.as_slice() {
            [v] => {
                let s = expect_string(v, "ParseInt")?;
                s.trim().parse::<i64>().map(|n| Some(Value::Number(n as f64))).map_err(|_| {
                    throw_err("ValueError", format!("cannot convert to integer: '{}'", s))
                })
            }
            _ => Err(throw_err("ValueError", "ParseInt expects exactly one string")),
        },
        "ParseFloat" => match resolved.as_slice() {
            [v] => {
                let s = expect_string(v, "ParseFloat")?;
                s.trim()
                    .parse::<f64>()
                    .map(Value::Number)
                    .map(Some)
                    .map_err(|_| {
                        throw_err("ValueError", format!("cannot convert to number: '{}'", s))
                    })
            }
            _ => Err(throw_err("ValueError", "ParseFloat expects exactly one string")),
        },
        "Match" => match resolved.as_slice() {
            [s, pat] => {
                let (s, pat) = (expect_string(s, "Match")?, expect_string(pat, "Match")?);
                glob_match(&pat, &s).map(Value::Logic).map(Some)
            }
            _ => Err(throw_err("ValueError", "Match expects (string, pattern)")),
        },
        _ => Err(throw_err(
            "NameError",
            format!("Unknown str function: {}", function),
        )),
    }
}

/// Blocking HTTP via ureq (30s global timeout). Transports and
/// non-2xx statuses surface as catchable `IOError`.
fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build()
        .into()
}

fn http_get(url: &str) -> Result<String, RuntimeFault> {
    let mut response = http_agent()
        .get(url)
        .call()
        .map_err(|e| throw_err("IOError", format!("GET '{}' failed: {}", url, e)))?;
    response
        .body_mut()
        .read_to_string()
        .map_err(|e| throw_err("IOError", format!("GET '{}' failed reading body: {}", url, e)))
}

fn http_post(url: &str, body: &str, content_type: &str) -> Result<String, RuntimeFault> {
    let mut response = http_agent()
        .post(url)
        .header("Content-Type", content_type)
        .send(body)
        .map_err(|e| throw_err("IOError", format!("POST '{}' failed: {}", url, e)))?;
    response
        .body_mut()
        .read_to_string()
        .map_err(|e| throw_err("IOError", format!("POST '{}' failed reading body: {}", url, e)))
}

/// Open a SQLite database file (created when missing) and register the
/// connection on the runtime, returning a `db` handle value.
fn db_open(rt: &mut ModuleRuntime, path: &str) -> Result<Value, RuntimeFault> {
    let conn = rusqlite::Connection::open(path)
        .map_err(|e| throw_err("DbError", format!("cannot open '{}': {}", path, e)))?;
    let id = rt.db_next;
    rt.db_next += 1;
    rt.db_conns.insert(id, Rc::new(conn));
    Ok(Value::Db { id })
}

fn db_conn(rt: &ModuleRuntime, handle: &Value, what: &str) -> Result<Rc<rusqlite::Connection>, RuntimeFault> {
    match handle {
        Value::Db { id } => rt.db_conns.get(id).cloned().ok_or_else(|| {
            throw_err("DbError", format!("{}: unknown database handle (was it closed?)", what))
        }),
        other => Err(throw_err(
            "TypeError",
            format!("{} expects a db handle from db.Open, got {:?}", what, other),
        )),
    }
}

/// Bind a Kalvita array of params to SQLite values. Integrals stay
/// integers, other numbers bind as floats; anything else is a `ValueError`.
fn db_param_value(v: &Value) -> Result<rusqlite::types::Value, RuntimeFault> {
    use rusqlite::types::Value as Sql;
    match v {
        Value::Null => Ok(Sql::Null),
        Value::Logic(b) => Ok(Sql::Integer(*b as i64)),
        Value::String(s) => Ok(Sql::Text(s.clone())),
        Value::Number(n) => {
            if !n.is_finite() {
                return Err(throw_err(
                    "ValueError",
                    format!("cannot bind non-finite number {} to SQL", n),
                ));
            }
            if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n < 9.223372036854776e18 {
                #[allow(clippy::cast_possible_truncation)]
                Ok(Sql::Integer(*n as i64))
            } else {
                Ok(Sql::Real(*n))
            }
        }
        other => Err(throw_err(
            "TypeError",
            format!("cannot bind {} as a SQL parameter", value_type_name(other)),
        )),
    }
}

fn db_row_to_object(
    row: &rusqlite::Row,
    names: &[String],
) -> Result<HashMap<String, Value>, rusqlite::Error> {
    use rusqlite::types::ValueRef as Ref;
    let mut map = HashMap::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let value = match row.get_ref(i)? {
            Ref::Null => Value::Null,
            Ref::Integer(n) => Value::Number(n as f64),
            Ref::Real(n) => Value::Number(n),
            Ref::Text(bytes) => Value::String(String::from_utf8_lossy(bytes).to_string()),
            Ref::Blob(bytes) => Value::String(String::from_utf8_lossy(bytes).to_string()),
        };
        map.insert(name.clone(), value);
    }
    Ok(map)
}

#[allow(clippy::too_many_arguments)]
fn invoke_db(
    function: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Option<Value>, RuntimeFault> {
    let resolved = resolve_args(args, environment, types, rt, cur)?;
    match function {
        "Open" => match resolved.as_slice() {
            [target] => {
                let path = resolve_file_target(target)?;
                db_open(rt, &path).map(Some)
            }
            _ => Err(throw_err(
                "ValueError",
                "Open expects a path string or file variable",
            )),
        },
        "Close" => match resolved.as_slice() {
            [handle] => {
                match handle {
                    Value::Db { id } => {
                        if rt.db_conns.remove(id).is_none() {
                            return Err(throw_err(
                                "DbError",
                                "Close: unknown database handle (was it closed?)",
                            ));
                        }
                        Ok(Some(Value::Null))
                    }
                    other => Err(throw_err(
                        "TypeError",
                        format!("Close expects a db handle, got {:?}", other),
                    )),
                }
            }
            _ => Err(throw_err("ValueError", "Close expects (db)")),
        },
        "Exec" => {
            let (handle, sql, params) = match resolved.as_slice() {
                [handle, sql] => (handle, expect_string(sql, "Exec")?, Vec::new()),
                [handle, sql, args] => {
                    let items = expect_array(args, "Exec")?;
                    let bound = items
                        .iter()
                        .map(db_param_value)
                        .collect::<Result<Vec<_>, _>>()?;
                    (handle, expect_string(sql, "Exec")?, bound)
                }
                _ => {
                    return Err(throw_err(
                        "ValueError",
                        "Exec expects (db, sql[, params])",
                    ))
                }
            };
            let conn = db_conn(rt, handle, "Exec")?;
            conn.execute(&sql, rusqlite::params_from_iter(params))
                .map(|_| Some(Value::Null))
                .map_err(|e| throw_err("DbError", format!("Exec failed: {}", e)))
        }
        "Query" => {
            let (handle, sql, params) = match resolved.as_slice() {
                [handle, sql] => (handle, expect_string(sql, "Query")?, Vec::new()),
                [handle, sql, args] => {
                    let items = expect_array(args, "Query")?;
                    let bound = items
                        .iter()
                        .map(db_param_value)
                        .collect::<Result<Vec<_>, _>>()?;
                    (handle, expect_string(sql, "Query")?, bound)
                }
                _ => {
                    return Err(throw_err(
                        "ValueError",
                        "Query expects (db, sql[, params])",
                    ))
                }
            };
            let conn = db_conn(rt, handle, "Query")?;
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| throw_err("DbError", format!("Query failed: {}", e)))?;
            let names: Vec<String> =
                stmt.column_names().iter().map(|s| s.to_string()).collect();
            let rows = stmt
                .query_map(rusqlite::params_from_iter(params), |row| {
                    db_row_to_object(row, &names)
                })
                .map_err(|e| throw_err("DbError", format!("Query failed: {}", e)))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(Value::Object(
                    row.map_err(|e| throw_err("DbError", format!("Query failed: {}", e)))?,
                ));
            }
            Ok(Some(Value::Array(out)))
        }
        _ => Err(throw_err(
            "NameError",
            format!("Unknown db function: {}", function),
        )),
    }
}

/// Construct a widget registry entry from a title/text string.
/// Pure data — no display touch, so this is headless-safe.
fn gui_construct(rt: &mut ModuleRuntime, kind: GuiKind, text: String) -> Value {
    let id = rt.gui_next;
    rt.gui_next += 1;
    let entry = match kind {
        GuiKind::Window => GuiEntry::window(text),
        GuiKind::Button => GuiEntry::button(text),
    };
    rt.gui.insert(id, entry);
    Value::Gui { kind, id }
}

fn gui_kind_of(type_name: &str) -> Option<GuiKind> {
    match type_name {
        "window" => Some(GuiKind::Window),
        "button" => Some(GuiKind::Button),
        _ => None,
    }
}

/// Constructor-by-declaration: `var local w window = "Title"`.
/// Strings build widgets; anything else falls through to the normal
/// `check_type` error (e.g. a number in a `window` slot is a `TypeError`).
/// Callers guarantee `type_name` is a widget type.
fn gui_declare_value(
    rt: &mut ModuleRuntime,
    slot: &str,
    type_name: &str,
    kind: GuiKind,
    resolved: &Value,
) -> Result<Value, RuntimeFault> {
    match resolved {
        Value::String(text) => Ok(gui_construct(rt, kind, text.clone())),
        _ => {
            check_type(slot, type_name, resolved)?;
            Ok(resolved.clone())
        }
    }
}

/// Assignment into a `window`/`button` slot. Strings retitle in place,
/// `null` closes the widget (slot keeps its type), a same-kind handle
/// swaps in. Callers guarantee `declared` is a widget type.
fn gui_assign_slot(
    rt: &mut ModuleRuntime,
    slot: &str,
    declared: &str,
    kind: GuiKind,
    current: Option<Value>,
    new_value: &Value,
) -> Result<Value, RuntimeFault> {
    match (current, new_value) {
        (Some(Value::Gui { id, .. }), Value::String(text)) => {
            match rt.gui.get_mut(&id) {
                Some(entry) => {
                    entry.title = text.clone();
                    Ok(Value::Gui { kind, id })
                }
                None => Err(throw_err(
                    "GuiError",
                    format!("var.{}: widget was closed", slot),
                )),
            }
        }
        (Some(Value::Gui { id, .. }), Value::Null) => {
            rt.gui.remove(&id);
            Ok(Value::Null)
        }
        (_, Value::String(text)) => Ok(gui_construct(rt, kind, text.clone())),
        (_, Value::Null) => Ok(Value::Null),
        (Some(Value::Gui { kind: ck, id: cid }), Value::Gui { kind: nk, id: nid })
            if ck == kind && *nk == kind =>
        {
            if !rt.gui.contains_key(nid) {
                return Err(throw_err(
                    "GuiError",
                    format!("var.{}: widget was closed", slot),
                ));
            }
            rt.gui.remove(&cid);
            Ok(Value::Gui { kind, id: *nid })
        }
        (None, Value::Gui { kind: nk, id: nid }) if *nk == kind => {
            if !rt.gui.contains_key(nid) {
                return Err(throw_err(
                    "GuiError",
                    format!("var.{}: widget was closed", slot),
                ));
            }
            Ok(Value::Gui { kind, id: *nid })
        }
        _ => {
            check_type(slot, declared, new_value)?;
            Ok(new_value.clone())
        }
    }
}

/// Dispatch a method call on a widget handle (`w.Run()`,
/// `b.SetPos(x, y)`). Unknown methods are `NameError`, mirroring the
/// unknown-builtin convention; operations on closed widgets are `GuiError`.
#[allow(clippy::too_many_arguments)]
fn gui_method(
    kind: GuiKind,
    id: u64,
    method: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Option<Value>, RuntimeFault> {
    if !rt.gui.contains_key(&id) {
        return Err(throw_err(
            "GuiError",
            format!("{} was closed", kind.type_name()),
        ));
    }
    let resolved = resolve_args(args, environment, types, rt, cur)?;
    match (kind, method) {
        (GuiKind::Window, "Run") => match resolved.as_slice() {
            [] => gui_run_window(rt, cur, environment, types, id).map(|()| None::<Value>),
            _ => Err(throw_err("ValueError", "Run expects no arguments")),
        },
        (GuiKind::Button, "AttachToWindow") => match resolved.as_slice() {
            [target] => {
                let window_id = match target {
                    Value::Gui { kind: GuiKind::Window, id } => *id,
                    other => {
                        return Err(throw_err(
                            "TypeError",
                            format!(
                                "AttachToWindow expects a window handle, got {:?}",
                                other
                            ),
                        ))
                    }
                };
                if !rt.gui.contains_key(&window_id) {
                    return Err(throw_err("GuiError", "window was closed"));
                }
                if let Some(button) = rt.gui.get_mut(&id) {
                    button.parent = Some(window_id);
                }
                if let Some(window) = rt.gui.get_mut(&window_id)
                    && !window.children.contains(&id)
                {
                    window.children.push(id);
                }
                Ok(None)
            }
            _ => Err(throw_err(
                "ValueError",
                "AttachToWindow expects (window)",
            )),
        },
        (GuiKind::Button, "SetPos") => match resolved.as_slice() {
            [x, y] => {
                let (x, y) = (expect_int(x, "SetPos")?, expect_int(y, "SetPos")?);
                if x < 0 || y < 0 {
                    return Err(throw_err(
                        "ValueError",
                        format!("SetPos coordinates must be >= 0, got ({}, {})", x, y),
                    ));
                }
                match rt.gui.get_mut(&id) {
                    Some(button) => {
                        button.pos = Some((x as i32, y as i32));
                        Ok(None)
                    }
                    None => Err(throw_err("GuiError", "button was closed")),
                }
            }
            _ => Err(throw_err("ValueError", "SetPos expects (x, y)")),
        },
        (GuiKind::Button, "OnClick") => match resolved.as_slice() {
            [handler] => match handler {
                Value::Function { .. } => {
                    match rt.gui.get_mut(&id) {
                        Some(button) => {
                            button
                                .handlers
                                .entry("OnClick".to_string())
                                .or_default()
                                .push(handler.clone());
                            Ok(None)
                        }
                        None => Err(throw_err("GuiError", "button was closed")),
                    }
                }
                other => Err(throw_err(
                    "TypeError",
                    format!("OnClick expects a function, got {:?}", other),
                )),
            },
            _ => Err(throw_err("ValueError", "OnClick expects (function)")),
        },
        _ => Err(throw_err(
            "NameError",
            format!("Unknown {} method: {}", kind.type_name(), method),
        )),
    }
}

/// Register a widget event block (`mywindow.OnStart { ... }`). The body
/// desugars to an anonymous no-param function appended to the widget's
/// handler list, so blocks and `OnClick(fn)` compose in definition order.
fn gui_register_event(
    object: &str,
    event: &str,
    body: &[Statement],
    environment: &HashMap<String, Value>,
    _types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Flow, RuntimeFault> {
    let handle = lookup_scoped(object, environment, rt, cur).ok_or_else(|| {
        throw_err("NameError", format!("Unknown variable: {}", object))
    })?;
    let (kind, id) = match handle {
        Value::Gui { kind, id } => (kind, id),
        other => {
            return Err(throw_err(
                "TypeError",
                format!(
                    "{}.{}: {} is not a widget",
                    object,
                    event,
                    value_type_name(&other)
                ),
            ))
        }
    };
    let valid = matches!(
        (kind, event),
        (GuiKind::Window, "OnStart")
            | (GuiKind::Window, "OnExit")
            | (GuiKind::Button, "OnClick")
    );
    if !valid {
        return Err(throw_err(
            "NameError",
            format!("Unknown {} event: {}", kind.type_name(), event),
        ));
    }
    match rt.gui.get_mut(&id) {
        Some(entry) => {
            entry.handlers.entry(event.to_string()).or_default().push(
                Value::Function {
                    params: Vec::new(),
                    defaults: HashMap::new(),
                    body: body.to_vec(),
                },
            );
            Ok(Flow::Normal)
        }
        None => Err(throw_err(
            "GuiError",
            format!("{} was closed", kind.type_name()),
        )),
    }
}

/// Fire a widget event: run each registered handler with the ambient
/// scope cloned, `pass` bound to the widget, `arg`/`args` bound to `[]`
/// (reserved for future payloads). Handler `return` ends that handler;
/// `break`/`continue` are fatal misuse, mirroring function bodies.
/// Globals written by handlers persist via the live store write-through.
#[allow(clippy::too_many_arguments)]
fn fire_gui_event(
    rt: &mut ModuleRuntime,
    alias: &str,
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    handle: &Value,
    event: &str,
    display: &str,
) -> Result<(), RuntimeFault> {
    let id = match handle {
        Value::Gui { id, .. } => *id,
        _ => return Err(fatal_err("gui handler fired on non-widget")),
    };
    let handlers = rt
        .gui
        .get(&id)
        .map(|entry| entry.handlers.get(event).cloned().unwrap_or_default())
        .unwrap_or_default();
    for handler in handlers {
        let (params, defaults, body) = match handler {
            Value::Function { params, defaults, body } => (params, defaults, body),
            _ => return Err(fatal_err("gui handler registry corrupted")),
        };
        let _guard = enter_call(rt)?;
        let mut local_env = environment.clone();
        let mut local_types = types.clone();
        local_env.insert("pass".to_string(), handle.clone());
        let empty = Value::Array(Vec::new());
        local_env.insert("arg".to_string(), empty.clone());
        local_env.insert("args".to_string(), empty);
        bind_params(
            &params,
            &defaults,
            &[],
            environment,
            types,
            rt,
            alias,
            display,
            &mut local_env,
        )?;
        for stmt in &body {
            match execute_statement(stmt, &mut local_env, &mut local_types, rt, alias)? {
                Flow::Normal => {}
                Flow::Break => return Err(fatal_err("break outside loop")),
                Flow::Continue => return Err(fatal_err("continue outside loop")),
                Flow::Return(_) => break,
            }
        }
    }
    Ok(())
}

/// Blocks pumping the OS event loop for one window until it closes.
/// Single eframe loop, one viewport per registry window; clicks invoke
/// Kal handlers synchronously on the UI thread. No display (or another
/// backend failure) is a catchable `IOError`. Uncaught handler throws
/// abort the run *after* `OnExit` fires; fatals skip `OnExit`.
fn gui_run_window(
    rt: &mut ModuleRuntime,
    alias: &str,
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    id: u64,
) -> Result<(), RuntimeFault> {
    let title = match rt.gui.get(&id) {
        Some(entry) if entry.kind == GuiKind::Window && !entry.closed => entry.title.clone(),
        _ => {
            return Err(throw_err(
                "GuiError",
                "cannot Run a closed or missing window",
            ))
        }
    };
    if rt.gui_running {
        return Err(throw_err(
            "GuiError",
            "Run cannot nest inside a running window",
        ));
    }
    let handle = Value::Gui { kind: GuiKind::Window, id };
    // `OnStart` fires before the loop; an uncaught throw still runs
    // `OnExit` on the way out (cleanup rule); fatals skip it.
    let mut failure: Option<RuntimeFault> = match fire_gui_event(
        rt, alias, environment, types, &handle, "OnStart", "OnStart",
    ) {
        Ok(()) => None,
        Err(RuntimeFault::Throw(value)) => Some(RuntimeFault::Throw(value)),
        Err(fatal) => return Err(fatal),
    };
    // eframe owns the app, so runtime state round-trips through shared
    // ownership. Handler scope is a clone of the ambient scope (like a
    // cross-module call): handlers READ all entry locals, but LOCAL writes
    // stay call-local — widgets talk to the script through `var global`
    // slots, which persist via the live-store write-through.
    let real_rt = std::mem::take(rt);
    let test_frames = std::env::var("KALVITA_GUI_TEST_FRAMES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok());
    let shared = std::rc::Rc::new(std::cell::RefCell::new(KalGuiState {
        rt: real_rt,
        alias: alias.to_string(),
        env: environment.clone(),
        types: types.clone(),
        run_id: id,
        done: false,
        pending: None,
        frames: 0,
        test_frames,
    }));
    shared.borrow_mut().rt.gui_running = true;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size([480.0, 320.0]),
        ..Default::default()
    };
    let app_shared = shared.clone();
    let native_result = eframe::run_native(
        "kalvita",
        options,
        Box::new(move |_cc| Ok(Box::new(KalApp { state: app_shared }))),
    );
    // Reclaim state (the app drops with the loop) and sync the runtime
    // tables back (registry mutations like closed windows persist).
    let mut state = match std::rc::Rc::try_unwrap(shared) {
        Ok(cell) => cell.into_inner(),
        Err(_) => {
            *rt = ModuleRuntime::default();
            return Err(fatal_err("gui run lost its state"));
        }
    };
    state.rt.gui_running = false;
    *rt = std::mem::replace(&mut state.rt, ModuleRuntime::default());
    // Backend failure (no display, no GPU): OnStart side effects already
    // happened, so OnExit still fires to keep the pairing invariant.
    if let Err(e) = native_result {
        let _ = fire_gui_event(rt, alias, environment, types, &handle, "OnExit", "OnExit");
        return Err(throw_err(
            "IOError",
            format!("cannot open window: {}", e),
        ));
    }
    if failure.is_none() {
        failure = state.pending;
    }
    match failure {
        None => {
            let _ = fire_gui_event(rt, alias, environment, types, &handle, "OnExit", "OnExit");
            Ok(())
        }
        Some(RuntimeFault::Throw(value)) => {
            let _ = fire_gui_event(rt, alias, environment, types, &handle, "OnExit", "OnExit");
            Err(RuntimeFault::Throw(value))
        }
        Some(fatal) => Err(fatal),
    }
}

/// Mutable GUI-loop state shared with the eframe app.
#[derive(Debug, Default)]
struct KalGuiState {
    rt: ModuleRuntime,
    alias: String,
    env: HashMap<String, Value>,
    types: HashMap<String, String>,
    run_id: u64,
    done: bool,
    pending: Option<RuntimeFault>,
    frames: u64,
    test_frames: Option<u64>,
}

/// Frozen render snapshot: cloned per frame so rendering never holds
/// registry borrows while click dispatch mutates them.
#[derive(Debug, Clone)]
struct KalGuiWinSnap {
    id: u64,
    title: String,
    buttons: Vec<KalGuiBtnSnap>,
}

#[derive(Debug, Clone)]
struct KalGuiBtnSnap {
    id: u64,
    text: String,
    pos: Option<(i32, i32)>,
}

struct KalApp {
    state: std::rc::Rc<std::cell::RefCell<KalGuiState>>,
}

impl KalApp {
    fn snapshot(rt: &ModuleRuntime, run_id: u64) -> Vec<KalGuiWinSnap> {
        let mut wins = Vec::new();
        let mut ids: Vec<u64> = rt
            .gui
            .iter()
            .filter(|(_, e)| e.kind == GuiKind::Window && !e.closed)
            .map(|(id, _)| *id)
            .collect();
        // The run window renders first (root viewport); the rest follow.
        ids.sort_by_key(|id| if *id == run_id { 0 } else { 1 });
        for wid in ids {
            let Some(window) = rt.gui.get(&wid) else {
                continue;
            };
            let mut buttons = Vec::new();
            for bid in &window.children {
                if let Some(button) = rt.gui.get(bid) {
                    if button.kind == GuiKind::Button && !button.closed {
                        buttons.push(KalGuiBtnSnap {
                            id: *bid,
                            text: button.title.clone(),
                            pos: button.pos,
                        });
                    }
                }
            }
            // Unattached buttons land in the run window's flow.
            if wid == run_id {
                let mut orphans: Vec<u64> = rt
                    .gui
                    .iter()
                    .filter(|(_, e)| {
                        e.kind == GuiKind::Button && !e.closed && e.parent.is_none()
                    })
                    .map(|(id, _)| *id)
                    .collect();
                orphans.sort();
                for bid in orphans {
                    if buttons.iter().any(|b| b.id == bid) {
                        continue;
                    }
                    if let Some(button) = rt.gui.get(&bid) {
                        buttons.push(KalGuiBtnSnap {
                            id: bid,
                            text: button.title.clone(),
                            pos: button.pos,
                        });
                    }
                }
            }
            wins.push(KalGuiWinSnap {
                id: wid,
                title: window.title.clone(),
                buttons,
            });
        }
        wins
    }

    fn render_buttons(ui: &mut egui::Ui, buttons: &[KalGuiBtnSnap], clicks: &mut Vec<u64>) {
        for button in buttons {
            let clicked = match button.pos {
                Some((x, y)) => ui
                    .put(
                        egui::Rect::from_min_size(
                            egui::pos2(x as f32, y as f32),
                            egui::vec2(140.0, 30.0),
                        ),
                        egui::Button::new(&button.text),
                    )
                    .clicked(),
                None => ui.button(&button.text).clicked(),
            };
            if clicked {
                clicks.push(button.id);
            }
        }
    }
}

impl eframe::App for KalApp {
    /// Once per pass, no painting: frame bookkeeping, test-hook close,
    /// run-window liveness, and pending-error shutdown.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut state = self.state.borrow_mut();
        state.frames += 1;
        if let Some(n) = state.test_frames {
            // Headless/smoke runs have no input to drive repaints.
            ctx.request_repaint();
            if state.frames >= n {
                state.done = true;
            }
        }
        if !state.rt.gui.contains_key(&state.run_id) {
            state.done = true; // run window was null-closed from a handler
        }
        if state.done || state.pending.is_some() {
            state.done = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    /// Root viewport: one draggable `egui::Window` per Kal window
    /// (single native window in v1; true multi-viewport is follow-up).
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if ui.ctx().input(|i| i.viewport().close_requested()) {
            // Native X: let the close proceed; the loop drains on its own.
            self.state.borrow_mut().done = true;
            return;
        }
        let snapshot = {
            let state = self.state.borrow();
            Self::snapshot(&state.rt, state.run_id)
        };
        let mut clicks: Vec<u64> = Vec::new();
        let mut closed_windows: Vec<u64> = Vec::new();
        for win in &snapshot {
            let mut open = true;
            egui::Window::new(&win.title)
                .id(egui::Id::new(("kalvita", win.id)))
                .open(&mut open)
                .show(ui.ctx(), |window_ui| {
                    Self::render_buttons(window_ui, &win.buttons, &mut clicks);
                });
            if !open {
                closed_windows.push(win.id);
            }
        }
        // Apply closes, then dispatch clicks (registry mutations visible
        // at once). Disjoint field borrows through one `RefMut` are fine.
        let mut state = self.state.borrow_mut();
        for wid in closed_windows {
            state.rt.gui.remove(&wid);
            if wid == state.run_id {
                state.done = true;
            }
        }
        let alias = state.alias.clone();
        for bid in clicks {
            if state.pending.is_some() {
                break;
            }
            let handle = Value::Gui { kind: GuiKind::Button, id: bid };
            let state_ref = &mut *state;
            match fire_gui_event(
                &mut state_ref.rt,
                &alias,
                &state_ref.env,
                &state_ref.types,
                &handle,
                "OnClick",
                "OnClick",
            ) {
                Ok(()) => {}
                Err(fault) => {
                    state_ref.pending = Some(fault);
                    state_ref.done = true;
                }
            }
        }
        if state.done {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

/// Accept a path string or an open `file` variable for file builtins.
fn resolve_file_target(target: &Value) -> Result<String, RuntimeFault> {
    match target {
        Value::String(path) => Ok(path.clone()),
        Value::File { path } => Ok(path.clone()),
        other => Err(throw_err(
            "TypeError",
            format!("expected a path string or file variable, got {:?}", other),
        )),
    }
}

/// Bare `selectFile(path)` builtin: wraps a path into a `file` value for
/// `var local f file = selectFile("./notes.txt")`. No I/O happens here;
/// `file.Read`/`file.Write` report missing files as catchable `IOError`.
fn select_file_builtin(
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Value, RuntimeFault> {
    let resolved = resolve_args(args, environment, types, rt, cur)?;
    match resolved.as_slice() {
        [Value::String(path)] => {
            if path.trim().is_empty() {
                return Err(throw_err("ValueError", "selectFile needs a non-empty path"));
            }
            Ok(Value::File { path: path.clone() })
        }
        [other] => Err(throw_err(
            "TypeError",
            format!("selectFile expects a path string, got {:?}", other),
        )),
        _ => Err(throw_err("ValueError", "selectFile expects exactly one path")),
    }
}

/// Bare `assert(cond[, msg])`: passes through (`null`) when `cond` is
/// truthy, throws a catchable `AssertError` otherwise. Powers
/// assertion-style tests: a passing file prints nothing, a failing one
/// aborts with `Runtime error: Uncaught AssertError: <msg>`.
fn assert_builtin(
    args: &[Value],
    environment: &HashMap<String, Value>,
    types: &HashMap<String, String>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Value, RuntimeFault> {
    let resolved = resolve_args(args, environment, types, rt, cur)?;
    let (cond, msg) = match resolved.as_slice() {
        [cond] => (cond, "assertion failed".to_string()),
        [cond, msg] => (cond, stringify_value(msg)),
        _ => {
            return Err(throw_err(
                "ValueError",
                "assert expects (condition[, message])",
            ))
        }
    };
    if is_truthy(cond) {
        Ok(Value::Null)
    } else {
        Err(throw_err("AssertError", msg))
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

/// Format unix milliseconds (UTC) with tokens `YYYY MM DD HH mm SS`.
/// Unknown text passes through untouched.
fn format_unix_millis(millis: i64, pattern: &str) -> String {
    let secs = millis.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let time = secs.rem_euclid(86_400);
    // Howard Hinnant's days-to-civil (days since 1970-01-01).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    pattern
        .replace("YYYY", &format!("{:04}", year))
        .replace("MM", &format!("{:02}", m))
        .replace("DD", &format!("{:02}", d))
        .replace("HH", &format!("{:02}", time / 3_600))
        .replace("mm", &format!("{:02}", (time % 3_600) / 60))
        .replace("SS", &format!("{:02}", time % 60))
}

/// Convert `serde_json::Value` into Kalvita values. Objects become
/// `object`, arrays become `array`, numbers/bools/null map directly.
/// Depth-capped so hostile nesting is a `ValueError`, not a stack overflow.
fn json_to_value(v: &serde_json::Value, depth: usize) -> Result<Value, RuntimeFault> {
    const MAX_JSON_DEPTH: usize = 128;
    if depth > MAX_JSON_DEPTH {
        return Err(throw_err("ValueError", "JSON nesting exceeds 128 levels"));
    }
    match v {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(b) => Ok(Value::Logic(*b)),
        serde_json::Value::Number(n) => n
            .as_f64()
            .map(Value::Number)
            .ok_or_else(|| throw_err("ValueError", format!("JSON number out of range: {}", n))),
        serde_json::Value::String(s) => Ok(Value::String(s.clone())),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| json_to_value(item, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, item)| json_to_value(item, depth + 1).map(|value| (k.clone(), value)))
            .collect::<Result<HashMap<_, _>, _>>()
            .map(Value::Object),
    }
}

/// Convert Kalvita values back to JSON. Object keys are sorted for
/// deterministic output. Functions, files, and errors cannot cross
/// the boundary and are a `ValueError`.
fn value_to_json(v: &Value, depth: usize) -> Result<serde_json::Value, String> {
    const MAX_JSON_DEPTH: usize = 128;
    if depth > MAX_JSON_DEPTH {
        return Err("value nesting exceeds 128 levels".to_string());
    }
    match v {
        Value::Null => Ok(serde_json::Value::Null),
        Value::Logic(b) => Ok(serde_json::Value::Bool(*b)),
        Value::Number(n) => serde_json::Number::from_f64(*n)
            .map(serde_json::Value::Number)
            .ok_or_else(|| format!("cannot stringify non-finite number: {}", n)),
        Value::String(s) => Ok(serde_json::Value::String(s.clone())),
        Value::Array(items) => items
            .iter()
            .map(|item| value_to_json(item, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(serde_json::Value::Array),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::with_capacity(map.len());
            for k in keys {
                out.insert(k.clone(), value_to_json(&map[k], depth + 1)?);
            }
            Ok(serde_json::Value::Object(out))
        }
        other => Err(format!(
            "cannot stringify {} to JSON (only string/number/logic/null/array/object cross the boundary)",
            value_type_name(other)
        )),
    }
}

/// Glob match (`*` any run, `?` one char, `[...]` class, `\` escape).
/// Must consume the whole string. Unclosed `[` is a `ValueError`.
fn glob_match(pattern: &str, text: &str) -> Result<bool, RuntimeFault> {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_here(&p, 0, &t, 0)
}

fn glob_here(p: &[char], pi: usize, t: &[char], ti: usize) -> Result<bool, RuntimeFault> {
    if pi == p.len() {
        return Ok(ti == t.len());
    }
    match p[pi] {
        '*' => {
            let mut k = ti;
            loop {
                if glob_here(p, pi + 1, t, k)? {
                    return Ok(true);
                }
                if k == t.len() {
                    return Ok(false);
                }
                k += 1;
            }
        }
        '?' => {
            if ti == t.len() {
                return Ok(false);
            }
            glob_here(p, pi + 1, t, ti + 1)
        }
        '[' => {
            if ti == t.len() {
                return Ok(false);
            }
            let mut j = pi + 1;
            let mut negate = false;
            if j < p.len() && p[j] == '^' {
                negate = true;
                j += 1;
            }
            let mut matched = negate;
            let mut closed = false;
            let mut first = true; // a `]` in first position is a literal
            while j < p.len() {
                if p[j] == ']' && !first {
                    closed = true;
                    j += 1;
                    break;
                }
                first = false;
                if j + 2 < p.len() && p[j + 1] == '-' && p[j + 2] != ']' {
                    if p[j] <= t[ti] && t[ti] <= p[j + 2] {
                        matched = !negate;
                    }
                    j += 3;
                } else {
                    if p[j] == t[ti] {
                        matched = !negate;
                    }
                    j += 1;
                }
            }
            if !closed {
                return Err(throw_err("ValueError", "Match pattern has unclosed '['"));
            }
            if !matched {
                return Ok(false);
            }
            glob_here(p, j, t, ti + 1)
        }
        '\\' => {
            if pi + 1 >= p.len() {
                return Err(throw_err("ValueError", "Match pattern ends with '\\'"));
            }
            if ti == t.len() || t[ti] != p[pi + 1] {
                return Ok(false);
            }
            glob_here(p, pi + 2, t, ti + 1)
        }
        c => {
            if ti == t.len() || t[ti] != c {
                return Ok(false);
            }
            glob_here(p, pi + 1, t, ti + 1)
        }
    }
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
            match function {
                "Now" => {
                    if !args.is_empty() {
                        return Err(throw_err("ValueError", "Now expects no arguments"));
                    }
                    use std::time::{SystemTime, UNIX_EPOCH};
                    let millis = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map_err(|e| throw_err("IOError", format!("clock failed: {}", e)))?
                        .as_millis();
                    #[allow(clippy::cast_precision_loss)]
                    Ok(Some(Value::Number(millis as f64)))
                }
                "Format" => {
                    let resolved = resolve_args(args, environment, types, rt, cur)?;
                    let (millis, pattern) = match resolved.as_slice() {
                        [ms] => (expect_number(ms, "Format")?, "YYYY-MM-DD HH:mm:SS".to_string()),
                        [ms, pat] => (
                            expect_number(ms, "Format")?,
                            expect_string(pat, "Format")?,
                        ),
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Format expects (milliseconds[, pattern])",
                            ))
                        }
                    };
                    if !millis.is_finite() || millis < 0.0 {
                        return Err(throw_err(
                            "ValueError",
                            format!("Format expects non-negative milliseconds, got {}", millis),
                        ));
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    Ok(Some(Value::String(format_unix_millis(
                        millis as i64,
                        &pattern,
                    ))))
                }
                _ => Err(throw_err(
                    "NameError",
                    format!("Unknown time function: {}", function),
                )),
            }
        }
        Some(obj) if obj == "json" => {
            let resolved = resolve_args(args, environment, types, rt, cur)?;
            match function {
                "Parse" => match resolved.as_slice() {
                    [s] => {
                        let s = expect_string(s, "Parse")?;
                        serde_json::from_str::<serde_json::Value>(&s)
                            .map_err(|e| {
                                throw_err("ValueError", format!("invalid JSON: {}", e))
                            })
                            .and_then(|v| json_to_value(&v, 0))
                            .map(Some)
                    }
                    _ => Err(throw_err("ValueError", "Parse expects (json_string)")),
                },
                "Stringify" => match resolved.as_slice() {
                    [v] => value_to_json(v, 0)
                        .map(|j| Some(Value::String(j.to_string())))
                        .map_err(|e| throw_err("ValueError", e)),
                    _ => Err(throw_err("ValueError", "Stringify expects (value)")),
                },
                _ => Err(throw_err(
                    "NameError",
                    format!("Unknown json function: {}", function),
                )),
            }
        }
        Some(obj) if obj == "sys" => {            let resolved = resolve_args(args, environment, types, rt, cur)?;
            match function {
                "Args" => {
                    if !resolved.is_empty() {
                        return Err(throw_err("ValueError", "Args expects no arguments"));
                    }
                    Ok(Some(Value::Array(
                        std::env::args().map(Value::String).collect(),
                    )))
                }
                "Getenv" => match resolved.as_slice() {
                    [name] => {
                        let name = expect_string(name, "Getenv")?;
                        Ok(Some(
                            std::env::var(&name).map(Value::String).unwrap_or(Value::Null),
                        ))
                    }
                    _ => Err(throw_err("ValueError", "Getenv expects (name)")),
                },
                "Cwd" => {
                    if !resolved.is_empty() {
                        return Err(throw_err("ValueError", "Cwd expects no arguments"));
                    }
                    std::env::current_dir()
                        .map(|p| Some(Value::String(p.display().to_string())))
                        .map_err(|e| throw_err("IOError", format!("cannot read cwd: {}", e)))
                }
                _ => Err(throw_err(
                    "NameError",
                    format!("Unknown sys function: {}", function),
                )),
            }
        }
        Some(obj) if obj == "gui" => {
            let resolved = resolve_args(args, environment, types, rt, cur)?;
            match function {
                "PickFile" => match resolved.as_slice() {
                    [] => Ok(rfd::FileDialog::new()
                        .pick_file()
                        .map(|p| Value::String(p.display().to_string()))
                        .or(Some(Value::Null))),
                    [filter_name, exts] => {
                        let filter_name = expect_string(filter_name, "PickFile")?;
                        let exts = expect_array(exts, "PickFile")?;
                        let mut extensions = Vec::with_capacity(exts.len());
                        for ext in &exts {
                            match ext {
                                Value::String(s) => extensions.push(s.clone()),
                                other => {
                                    return Err(throw_err(
                                        "TypeError",
                                        format!(
                                            "PickFile extensions must be strings, got {:?}",
                                            other
                                        ),
                                    ))
                                }
                            }
                        }
                        Ok(rfd::FileDialog::new()
                            .add_filter(filter_name, &extensions)
                            .pick_file()
                            .map(|p| Value::String(p.display().to_string()))
                            .or(Some(Value::Null)))
                    }
                    _ => Err(throw_err(
                        "ValueError",
                        "PickFile expects () or (filter_name, extensions)",
                    )),
                },
                "PickFolder" => match resolved.as_slice() {
                    [] => Ok(rfd::FileDialog::new()
                        .pick_folder()
                        .map(|p| Value::String(p.display().to_string()))
                        .or(Some(Value::Null))),
                    _ => Err(throw_err("ValueError", "PickFolder expects no arguments")),
                },
                "SaveFile" => match resolved.as_slice() {
                    [] => Ok(rfd::FileDialog::new()
                        .save_file()
                        .map(|p| Value::String(p.display().to_string()))
                        .or(Some(Value::Null))),
                    [default_name] => {
                        let default_name = expect_string(default_name, "SaveFile")?;
                        Ok(rfd::FileDialog::new()
                            .set_file_name(default_name)
                            .save_file()
                            .map(|p| Value::String(p.display().to_string()))
                            .or(Some(Value::Null)))
                    }
                    _ => Err(throw_err(
                        "ValueError",
                        "SaveFile expects () or (default_name)",
                    )),
                },
                "Message" => {
                    let (title, text, kind, buttons) = match resolved.as_slice() {
                        [title, text] => (
                            expect_string(title, "Message")?,
                            expect_string(text, "Message")?,
                            "info".to_string(),
                            "ok".to_string(),
                        ),
                        [title, text, kind] => (
                            expect_string(title, "Message")?,
                            expect_string(text, "Message")?,
                            expect_string(kind, "Message")?,
                            "ok".to_string(),
                        ),
                        [title, text, kind, buttons] => (
                            expect_string(title, "Message")?,
                            expect_string(text, "Message")?,
                            expect_string(kind, "Message")?,
                            expect_string(buttons, "Message")?,
                        ),
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Message expects (title, text[, kind[, buttons]])",
                            ))
                        }
                    };
                    let level = match kind.as_str() {
                        "info" => rfd::MessageLevel::Info,
                        "warn" | "warning" => rfd::MessageLevel::Warning,
                        "error" => rfd::MessageLevel::Error,
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Message kind must be info, warn, or error",
                            ))
                        }
                    };
                    let buttons = match buttons.as_str() {
                        "ok" => rfd::MessageButtons::Ok,
                        "okcancel" => rfd::MessageButtons::OkCancel,
                        "yesno" => rfd::MessageButtons::YesNo,
                        "yesnocancel" => rfd::MessageButtons::YesNoCancel,
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Message buttons must be ok, okcancel, yesno, or yesnocancel",
                            ))
                        }
                    };
                    match rfd::MessageDialog::new()
                        .set_title(title)
                        .set_description(text)
                        .set_level(level)
                        .set_buttons(buttons)
                        .show()
                    {
                        rfd::MessageDialogResult::Yes => Ok(Some(Value::String("yes".to_string()))),
                        rfd::MessageDialogResult::No => Ok(Some(Value::String("no".to_string()))),
                        rfd::MessageDialogResult::Cancel => {
                            Ok(Some(Value::String("cancel".to_string())))
                        }
                        rfd::MessageDialogResult::Ok => Ok(Some(Value::String("ok".to_string()))),
                        rfd::MessageDialogResult::Custom(choice) => {
                            Ok(Some(Value::String(choice)))
                        }
                    }
                }
                _ => Err(throw_err(
                    "NameError",
                    format!("Unknown gui function: {}", function),
                )),
            }
        }
        Some(obj) if obj == "http" => {
            let resolved = resolve_args(args, environment, types, rt, cur)?;
            match function {
                "Get" => match resolved.as_slice() {
                    [url] => {
                        let url = expect_string(url, "Get")?;
                        http_get(&url).map(|body| Some(Value::String(body)))
                    }
                    _ => Err(throw_err("ValueError", "Get expects (url)")),
                },
                "Post" => match resolved.as_slice() {
                    [url, body] => {
                        let (url, body) = (
                            expect_string(url, "Post")?,
                            stringify_value(body),
                        );
                        http_post(&url, &body, "text/plain")
                            .map(|resp| Some(Value::String(resp)))
                    }
                    [url, body, content_type] => {
                        let (url, body, content_type) = (
                            expect_string(url, "Post")?,
                            stringify_value(body),
                            expect_string(content_type, "Post")?,
                        );
                        http_post(&url, &body, &content_type)
                            .map(|resp| Some(Value::String(resp)))
                    }
                    _ => Err(throw_err(
                        "ValueError",
                        "Post expects (url, body[, content_type])",
                    )),
                },
                _ => Err(throw_err(
                    "NameError",
                    format!("Unknown http function: {}", function),
                )),
            }
        }
        Some(obj) if obj == "db" => invoke_db(function, args, environment, types, rt, cur),
        Some(obj) if obj == "file" => {
            let resolved: Vec<Value> = args
                .iter()
                .map(|arg| resolve_value(arg, environment, types, rt, cur))
                .collect::<Result<Vec<_>, RuntimeFault>>()?;
            match function {
                "Read" => {
                    let path = match resolved.as_slice() {
                        [target] => resolve_file_target(target)?,
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Read expects a path string or file variable",
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
                        [target, content] => (resolve_file_target(target)?, stringify_value(content)),
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Write expects a path string or file variable, plus content",
                            ))
                        }
                    };
                    fs::write(&path, content)
                        .map(|()| Some(Value::Null))
                        .map_err(|e| throw_err("IOError", format!("cannot write '{}': {}", path, e)))
                }
                "Append" => {
                    let (path, content) = match resolved.as_slice() {
                        [target, content] => (resolve_file_target(target)?, stringify_value(content)),
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "Append expects a path string or file variable, plus content",
                            ))
                        }
                    };
                    use std::fs::OpenOptions;
                    use std::io::Write as _;
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                        .and_then(|mut f| f.write_all(content.as_bytes()))
                        .map(|()| Some(Value::Null))
                        .map_err(|e| throw_err("IOError", format!("cannot append '{}': {}", path, e)))
                }
                "Exists" => match resolved.as_slice() {
                    [target] => {
                        let path = resolve_file_target(target)?;
                        Ok(Some(Value::Logic(Path::new(&path).exists())))
                    }
                    _ => Err(throw_err(
                        "ValueError",
                        "Exists expects a path string or file variable",
                    )),
                },
                "Remove" => match resolved.as_slice() {
                    [target] => {
                        let path = resolve_file_target(target)?;
                        let target_path = Path::new(&path);
                        let result = if target_path.is_dir() {
                            fs::remove_dir(&path)
                        } else {
                            fs::remove_file(&path)
                        };
                        result
                            .map(|()| Some(Value::Null))
                            .map_err(|e| throw_err("IOError", format!("cannot remove '{}': {}", path, e)))
                    }
                    _ => Err(throw_err(
                        "ValueError",
                        "Remove expects a path string or file variable",
                    )),
                },
                "ListDir" => {
                    let path = match resolved.as_slice() {
                        [] => ".".to_string(),
                        [target] => resolve_file_target(target)?,
                        _ => {
                            return Err(throw_err(
                                "ValueError",
                                "ListDir expects zero or one path",
                            ))
                        }
                    };
                    fs::read_dir(&path)
                        .map_err(|e| throw_err("IOError", format!("cannot list '{}': {}", path, e)))?
                        .map(|entry| {
                            entry
                                .map(|e| {
                                    Value::String(e.file_name().to_string_lossy().to_string())
                                })
                                .map_err(|e| {
                                    throw_err("IOError", format!("cannot list '{}': {}", path, e))
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map(|mut names| {
                            names.sort_by(|a, b| match (a, b) {
                                (Value::String(x), Value::String(y)) => x.cmp(y),
                                _ => std::cmp::Ordering::Equal,
                            });
                            Some(Value::Array(names))
                        })
                }
                "MkDir" => match resolved.as_slice() {
                    [target] => {
                        let path = resolve_file_target(target)?;
                        fs::create_dir_all(&path)
                            .map(|()| Some(Value::Null))
                            .map_err(|e| throw_err("IOError", format!("cannot create dir '{}': {}", path, e)))
                    }
                    _ => Err(throw_err(
                        "ValueError",
                        "MkDir expects a path string or file variable",
                    )),
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

            // Widget-handle methods (`mybutton.SetPos(...)`): when the
            // receiver variable holds a `Gui` handle, dispatch to Rust.
            // A null receiver with a widget-method name means the widget
            // was closed (or never created) — `GuiError`, not a bare
            // `NameError`. Anything else falls through to the
            // bare-function path below, so non-widget calls behave
            // exactly as before.
            if let Some(obj) = object {
                match lookup_scoped(obj, environment, rt, cur) {
                    Some(Value::Gui { kind, id }) => {
                        return gui_method(kind, id, function, args, environment, types, rt, cur);
                    }
                    Some(Value::Null)
                        if matches!(
                            function,
                            "Run" | "AttachToWindow" | "SetPos" | "OnClick"
                        ) =>
                    {
                        return Err(throw_err(
                            "GuiError",
                            format!("{} was closed", obj),
                        ));
                    }
                    _ => {}
                }
            }

            let callee = match resolve_value(&Value::Variable(function.to_string()), environment, types, rt, cur)
                .or_else(|_| {
                    lookup_scoped(function, environment, rt, cur).ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", function)))
                }) {
                Ok(callee) => callee,
                // Bare `selectFile(path)` builtin. User-defined functions
                // take precedence: only reached when no such variable exists.
                Err(RuntimeFault::Throw(Value::Error { error_type, .. }))
                    if error_type == "NameError"
                        && function == "selectFile"
                        && object.is_none() =>
                {
                    return select_file_builtin(args, environment, types, rt, cur).map(Some);
                }
                // Bare `assert(cond[, msg])`: truthy passes, falsy throws a
                // catchable `AssertError`. Same precedence rule as selectFile.
                Err(RuntimeFault::Throw(Value::Error { error_type, .. }))
                    if error_type == "NameError"
                        && function == "assert"
                        && object.is_none() =>
                {
                    return assert_builtin(args, environment, types, rt, cur).map(Some);
                }
                Err(other) => return Err(other),
            };

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
        ("file", Value::File { .. }) => true,
        ("db", Value::Db { .. }) => true,
        ("window", Value::Gui { kind: GuiKind::Window, .. }) => true,
        ("button", Value::Gui { kind: GuiKind::Button, .. }) => true,
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
        Value::File { .. } => "file",
        Value::Db { .. } => "db",
        Value::Gui { kind, .. } => kind.type_name(),
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
            let stored = match gui_kind_of(type_name) {
                Some(kind) => gui_declare_value(rt, name, type_name, kind, &resolved)?,
                None => {
                    check_type(name, type_name, &resolved)?;
                    resolved
                }
            };
            environment.insert(format!("var.{}", name), stored);
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
            let stored = match gui_kind_of(type_name) {
                Some(kind) => gui_declare_value(rt, name, type_name, kind, &resolved)?,
                None => {
                    check_type(name, type_name, &resolved)?;
                    resolved.clone()
                }
            };
            environment.insert(format!("var.{}", name), stored.clone());
            types.insert(format!("var.{}", name), type_name.clone());
            rt.ensure_module(cur);
            let module = rt.modules.get_mut(cur).unwrap();
            module.globals.insert(format!("var.{}", name), stored);
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
            } else if object == "kal" {
                // Reserved for future `kal.*` events; ignored for now.
                Ok(Flow::Normal)
            } else {
                // Widget event blocks (`mywindow.OnStart { ... }`): resolve
                // the handle and append the body to its handler list.
                gui_register_event(object, name, body, environment, types, rt, cur)
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
            // Enforce the slot's declared type (if any). Widget slots
            // (`window`/`button`) retitle on strings and close on `null`.
            if let Some(declared) = types
                .get(&key)
                .cloned()
                .or_else(|| rt.lookup_type(cur, &key))
            {
                if let Some(kind) = gui_kind_of(&declared) {
                    let current = lookup_scoped(name, environment, rt, cur);
                    let stored =
                        gui_assign_slot(rt, name, &declared, kind, current, &new_value)?;
                    environment.insert(key.clone(), stored.clone());
                    if rt.is_global(cur, name) {
                        rt.ensure_module(cur);
                        rt.modules
                            .get_mut(cur)
                            .unwrap()
                            .globals
                            .insert(key, stored);
                    }
                    return Ok(());
                }
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
        Value::File { .. } => true,
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
        Value::File { path } => format!("file({})", path),
        Value::Db { id } => format!("db({})", id),
        Value::Gui { kind, id } => format!("{}({})", kind.type_name(), id),
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
        // Runs on a roomy stack: this asserts language semantics (the
        // depth guard fires with a fatal error), not native stack
        // consumption, which legitimately grows as builtins are added.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local loop function = () {\n        return(loop())\n    }\n    var local r number = loop()\n}\n"
        .to_string();
        let saw_fatal = std::thread::Builder::new()
            .name("recursion-guard".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let program = Parser::parse(&source).unwrap();
                let body = match &program.statements[0] {
                    Statement::Event { body, .. } => body.clone(),
                    _ => panic!("expected event"),
                };
                let mut env = HashMap::new();
                let mut saw_fatal = false;
                for stmt in &body {
                    if let Err(RuntimeFault::Fatal(_)) =
                        execute_statement(stmt, &mut env, &mut HashMap::new(), &mut ModuleRuntime::default(), TEST_ALIAS)
                    {
                        saw_fatal = true;
                    }
                }
                saw_fatal
            })
            .unwrap()
            .join()
            .unwrap();
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

    #[test]
    fn file_variables_round_trip() {
        let dir = std::env::temp_dir().join("kalvita_file_test");
        let _ = std::fs::create_dir_all(&dir);
        let target = dir.join("note.txt");
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local myfile file = selectFile(\"{}\")\n    file.Write(var.myfile, \"hello file\")\n    var local back string = file.Read(var.myfile)\n    con.Print(var.back)\n}}\n",
            target.display()
        );
        let (env, err) = run_with_types(&source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(
            env.get("var.back"),
            Some(&Value::String("hello file".to_string()))
        );
        assert!(matches!(
            env.get("var.myfile"),
            Some(Value::File { .. })
        ));
        let _ = std::fs::remove_file(&target);
    }

    #[test]
    fn file_type_is_enforced_and_select_validates() {
        // Non-file value into a file slot.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local myfile file = \"nope\"\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.unwrap().starts_with("TypeError"));
        // selectFile arity + type errors.
        for call in ["selectFile()", "selectFile(1, 2)", "selectFile(42)"] {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local f file = {}\n}}\n",
                call
            );
            let (_, err) = run_with_types(&source);
            assert!(err.is_some(), "{} should fail", call);
        }
        // Legacy string paths still work.
        let dir = std::env::temp_dir().join("kalvita_file_test");
        let _ = std::fs::create_dir_all(&dir);
        let target = dir.join("legacy.txt");
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    file.Write(\"{}\", \"legacy\")\n    var local back string = file.Read(\"{}\")\n}}\n",
            target.display(),
            target.display()
        );
        let (env, err) = run_with_types(&source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(
            env.get("var.back"),
            Some(&Value::String("legacy".to_string()))
        );
        let _ = std::fs::remove_file(&target);
    }

    #[test]
    fn sys_builtins_read_process() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local argv array = sys.Args()\n    var local missing null = sys.Getenv(\"KALVITA_DEFINITELY_MISSING_XYZ\")\n    var local cwd string = sys.Cwd()\n}\n";
        let (env, err) = run_with_types(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert!(matches!(env.get("var.argv"), Some(Value::Array(_))));
        assert_eq!(env.get("var.missing"), Some(&Value::Null));
        assert!(matches!(env.get("var.cwd"), Some(Value::String(s)) if !s.is_empty()));
        // Arity errors.
        for stmt in [
            "sys.Args(1)",
            "sys.Getenv()",
            "sys.Getenv(1, 2)",
            "sys.Cwd(\".\")",
            "sys.Nope()",
        ] {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    {}\n}}\n",
                stmt
            );
            let (_, err) = run_with_types(&source);
            assert!(err.is_some(), "{} should fail", stmt);
        }
    }

    #[test]
    fn time_format_known_dates() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local epoch string = time.Format(0)\n    var local y2021 string = time.Format(1609459200000)\n    var local custom string = time.Format(0, \"DD/MM/YYYY HH:mm\")\n}\n";
        let (env, err) = run_with_types(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(
            env.get("var.epoch"),
            Some(&Value::String("1970-01-01 00:00:00".to_string()))
        );
        assert_eq!(
            env.get("var.y2021"),
            Some(&Value::String("2021-01-01 00:00:00".to_string()))
        );
        assert_eq!(
            env.get("var.custom"),
            Some(&Value::String("01/01/1970 00:00".to_string()))
        );
        for stmt in ["time.Format(0 - 1)", "time.Format(\"x\")", "time.Format(0, \"x\", 1)"] {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    {}\n}}\n",
                stmt
            );
            let (_, err) = run_with_types(&source);
            assert!(err.is_some(), "{} should fail", stmt);
        }
    }

    #[test]
    fn str_new_builtins() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local lines array = str.Lines(\"a\\nb\\nc\")\n    var local chars array = str.Chars(\"hey\")\n    var local sw logic = str.StartsWith(\"hello\", \"he\")\n    var local ew logic = str.EndsWith(\"hello\", \"lo\")\n    var local tp string = str.TrimPrefix(\"xxhey\", \"xx\")\n    var local ts string = str.TrimSuffix(\"heyzz\", \"zz\")\n    var local rep string = str.Repeat(\"ab\", 3)\n    var local pi number = str.ParseInt(\"  -42 \")\n    var local pf number = str.ParseFloat(\"3.5\")\n    var local m logic = str.Match(\"hello.txt\", \"*.txt\")\n}\n";
        let (env, err) = run_with_types(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(
            env.get("var.lines"),
            Some(&Value::Array(vec![
                Value::String("a".to_string()),
                Value::String("b".to_string()),
                Value::String("c".to_string()),
            ]))
        );
        assert_eq!(env.get("var.sw"), Some(&Value::Logic(true)));
        assert_eq!(env.get("var.ew"), Some(&Value::Logic(true)));
        assert_eq!(env.get("var.tp"), Some(&Value::String("hey".to_string())));
        assert_eq!(env.get("var.ts"), Some(&Value::String("hey".to_string())));
        assert_eq!(env.get("var.rep"), Some(&Value::String("ababab".to_string())));
        assert_eq!(env.get("var.pi"), Some(&Value::Number(-42.0)));
        assert_eq!(env.get("var.pf"), Some(&Value::Number(3.5)));
        assert_eq!(env.get("var.m"), Some(&Value::Logic(true)));
        // Error cases leave no residue and report typed faults.
        let cases = [
            ("str.Repeat(\"x\", 0 - 1)", "ValueError"),
            ("str.Repeat(\"x\", 99999999999)", "ValueError"),
            ("str.ParseInt(\"12x\")", "ValueError"),
            ("str.ParseFloat(\"nan!\")", "ValueError"),
            ("str.Match(\"a\", \"[unclosed\")", "ValueError"),
            ("str.Match(\"trail\", \"trail\\\\\")", "ValueError"),
            ("str.Lines(1)", "TypeError"),
        ];
        for (stmt, want) in cases {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    {}\n}}\n",
                stmt
            );
            let (_, err) = run_with_types(&source);
            let err = err.unwrap_or_else(|| panic!("{} should fail", stmt));
            assert!(err.starts_with(want), "{}: got {}", stmt, err);
        }
    }

    #[test]
    fn str_match_glob_table() {
        let cases = [
            ("*", "anything", true),
            ("?.txt", "a.txt", true),
            ("?.txt", "ab.txt", false),
            ("h[ae]llo", "hello", true),
            ("h[ae]llo", "hallo", true),
            ("h[ae]llo", "hullo", false),
            ("h[^ae]llo", "hullo", true),
            ("[a-c]x", "bx", true),
            ("[]]x", "]x", true),
            ("a\\*b", "a*b", true),
            ("a\\*b", "axb", false),
            ("*.txt", "a.txt", true),
            ("*.txt", "a.txt.bak", false),
        ];
        for (pat, text, want) in cases {
            // Re-escape backslashes: Kal string literals process `\\`
            // themselves, so a glob `\` needs `\\` in the source text.
            let kal_pat = pat.replace('\\', "\\\\");
            let kal_text = text.replace('\\', "\\\\");
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local m logic = str.Match(\"{}\", \"{}\")\n}}\n",
                kal_text, kal_pat
            );
            let (env, err) = run_with_types(&source);
            assert!(err.is_none(), "{:?} on {:?}: {:?}", pat, text, err);
            assert_eq!(env.get("var.m"), Some(&Value::Logic(want)), "{:?} on {:?}", pat, text);
        }
    }

    #[test]
    fn file_new_builtins_round_trip() {
        let dir = std::env::temp_dir().join("kalvita_fileops_test");
        let _ = std::fs::create_dir_all(&dir);
        let dir_s = dir.display().to_string().replace('\\', "/");
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local base string = \"{dir_s}\"\n    var local f string = \"${{var.base}}/note.txt\"\n    file.Write(var.f, \"hi\")\n    file.Append(var.f, \"-there\")\n    var local back string = file.Read(var.f)\n    var local here logic = file.Exists(var.f)\n    var local gone logic = file.Exists(\"${{var.base}}/nope.txt\")\n    file.MkDir(\"${{var.base}}/sub/deep\")\n    var local names array = file.ListDir(var.base)\n    var local has logic = arr.Has(var.names, \"note.txt\")\n    file.Remove(var.f)\n    var local gone2 logic = file.Exists(var.f)\n}}\n",
        );
        let (env, err) = run_with_types(&source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(
            env.get("var.back"),
            Some(&Value::String("hi-there".to_string()))
        );
        assert_eq!(env.get("var.here"), Some(&Value::Logic(true)));
        assert_eq!(env.get("var.gone"), Some(&Value::Logic(false)));
        assert_eq!(env.get("var.has"), Some(&Value::Logic(true)));
        assert_eq!(env.get("var.gone2"), Some(&Value::Logic(false)));
        // Missing paths are catchable IOError, not fatal.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    file.Read(\"/definitely/missing/kalvita_xyz.txt\")\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.unwrap().starts_with("IOError"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn assert_passes_and_throws() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    assert(1 + 1 == 2)\n    assert(\"x\", \"custom\")\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    assert(false, \"boom-msg\")\n}\n";
        let (_, err) = run_with_types(source);
        assert_eq!(err.unwrap(), "AssertError: boom-msg");
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    try {\n        assert(0)\n        catch (AssertError) { con.Print(var.err.message) }\n    }\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.is_none(), "AssertError must be catchable: {:?}", err);
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    assert()\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.unwrap().starts_with("ValueError"));
    }

    #[test]
    fn json_round_trip_and_errors() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local cfg object = json.Parse(\"{\\\"name\\\": \\\"kal\\\", \\\"n\\\": 3, \\\"tags\\\": [\\\"a\\\"], \\\"ok\\\": true, \\\"x\\\": null}\")\n    var local out string = json.Stringify(var.cfg)\n    var local nums array = json.Parse(\"[1, 2.5]\")\n}\n";
        let (env, err) = run_with_types(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(
            env.get("var.out"),
            Some(&Value::String(
                "{\"n\":3.0,\"name\":\"kal\",\"ok\":true,\"tags\":[\"a\"],\"x\":null}".to_string()
            ))
        );
        assert_eq!(
            env.get("var.nums"),
            Some(&Value::Array(vec![Value::Number(1.0), Value::Number(2.5)]))
        );
        for (stmt, want) in [
            ("json.Parse(\"{bad\")", "ValueError"),
            ("json.Parse(1)", "TypeError"),
            ("json.Stringify(1, 2)", "ValueError"),
        ] {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    {}\n}}\n",
                stmt
            );
            let (_, err) = run_with_types(&source);
            let err = err.unwrap_or_else(|| panic!("{} should fail", stmt));
            assert!(err.starts_with(want), "{}: got {}", stmt, err);
        }
        // Functions cannot cross the JSON boundary.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local f function = (a) { return(arg.a) }\n    json.Stringify(var.f)\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.unwrap().starts_with("ValueError"));
    }

    #[test]
    fn db_open_exec_query_close() {
        let dir = std::env::temp_dir().join("kalvita_db_test");
        let _ = std::fs::create_dir_all(&dir);
        let target = dir.join("t.db");
        let _ = std::fs::remove_file(&target);
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local db db = db.Open(\"{}\")\n    db.Exec(var.db, \"CREATE TABLE u (id INTEGER PRIMARY KEY, name TEXT, score REAL)\")\n    db.Exec(var.db, \"INSERT INTO u (name, score) VALUES (?, ?)\", [\"nova\", 9.5])\n    db.Exec(var.db, \"INSERT INTO u (name, score) VALUES (?, ?)\", [\"kael\", 7])\n    var local rows array = db.Query(var.db, \"SELECT name, score FROM u WHERE score > ?\", [8])\n    var local count number = arr.Len(var.rows)\n    var local first object = var.rows[0]\n    var local who string = var.first.name\n    db.Close(var.db)\n}}\n",
            target.display()
        );
        let (env, err) = run_with_types(&source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(env.get("var.count"), Some(&Value::Number(1.0)));
        assert_eq!(env.get("var.who"), Some(&Value::String("nova".to_string())));
        assert!(matches!(env.get("var.db"), Some(Value::Db { .. })));
        // Bad SQL, closed handles, and mistyped binds are typed faults.
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local db db = db.Open(\"{}\")\n    db.Query(var.db, \"SELECT * FROM nope\")\n}}\n",
            target.display()
        );
        let (_, err) = run_with_types(&source);
        assert!(err.unwrap().starts_with("DbError"));
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local db db = db.Open(\"{}\")\n    db.Close(var.db)\n    db.Query(var.db, \"SELECT 1\")\n}}\n",
            target.display()
        );
        let (_, err) = run_with_types(&source);
        assert!(err.unwrap().starts_with("DbError"));
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local db db = 1\n}\n";
        let (_, err) = run_with_types(source);
        assert!(err.unwrap().starts_with("TypeError"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn http_get_post_and_status_errors() {
        use std::io::{Read, Write};
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for _ in 0..3 {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut buf = vec![0u8; 4096];
                let Ok(n) = stream.read(&mut buf) else {
                    return;
                };
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let body_len = req
                    .lines()
                    .find(|l| l.to_lowercase().starts_with("content-length:"))
                    .and_then(|l| l.split(':').nth(1))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                let split = req.find("\r\n\r\n").map(|i| i + 4).unwrap_or(req.len());
                let mut body = req.as_bytes()[split..].to_vec();
                while body.len() < body_len {
                    let mut extra = vec![0u8; 1024];
                    let Ok(m) = stream.read(&mut extra) else {
                        break;
                    };
                    if m == 0 {
                        break;
                    }
                    body.extend_from_slice(&extra[..m]);
                }
                let first_line = req.lines().next().unwrap_or("").to_string();
                let payload: Vec<u8> = if first_line.starts_with("GET /ok") {
                    b"{\"hello\":\"world\"}".to_vec()
                } else if first_line.starts_with("POST /echo") {
                    [b"echo:".as_slice(), &body].concat()
                } else {
                    let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found";
                    let _ = stream.write_all(resp.as_bytes());
                    continue;
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.write_all(&payload);
            }
        });
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    var local body string = http.Get(\"http://127.0.0.1:{port}/ok\")\n    var local back string = http.Post(\"http://127.0.0.1:{port}/echo\", \"abc\")\n}}\n",
        );
        let (env, err) = run_with_types(&source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(
            env.get("var.body"),
            Some(&Value::String("{\"hello\":\"world\"}".to_string()))
        );
        assert_eq!(env.get("var.back"), Some(&Value::String("echo:abc".to_string())));
        let source = format!(
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    http.Get(\"http://127.0.0.1:{port}/missing\")\n}}\n",
        );
        let (_, err) = run_with_types(&source);
        assert!(err.unwrap().starts_with("IOError"));
    }

    /// Harness that keeps the runtime: GUI tests assert on the widget
    /// registry (`rt.gui`) as well as the script env.
    fn run_gui(source: &str) -> (HashMap<String, Value>, ModuleRuntime, Option<String>) {
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
                Err(RuntimeFault::Fatal(msg)) => {
                    return (env, rt, Some(format!("fatal: {}", msg)))
                }
                Err(RuntimeFault::Throw(Value::Error { error_type, message })) => {
                    return (env, rt, Some(format!("{}: {}", error_type, message)))
                }
                Err(RuntimeFault::Throw(other)) => {
                    return (env, rt, Some(format!("throw: {:?}", other)))
                }
            }
        }
        (env, rt, None)
    }

    fn gui_id(env: &HashMap<String, Value>, name: &str) -> u64 {
        match env.get(name) {
            Some(Value::Gui { id, .. }) => *id,
            other => panic!("{} is not a widget: {:?}", name, other),
        }
    }

    #[test]
    fn gui_declare_attach_position_assign() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local mywindow window = \"Window Title\"\n    var local mybutton button = \"Text\"\n    mybutton.AttachToWindow(var.mywindow)\n    mybutton.SetPos(10, 20)\n    var.mybutton = \"Clicked!\"\n    var.mywindow = \"New Title\"\n}\n";
        let (env, rt, err) = run_gui(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        let wid = gui_id(&env, "var.mywindow");
        let bid = gui_id(&env, "var.mybutton");
        assert_eq!(rt.gui[&wid].title, "New Title");
        assert_eq!(rt.gui[&bid].title, "Clicked!");
        assert_eq!(rt.gui[&bid].parent, Some(wid));
        assert_eq!(rt.gui[&bid].pos, Some((10, 20)));
        assert!(rt.gui[&wid].children.contains(&bid));
        // Null closes the widget; the slot keeps its type.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local w window = \"T\"\n    var local b button = \"B\"\n    b.AttachToWindow(var.w)\n    var.b = null\n}\n";
        let (env, rt, err) = run_gui(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        assert_eq!(env.get("var.b"), Some(&Value::Null));
        assert!(rt.gui.values().all(|e| e.kind != GuiKind::Button));
        // Reassigning a string to the nulled slot constructs anew.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local b button = \"B\"\n    var.b = null\n    var.b = \"Again\"\n}\n";
        let (env, rt, err) = run_gui(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        let bid = gui_id(&env, "var.b");
        assert_eq!(rt.gui[&bid].title, "Again");
        // Error table: all headless, no display touch.
        let cases = [
            ("var local w window = 42", "TypeError"),
            ("var local b button = 42", "TypeError"),
            (
                "var local w window = \"T\"\n    var local b button = var.w",
                "TypeError",
            ),
            (
                "var local w window = \"T\"\n    var local c window = var.w",
                "ok:copy",
            ),
        ];
        for (stmt, want) in cases {
            let source = format!(
                "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {{\n    {}\n}}\n",
                stmt
            );
            let (_, _, err) = run_gui(&source);
            if want == "ok:copy" {
                assert!(err.is_none(), "{} should pass: {:?}", stmt, err);
                continue;
            }
            let err = err.unwrap_or_else(|| panic!("{} should fail", stmt));
            assert!(err.starts_with(want), "{}: got {}", stmt, err);
        }
    }

    #[test]
    fn gui_method_validation() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local w window = \"T\"\n    var local b button = \"B\"\n}\n";
        let (env, mut rt, err) = run_gui(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        let mut types = HashMap::new();
        types.insert("var.w".to_string(), "window".to_string());
        types.insert("var.b".to_string(), "button".to_string());
        // Unknown methods are NameError; bad args are typed faults.
        let w = env["var.w"].clone();
        let b = env["var.b"].clone();
        let no_args: Vec<Value> = Vec::new();
        let err = gui_method(GuiKind::Button, gui_id(&env, "var.b"), "Nope", &no_args, &env, &types, &mut rt, TEST_ALIAS)
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "NameError"));
        let err = gui_method(GuiKind::Window, gui_id(&env, "var.w"), "OnClick", &no_args, &env, &types, &mut rt, TEST_ALIAS)
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "NameError"));
        let bad_parent = vec![Value::String("x".to_string())];
        let err = gui_method(GuiKind::Button, gui_id(&env, "var.b"), "AttachToWindow", &bad_parent, &env, &types, &mut rt, TEST_ALIAS)
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "TypeError"));
        let neg = vec![Value::Number(0.0 - 1.0), Value::Number(0.0)];
        let err = gui_method(GuiKind::Button, gui_id(&env, "var.b"), "SetPos", &neg, &env, &types, &mut rt, TEST_ALIAS)
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "ValueError"));
        let not_fn = vec![Value::Number(1.0)];
        let err = gui_method(GuiKind::Button, gui_id(&env, "var.b"), "OnClick", &not_fn, &env, &types, &mut rt, TEST_ALIAS)
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "TypeError"));
        let _ = (w, b);
    }

    #[test]
    fn gui_events_register_fire_in_order() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var global log array = []\n    var global seen button = null\n    var global nargs number = 0 - 1\n    var local w window = \"T\"\n    var local b button = \"B\"\n    b.AttachToWindow(var.w)\n    b.OnClick {\n        var global log array = arr.Push(var.log, \"first\")\n    }\n    var local extra function = () {\n        var global log array = arr.Push(var.log, \"second\")\n        var global seen button = pass\n        var global nargs number = arr.Len(arg)\n    }\n    b.OnClick(var.extra)\n    b.OnClick {\n        var global log array = arr.Push(var.log, \"third\")\n    }\n    w.OnStart {\n        var global log array = arr.Push(var.log, \"start\")\n    }\n    w.OnExit {\n        var global log array = arr.Push(var.log, \"exit\")\n    }\n}\n";
        let (env, mut rt, err) = run_gui(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        let bid = gui_id(&env, "var.b");
        let wid = gui_id(&env, "var.w");
        assert_eq!(rt.gui[&bid].handlers["OnClick"].len(), 3);
        assert_eq!(rt.gui[&wid].handlers["OnStart"].len(), 1);
        // Fire click: block, function, block — in order, with pass/arg.
        let handle = Value::Gui { kind: GuiKind::Button, id: bid };
        let mut types = HashMap::new();
        fire_gui_event(&mut rt, TEST_ALIAS, &env, &types, &handle, "OnClick", "OnClick").unwrap();
        let log = rt.read_global(TEST_ALIAS, "log").expect("log global");
        assert_eq!(
            log,
            Value::Array(vec![
                Value::String("first".to_string()),
                Value::String("second".to_string()),
                Value::String("third".to_string()),
            ])
        );
        assert_eq!(
            rt.read_global(TEST_ALIAS, "seen"),
            Some(Value::Gui { kind: GuiKind::Button, id: bid })
        );
        assert_eq!(
            rt.read_global(TEST_ALIAS, "nargs"),
            Some(Value::Number(0.0))
        );
        // Start/exit fire on demand too.
        let whandle = Value::Gui { kind: GuiKind::Window, id: wid };
        fire_gui_event(&mut rt, TEST_ALIAS, &env, &types, &whandle, "OnStart", "OnStart").unwrap();
        fire_gui_event(&mut rt, TEST_ALIAS, &env, &types, &whandle, "OnExit", "OnExit").unwrap();
        let log = rt.read_global(TEST_ALIAS, "log").expect("log global");
        assert_eq!(
            log,
            Value::Array(vec![
                Value::String("first".to_string()),
                Value::String("second".to_string()),
                Value::String("third".to_string()),
                Value::String("start".to_string()),
                Value::String("exit".to_string()),
            ])
        );
        // Unknown events and non-widget receivers are typed faults.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local w window = \"T\"\n    w.OnClick {\n    }\n}\n";
        let (_, _, err) = run_gui(source);
        assert!(err.unwrap().starts_with("NameError"));
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local s string = \"x\"\n    s.OnClick {\n    }\n}\n";
        let (_, _, err) = run_gui(source);
        assert!(err.unwrap().starts_with("TypeError"));
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    nosuch.OnClick {\n    }\n}\n";
        let (_, _, err) = run_gui(source);
        assert!(err.unwrap().starts_with("NameError"));
    }

    #[test]
    fn gui_handler_abort_stops_chain() {
        // An uncaught throw aborts the remaining handlers with its fault.
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var global log array = []\n    var local b button = \"B\"\n    b.OnClick {\n        var global log array = arr.Push(var.log, \"one\")\n    }\n    b.OnClick {\n        throw(Boom, \"bang\")\n    }\n    b.OnClick {\n        var global log array = arr.Push(var.log, \"three\")\n    }\n}\n";
        let (env, mut rt, err) = run_gui(source);
        assert!(err.is_none(), "unexpected: {:?}", err);
        let bid = gui_id(&env, "var.b");
        let handle = Value::Gui { kind: GuiKind::Button, id: bid };
        let types = HashMap::new();
        let err = fire_gui_event(&mut rt, TEST_ALIAS, &env, &types, &handle, "OnClick", "OnClick")
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "Boom"));
        let log = rt.read_global(TEST_ALIAS, "log").expect("log global");
        assert_eq!(
            log,
            Value::Array(vec![Value::String("one".to_string())])
        );
    }

    #[test]
    fn gui_dialog_validation_needs_no_display() {
        // Only pre-backend validation is tested: success paths would
        // block on a real dialog, so they stay manual/gated.
        let env: HashMap<String, Value> = HashMap::new();
        let mut rt = ModuleRuntime::default();
        let types = HashMap::new();
        let bad = vec![Value::Number(1.0), Value::Number(2.0)];
        let err = invoke_function(Some("gui"), None, "PickFile", &bad, &env, &types, &mut rt, TEST_ALIAS)
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "TypeError"));
        let err = invoke_function(
            Some("gui"),
            None,
            "Message",
            &[
                Value::String("t".to_string()),
                Value::String("m".to_string()),
                Value::String("bogus".to_string()),
            ],
            &env,
            &types,
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "ValueError"));
        let err = invoke_function(Some("gui"), None, "Nope", &[], &env, &types, &mut rt, TEST_ALIAS)
            .unwrap_err();
        assert!(matches!(err, RuntimeFault::Throw(Value::Error { error_type, .. }) if error_type == "NameError"));
    }

    fn gui_display_present() -> bool {
        std::env::var("DISPLAY").is_ok() || std::env::var("WAYLAND_DISPLAY").is_ok()
    }

    #[test]
    fn gui_run_without_display_is_ioerror() {
        if gui_display_present() {
            return; // needs a real headless box; covered by CI convention
        }
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local w window = \"T\"\n    w.Run()\n}\n";
        let (_, _, err) = run_gui(source);
        let err = err.expect("Run without display should fail");
        assert!(err.starts_with("IOError"), "got: {}", err);
    }

    #[test]
    fn gui_run_pumps_and_returns_with_hook() {
        if !gui_display_present() {
            return;
        }
        // winit requires the main thread, and `cargo test` runs on
        // spawned threads — so the real loop runs in a child process
        // (same pattern as the golden harness), auto-closing after
        // 3 frames via the test hook.
        let dir = std::env::temp_dir().join("kalvita_gui_run_test");
        let _ = std::fs::create_dir_all(&dir);
        let script = dir.join("run_hook.kal");
        std::fs::write(
            &script,
            "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local w window = \"Hook\"\n    var local b button = \"B\"\n    b.AttachToWindow(var.w)\n    w.OnStart {\n        con.Print(\"hook-start\")\n    }\n    w.OnExit {\n        con.Print(\"hook-exit\")\n    }\n    w.Run()\n    con.Print(\"hook-after\")\n}\n",
        )
        .unwrap();
        let exe = std::env::current_exe().unwrap();
        // Under `cargo test`, current_exe is the harness
        // (`target/debug/deps/kalvita-<hash>`); the CLI lives next to
        // `deps/` as `target/debug/kalvita`.
        let mut exe = exe;
        exe.pop();
        if exe.file_name().and_then(|n| n.to_str()) == Some("deps") {
            exe.pop();
        }
        exe.push("kalvita");
        let output = std::process::Command::new(&exe)
            .arg("run")
            .arg(&script)
            .env("KALVITA_GUI_TEST_FRAMES", "3")
            .output()
            .expect("spawn child");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            output.status.success(),
            "child failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        assert_eq!(stdout, "hook-start\nhook-exit\nhook-after\n");
    }
}
