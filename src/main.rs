mod error;
mod interpreter;
mod lexer;
mod parser;

use std::env;
use std::path::{Path, PathBuf};

fn usage() -> String {
    "kalvita — a small array-first scripting language\n\
     \n\
     Usage:\n  \
     kalvita [run] <file.kal> [--debug-tokens] [--debug-ast]\n  \
     kalvita run [file.kal]      run a script (kal.toml main, else sample.kal)\n  \
     kalvita check <file.kal>     parse only, report errors\n  \
     kalvita fmt <file.kal> [--write]  print (or rewrite) formatted source\n  \
     kalvita test [dir]          run golden tests (default: tests/)\n  \
     kalvita repl                interactive session\n  \
     kalvita <script> [args...]  run a kal.toml [scripts] shortcut"
        .to_string()
}

/// Minimal kal.toml reader: returns the `main = "..."` value if present.
fn project_main(dir: &Path) -> Option<PathBuf> {
    let content = std::fs::read_to_string(dir.join("kal.toml")).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some(rest) = line.strip_prefix("main") {
            let rest = rest.trim().strip_prefix('=')?.trim();
            if rest.starts_with('"') && rest.ends_with('"') && rest.len() >= 2 {
                return Some(dir.join(&rest[1..rest.len() - 1]));
            }
        }
    }
    None
}

/// Look up `name` in the `[scripts]` table of `dir/kal.toml`.
/// Returns the whitespace-split command, or `None` when absent.
/// A present-but-malformed entry (not `name = "..."`) yields `Some(vec![])`
/// so the caller can report it instead of silently ignoring it.
fn project_script(dir: &Path, name: &str) -> Option<Vec<String>> {
    let content = std::fs::read_to_string(dir.join("kal.toml")).ok()?;
    let mut in_scripts = false;
    for raw_line in content.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            in_scripts = line == "[scripts]";
            continue;
        }
        if !in_scripts {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != name {
            continue;
        }
        let value = value.trim();
        if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
            return Some(
                value[1..value.len() - 1]
                    .split_whitespace()
                    .map(str::to_string)
                    .collect(),
            );
        }
        return Some(Vec::new());
    }
    None
}

fn is_builtin_command(word: &str) -> bool {
    matches!(word, "run" | "check" | "fmt" | "test" | "repl" | "help")
}

fn default_entry() -> PathBuf {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if let Some(main) = project_main(&cwd) {
        return main;
    }
    PathBuf::from("sample.kal")
}

fn print_parse_error(path: &str, err: &str) -> ! {
    if err.starts_with("Parse error") {
        eprintln!("{}", err);
    } else {
        eprintln!("Parse error in '{}': {}", path, err);
    }
    std::process::exit(1);
}

fn print_runtime_error(err: &str) -> ! {
    if err.starts_with("Module error")
        || err.starts_with("Parse error")
        || err.starts_with("Cannot ")
    {
        eprintln!("{}", err);
    } else {
        eprintln!("Runtime error: {}", err);
    }
    std::process::exit(1);
}

fn cmd_run(path: &str, debug_tokens: bool, debug_ast: bool) {
    if debug_tokens || debug_ast {
        let source = match std::fs::read_to_string(path) {
            Ok(src) => src,
            Err(err) => {
                eprintln!("Failed to read '{}': {}", path, err);
                std::process::exit(1);
            }
        };
        if debug_tokens {
            let mut lex = lexer::Lexer::new(&source);
            println!("Tokens for '{}':", path);
            println!("{:#?}", lex.tokenize());
        }
        if debug_ast {
            match parser::Parser::parse(&source) {
                Ok(program) => {
                    println!("AST for '{}':", path);
                    println!("{:#?}", program);
                }
                Err(err) => print_parse_error(path, &err),
            }
        }
    }
    if let Err(err) = interpreter::run_file(Path::new(path)) {
        print_runtime_error(&err);
    }
}

fn cmd_check(path: &str) {
    let source = match std::fs::read_to_string(path) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("Failed to read '{}': {}", path, err);
            std::process::exit(1);
        }
    };
    match parser::Parser::parse(&source) {
        Ok(_) => println!("{}: OK", path),
        Err(err) => print_parse_error(path, &err),
    }
}

/// Line-based formatter: re-indents by brace depth (string- and
/// comment-aware), trims trailing whitespace. Comments survive because
/// formatting never re-lexes the code.
fn fmt_source(source: &str) -> String {
    let mut out = String::new();
    let mut indent: usize = 0;
    for raw_line in source.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            out.push('\n');
            continue;
        }
        // Dedent lines that close a block first.
        let closes = count_braces_outside_strings(line, '}') - count_braces_outside_strings(line, '{');
        if closes > 0 {
            indent = indent.saturating_sub(closes as usize);
        }
        // Heuristic: `} else {`, `} elseif` keep single indent.
        let starts_close = line.starts_with('}');
        let effective = if starts_close {
            indent
        } else {
            indent
        };
        out.push_str(&"    ".repeat(effective));
        out.push_str(line);
        out.push('\n');
        let opens = count_braces_outside_strings(line, '{');
        let closed = count_braces_outside_strings(line, '}');
        if opens > closed {
            indent += (opens - closed) as usize;
        } else if starts_close && opens > 0 {
            indent += opens as usize;
        }
    }
    out
}

fn count_braces_outside_strings(line: &str, brace: char) -> i32 {
    let chars: Vec<char> = line.chars().collect();
    let mut count = 0;
    let mut in_string = false;
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if in_string {
            if ch == '\\' {
                i += 1;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
        } else if ch == '/' && chars.get(i + 1) == Some(&'/') {
            break; // line comment: rest is not code
        } else if ch == brace {
            count += 1;
        }
        i += 1;
    }
    count
}

fn cmd_fmt(path: &str, write: bool) {
    let source = match std::fs::read_to_string(path) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("Failed to read '{}': {}", path, err);
            std::process::exit(1);
        }
    };
    // Validate first so fmt never mangles broken code silently.
    if let Err(err) = parser::Parser::parse(&source) {
        print_parse_error(path, &err);
    }
    let formatted = fmt_source(&source);
    if write {
        if let Err(err) = std::fs::write(path, formatted) {
            eprintln!("Failed to write '{}': {}", path, err);
            std::process::exit(1);
        }
        println!("{}: formatted", path);
    } else {
        print!("{}", formatted);
    }
}

/// Golden tests: every `<name>.kal` in `dir` runs in a child process and
/// its stdout must equal `<name>.expected` byte-for-byte.
fn cmd_test(dir: &str) {
    let dir_path = Path::new(dir);
    let entries = match std::fs::read_dir(dir_path) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("Cannot read test dir '{}': {}", dir, err);
            std::process::exit(1);
        }
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "kal").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("No .kal files in '{}'", dir);
        std::process::exit(1);
    }
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("kalvita"));
    let mut passed = 0;
    let mut failed = 0;
    for file in &files {
        let expected_path = file.with_extension("expected");
        let expected = match std::fs::read_to_string(&expected_path) {
            Ok(content) => content,
            Err(_) => {
                println!("SKIP {} (no {})", file.display(), expected_path.display());
                continue;
            }
        };
        let output = std::process::Command::new(&exe)
            .arg("run")
            .arg(file)
            .output();
        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                if output.status.success() && stdout == expected {
                    println!("ok {}", file.display());
                    passed += 1;
                } else {
                    println!("FAIL {}", file.display());
                    if !output.status.success() {
                        println!("  exit: {}", output.status);
                        println!(
                            "  stderr: {}",
                            String::from_utf8_lossy(&output.stderr).trim()
                        );
                    } else {
                        println!("  --- expected ---\n{}", expected);
                        println!("  --- got ---\n{}", stdout);
                    }
                    failed += 1;
                }
            }
            Err(err) => {
                println!("FAIL {} (cannot spawn: {})", file.display(), err);
                failed += 1;
            }
        }
    }
    println!("{} passed, {} failed", passed, failed);
    if failed > 0 {
        std::process::exit(1);
    }
}

/// True if braces are balanced outside string literals (rough REPL
/// continuation check; comments ignored for counting).
fn braces_balanced(text: &str) -> bool {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if in_string {
            if ch == '\\' {
                i += 1;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
        } else if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
        }
        i += 1;
    }
    depth <= 0 && !in_string
}

fn cmd_repl() {
    use std::io::{BufRead, Write};
    println!("kalvita repl — :help for commands, :quit to exit");
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut session = interpreter::Repl::new(cwd);
    let stdin = std::io::stdin();
    let mut buffer = String::new();
    print!("kal> ");
    let _ = std::io::stdout().flush();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if buffer.is_empty() {
            match trimmed {
                ":quit" | ":exit" => break,
                ":help" => {
                    println!("type Kalvita statements; they run immediately");
                    println!("multi-line blocks continue with ... until braces balance");
                    println!(":clear resets all variables  :quit exits");
                    print!("kal> ");
                    let _ = std::io::stdout().flush();
                    continue;
                }
                ":clear" => {
                    let cwd =
                        env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                    session = interpreter::Repl::new(cwd);
                    println!("cleared");
                    print!("kal> ");
                    let _ = std::io::stdout().flush();
                    continue;
                }
                _ => {}
            }
        }
        buffer.push_str(&line);
        buffer.push('\n');
        if !braces_balanced(&buffer) {
            print!("... ");
            let _ = std::io::stdout().flush();
            continue;
        }
        let chunk = std::mem::take(&mut buffer);
        if chunk.trim().is_empty() {
            print!("kal> ");
            let _ = std::io::stdout().flush();
            continue;
        }
        if let Err(err) = session.run_chunk(&chunk) {
            if err.starts_with("Parse error") || err.starts_with("Module error") {
                eprintln!("{}", err);
            } else {
                eprintln!("Runtime error: {}", err);
            }
        }
        print!("kal> ");
        let _ = std::io::stdout().flush();
    }
    println!("bye");
}

fn main() {
    let raw: Vec<String> = env::args().skip(1).collect();
    let debug_tokens = raw.iter().any(|a| a == "--debug-tokens");
    let debug_ast = raw.iter().any(|a| a == "--debug-ast");
    let mut args: Vec<String> = raw
        .into_iter()
        .filter(|a| a != "--debug-tokens" && a != "--debug-ast")
        .collect();

    // `[scripts]` shortcuts: `kalvita demo` expands via kal.toml before
    // anything else. Unknown names fall through to the file shorthand below,
    // so projects without `[scripts]` behave exactly as before.
    if let Some(first) = args.first().cloned() {
        if !is_builtin_command(&first) && !first.starts_with('-') {
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            match project_script(&cwd, &first) {
                Some(tokens) if !tokens.is_empty() => {
                    if !is_builtin_command(&tokens[0]) && !tokens[0].ends_with(".kal") {
                        eprintln!(
                            "Script '{}' must expand to a kalvita command, got '{}'",
                            first, tokens[0]
                        );
                        std::process::exit(2);
                    }
                    let mut expanded = tokens;
                    expanded.extend(args.iter().skip(1).cloned());
                    args = expanded;
                }
                Some(_) => {
                    eprintln!(
                        "Script '{}' is invalid (expected {} = \"<command>\" in kal.toml)",
                        first, first
                    );
                    std::process::exit(2);
                }
                None => {}
            }
        }
    }

    match args.first().map(String::as_str) {
        None => {
            let entry = default_entry();
            cmd_run(&entry.to_string_lossy(), debug_tokens, debug_ast);
        }
        Some("run") => {
            let file = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| default_entry().to_string_lossy().to_string());
            cmd_run(&file, debug_tokens, debug_ast);
        }
        Some("check") => {
            let Some(file) = args.get(1) else {
                eprintln!("{}", usage());
                std::process::exit(2);
            };
            cmd_check(file);
        }
        Some("fmt") => {
            let file = args.iter().skip(1).find(|a| !a.starts_with('-'));
            let Some(file) = file else {
                eprintln!("{}", usage());
                std::process::exit(2);
            };
            let write = args.iter().any(|a| a == "--write");
            cmd_fmt(file, write);
        }
        Some("test") => {
            let dir = args.get(1).map(String::as_str).unwrap_or("tests");
            cmd_test(dir);
        }
        Some("repl") => cmd_repl(),
        Some("help") | Some("--help") | Some("-h") => println!("{}", usage()),
        Some(path) if path.starts_with('-') => {
            eprintln!("{}", usage());
            std::process::exit(2);
        }
        // Back-compat: `kalvita file.kal [flags]`.
        Some(path) => cmd_run(path, debug_tokens, debug_ast),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn script_fixture(toml: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kalvita_script_test_{}_{}",
            std::process::id(),
            toml.len()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("kal.toml"), toml).unwrap();
        dir
    }

    #[test]
    fn scripts_resolve_and_split() {
        let dir = script_fixture("[project]\nname = \"x\"\n\n[scripts]\ndemo = \"run sample.kal\"\n");
        assert_eq!(
            project_script(&dir, "demo"),
            Some(vec!["run".to_string(), "sample.kal".to_string()])
        );
        assert_eq!(project_script(&dir, "missing"), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn scripts_malformed_is_some_empty() {
        let dir = script_fixture("[scripts]\nbad = run sample.kal\n");
        assert_eq!(project_script(&dir, "bad"), Some(Vec::new()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
