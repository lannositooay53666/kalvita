mod lexer;
mod parser;

use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let debug_tokens = args.iter().any(|arg| arg == "--debug-tokens");
    let debug_ast = args.iter().any(|arg| arg == "--debug-ast");
    let path = args
        .iter()
        .skip(1)
        .find(|arg| !arg.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("sample.kal");

    if debug_tokens || debug_ast {
        let source = match std::fs::read_to_string(path) {
            Ok(src) => src,
            Err(err) => {
                eprintln!("Failed to read '{}': {}", path, err);
                std::process::exit(1);
            }
        };
        if debug_tokens {
            let mut lexer = lexer::Lexer::new(&source);
            let tokens = lexer.tokenize();
            println!("Tokens for '{}':", path);
            println!("{:#?}", tokens);
        }
        if debug_ast {
            match parser::Parser::parse(&source) {
                Ok(program) => {
                    println!("AST for '{}':", path);
                    println!("{:#?}", program);
                }
                Err(err) => {
                    eprintln!("Parse error: {}", err);
                    std::process::exit(1);
                }
            }
        }
    }

    if let Err(err) = parser::run_file(Path::new(path)) {
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
}
