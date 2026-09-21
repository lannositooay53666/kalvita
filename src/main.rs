mod lexer;
mod parser;

use std::env;
use std::fs;

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

    let source = match fs::read_to_string(path) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("Failed to read '{}': {}", path, err);
            std::process::exit(1);
        }
    };

    let mut lexer = lexer::Lexer::new(&source);
    let tokens = lexer.tokenize();

    if debug_tokens {
        println!("Tokens for '{}':", path);
        println!("{:#?}", tokens);
    }

    let program = match parser::Parser::parse(&source) {
        Ok(program) => program,
        Err(err) => {
            eprintln!("Parse error: {}", err);
            std::process::exit(1);
        }
    };

    if debug_ast {
        println!("AST for '{}':", path);
        println!("{:#?}", program);
    }

    if let Err(err) = parser::execute(&program) {
        eprintln!("Runtime error: {}", err);
        std::process::exit(1);
    }
}
