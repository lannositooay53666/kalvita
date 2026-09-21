mod lexer;
mod parser;

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("sample.kal");

    let source = match fs::read_to_string(path) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("Failed to read '{}': {}", path, err);
            std::process::exit(1);
        }
    };

    let program = match parser::Parser::parse(&source) {
        Ok(program) => program,
        Err(err) => {
            eprintln!("Parse error: {}", err);
            std::process::exit(1);
        }
    };

    if let Err(err) = parser::execute(&program) {
        eprintln!("Runtime error: {}", err);
        std::process::exit(1);
    }
}
