mod lexer;

fn main() {
    let mut lexer = lexer::Lexer::new("let x = 42;\nfn add(a, b) -> int { a + b }");
    println!("{:#?}", lexer.tokenize());
}
