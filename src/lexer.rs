#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Var,
    Local,
    Null,
    Function,
    If,
    ElseIf,
    Else,
    BoolLiteral(bool),
    NumberLiteral(f64),
    StringLiteral(String),
    StringType,
    NumberType,
    LogicType,
    FunctionType,
    Identifier(String),
    Assign,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
    Dot,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    ArrayType,
    Plus,
    Minus,
    Asterisk,
    Slash,
    Semicolon,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Lexer {
    source: Vec<char>,
    position: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            position: 0,
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();

        while !self.is_at_end() {
            let ch = self.peek().unwrap();

            match ch {
                ' ' | '\n' | '\r' | '\t' => {
                    self.advance();
                }
                '[' => {
                    self.advance();
                    tokens.push(Token::LBracket);
                }
                ']' => {
                    self.advance();
                    tokens.push(Token::RBracket);
                }
                '{' => {
                    self.advance();
                    tokens.push(Token::LBrace);
                }
                '}' => {
                    self.advance();
                    tokens.push(Token::RBrace);
                }
                '(' => {
                    self.advance();
                    tokens.push(Token::LParen);
                }
                ')' => {
                    self.advance();
                    tokens.push(Token::RParen);
                }
                ',' => {
                    self.advance();
                    tokens.push(Token::Comma);
                }
                ';' => {
                    self.advance();
                    tokens.push(Token::Semicolon);
                }
                '+' => {
                    self.advance();
                    tokens.push(Token::Plus);
                }
                '-' => {
                    self.advance();
                    tokens.push(Token::Minus);
                }
                '*' => {
                    self.advance();
                    tokens.push(Token::Asterisk);
                }
                '/' => {
                    if self.peek_next() == Some('/') {
                        self.skip_line_comment();
                    } else {
                        self.advance();
                        tokens.push(Token::Slash);
                    }
                }
                '=' => {
                    if self.peek_next() == Some('=') {
                        self.advance();
                        self.advance();
                        tokens.push(Token::Equal);
                    } else {
                        self.advance();
                        tokens.push(Token::Assign);
                    }
                }
                '>' => {
                    if self.peek_next() == Some('=') {
                        self.advance();
                        self.advance();
                        tokens.push(Token::GreaterEqual);
                    } else {
                        self.advance();
                        tokens.push(Token::GreaterThan);
                    }
                }
                '<' => {
                    if self.peek_next() == Some('=') {
                        self.advance();
                        self.advance();
                        tokens.push(Token::LessEqual);
                    } else {
                        self.advance();
                        tokens.push(Token::LessThan);
                    }
                }
                '!' => {
                    if self.peek_next() == Some('=') {
                        self.advance();
                        self.advance();
                        tokens.push(Token::NotEqual);
                    } else {
                        panic!("Unexpected character in lexer: '!' ");
                    }
                }
                '.' => {
                    if self.peek_next() == Some('/') && self.peek_next_next() == Some('/') {
                        self.skip_block_comment();
                    } else {
                        self.advance();
                        tokens.push(Token::Dot);
                    }
                }
                '"' => {
                    let value = self.read_string();
                    tokens.push(Token::StringLiteral(value));
                }
                _ if ch.is_ascii_digit() => {
                    tokens.push(Token::NumberLiteral(self.read_number()));
                }
                _ if ch.is_ascii_alphabetic() || ch == '_' => {
                    let ident = self.read_identifier();
                    let token = match ident.as_str() {
                        "var" => Token::Var,
                        "local" => Token::Local,
                        "null" => Token::Null,
                        "function" => Token::Function,
                        "if" => Token::If,
                        "elseif" => Token::ElseIf,
                        "else" => Token::Else,
                        "true" => Token::BoolLiteral(true),
                        "false" => Token::BoolLiteral(false),
                        "string" => Token::StringType,
                        "number" => Token::NumberType,
                        "logic" => Token::LogicType,
                        "array" => Token::ArrayType,
                        _ => Token::Identifier(ident),
                    };
                    tokens.push(token);
                }
                _ => {
                    panic!("Unexpected character in lexer: '{}'", ch);
                }
            }
        }

        tokens.push(Token::Eof);
        tokens
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.source.len()
    }

    fn peek(&self) -> Option<char> {
        self.source.get(self.position).copied()
    }

    fn peek_next(&self) -> Option<char> {
        self.source.get(self.position + 1).copied()
    }

    fn peek_next_next(&self) -> Option<char> {
        self.source.get(self.position + 2).copied()
    }

    fn skip_line_comment(&mut self) {
        self.advance();
        self.advance();
        while !self.is_at_end() && self.peek() != Some('\n') {
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) {
        self.advance();
        self.advance();
        self.advance();
        while !self.is_at_end() {
            if self.peek() == Some('/') && self.peek_next() == Some('/') && self.peek_next_next() == Some('.') {
                self.advance();
                self.advance();
                self.advance();
                return;
            }
            self.advance();
        }
        panic!("Unterminated block comment");
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.source.get(self.position).copied();
        if ch.is_some() {
            self.position += 1;
        }
        ch
    }

    fn read_identifier(&mut self) -> String {
        let start = self.position;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        self.source[start..self.position].iter().collect()
    }

    fn read_number(&mut self) -> f64 {
        let start = self.position;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                self.advance();
            } else {
                break;
            }
        }
        let value: String = self.source[start..self.position].iter().collect();
        value.parse().unwrap()
    }

    fn read_string(&mut self) -> String {
        self.advance();
        let mut value = String::new();
        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance();
                return value;
            }
            if ch == '\\' {
                self.advance();
                if let Some(next) = self.peek() {
                    match next {
                        'n' => value.push('\n'),
                        '"' => value.push('"'),
                        '\\' => value.push('\\'),
                        _ => value.push(next),
                    }
                    self.advance();
                    continue;
                }
            }
            value.push(ch);
            self.advance();
        }
        panic!("Unterminated string literal");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_header_and_kal_event() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\nkal.OnStart {\n    var local wow string = \"Hello world\"\n    con.Print(wow)\n}\n";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();

        assert!(tokens.contains(&Token::LBracket));
        assert!(tokens.contains(&Token::Identifier("SCRIPTTYPE".to_string())));
        assert!(tokens.contains(&Token::Identifier("KALVITA".to_string())));
        assert!(tokens.contains(&Token::Identifier("VERSION".to_string())));
        assert!(tokens.contains(&Token::NumberLiteral(1.0)));
        assert!(tokens.contains(&Token::Identifier("kal".to_string())));
        assert!(tokens.contains(&Token::Dot));
        assert!(tokens.contains(&Token::Identifier("OnStart".to_string())));
        assert!(tokens.contains(&Token::LBrace));
        assert!(tokens.contains(&Token::Var));
        assert!(tokens.contains(&Token::Local));
        assert!(tokens.contains(&Token::StringType));
        assert!(tokens.contains(&Token::Assign));
        assert!(tokens.contains(&Token::StringLiteral("Hello world".to_string())));
        assert!(tokens.contains(&Token::Identifier("con".to_string())));
        assert!(tokens.contains(&Token::LParen));
        assert!(tokens.contains(&Token::Identifier("wow".to_string())));
        assert!(tokens.contains(&Token::RParen));
    }

    #[test]
    fn ignores_single_line_and_block_comments() {
        let source = "[SCRIPTTYPE KALVITA VERSION 1]\n// single line comment\nkal.OnStart {\n    .// multi\n    line\n    comment //.\n    var local wow string = \"Hello world\"\n}\n";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();

        assert!(!tokens.iter().any(|token| matches!(token, Token::Identifier(name) if name == "single" || name == "line" || name == "multi" || name == "comment")));
        assert!(tokens.contains(&Token::LBracket));
        assert!(tokens.contains(&Token::Identifier("SCRIPTTYPE".to_string())));
        assert!(tokens.contains(&Token::Identifier("KALVITA".to_string())));
        assert!(tokens.contains(&Token::Identifier("VERSION".to_string())));
        assert!(tokens.contains(&Token::NumberLiteral(1.0)));
        assert!(tokens.contains(&Token::Identifier("kal".to_string())));
        assert!(tokens.contains(&Token::Dot));
        assert!(tokens.contains(&Token::Identifier("OnStart".to_string())));
        assert!(tokens.contains(&Token::LBrace));
        assert!(tokens.contains(&Token::Var));
        assert!(tokens.contains(&Token::Local));
        assert!(tokens.contains(&Token::StringType));
        assert!(tokens.contains(&Token::Assign));
        assert!(tokens.contains(&Token::StringLiteral("Hello world".to_string())));
    }
}