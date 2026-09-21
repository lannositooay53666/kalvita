#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Let,
    Fn,
    Identifier(String),
    Number(i64),
    Assign,
    Semicolon,
    LParen,
    RParen,
    Comma,
    Arrow,
    Type(String),
    LBrace,
    RBrace,
    Plus,
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
            match self.peek().unwrap() {
                ' ' | '\n' | '\r' | '\t' => {
                    self.advance();
                }
                '=' => {
                    self.advance();
                    tokens.push(Token::Assign);
                }
                ';' => {
                    self.advance();
                    tokens.push(Token::Semicolon);
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
                '{' => {
                    self.advance();
                    tokens.push(Token::LBrace);
                }
                '}' => {
                    self.advance();
                    tokens.push(Token::RBrace);
                }
                '+' => {
                    self.advance();
                    tokens.push(Token::Plus);
                }
                '-' => {
                    if self.peek_next() == Some('>') {
                        self.advance();
                        self.advance();
                        tokens.push(Token::Arrow);
                    } else {
                        self.advance();
                        tokens.push(Token::Type("-".to_string()));
                    }
                }
                _ if self.peek().unwrap().is_ascii_alphabetic() || self.peek().unwrap() == '_' => {
                    let ident = self.read_identifier();
                    let keyword = match ident.as_str() {
                        "let" => Token::Let,
                        "fn" => Token::Fn,
                        _ => {
                            if matches!(ident.as_str(), "int" | "float" | "str" | "bool") {
                                Token::Type(ident)
                            } else {
                                Token::Identifier(ident)
                            }
                        }
                    };
                    tokens.push(keyword);
                }
                _ if self.peek().unwrap().is_ascii_digit() => {
                    let value = self.read_number();
                    tokens.push(Token::Number(value));
                }
                ch => {
                    let unexpected = ch;
                    self.advance();
                    panic!("Unexpected character in lexer: '{}'", unexpected);
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

    fn read_number(&mut self) -> i64 {
        let start = self.position;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        let value: String = self.source[start..self.position].iter().collect();
        value.parse().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_basic_kalvita_program() {
        let mut lexer = Lexer::new("let x = 42;\nfn add(a, b) -> int { a + b }");

        let tokens = lexer.tokenize();
        assert_eq!(tokens.len(), 20);
        assert_eq!(tokens[0], Token::Let);
        assert_eq!(tokens[1], Token::Identifier("x".to_string()));
        assert_eq!(tokens[2], Token::Assign);
        assert_eq!(tokens[3], Token::Number(42));
        assert_eq!(tokens[4], Token::Semicolon);
        assert_eq!(tokens[5], Token::Fn);
        assert_eq!(tokens[6], Token::Identifier("add".to_string()));
        assert_eq!(tokens[7], Token::LParen);
        assert_eq!(tokens[8], Token::Identifier("a".to_string()));
        assert_eq!(tokens[9], Token::Comma);
        assert_eq!(tokens[10], Token::Identifier("b".to_string()));
        assert_eq!(tokens[11], Token::RParen);
        assert_eq!(tokens[12], Token::Arrow);
        assert_eq!(tokens[13], Token::Type("int".to_string()));
        assert_eq!(tokens[14], Token::LBrace);
        assert_eq!(tokens[15], Token::Identifier("a".to_string()));
        assert_eq!(tokens[16], Token::Plus);
        assert_eq!(tokens[17], Token::Identifier("b".to_string()));
        assert_eq!(tokens[18], Token::RBrace);
        assert_eq!(tokens[19], Token::Eof);
    }
}