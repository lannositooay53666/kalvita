use crate::lexer::{Lexer, Token};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Null,
    Number(f64),
    String(String),
    Logic(bool),
    Variable(String),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    VariableDecl {
        name: String,
        type_name: String,
        value: Value,
    },
    FunctionCall {
        object: String,
        function: String,
        args: Vec<Value>,
    },
    Event {
        object: String,
        name: String,
        body: Vec<Statement>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub struct Header {
    pub script_type: String,
    pub language: String,
    pub version: u64,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    pub header: Header,
    pub statements: Vec<Statement>,
}

pub struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

impl Parser {
    pub fn parse(source: &str) -> Result<Program, String> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Self { tokens, index: 0 };

        let header = parser.parse_header()?;
        let mut statements = Vec::new();

        while !parser.is_at_end() {
            if matches!(parser.peek(), Some(Token::Eof)) {
                break;
            }
            statements.push(parser.parse_statement()?);
        }

        Ok(Program { header, statements })
    }

    fn parse_header(&mut self) -> Result<Header, String> {
        self.expect(Token::LBracket)?;
        let script_type = self.consume_identifier()?;
        let language = self.consume_identifier()?;
        self.expect_identifier("VERSION")?;
        let version = self.consume_number_literal()? as u64;
        self.expect(Token::RBracket)?;
        Ok(Header {
            script_type,
            language,
            version,
        })
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.peek() {
            Some(Token::Var) => self.parse_variable_decl(),
            Some(Token::Identifier(_)) => {
                let object = self.consume_identifier()?;
                if self.match_token(Token::Dot) {
                    let function = self.consume_identifier()?;
                    if matches!(self.peek(), Some(Token::LBrace)) {
                        self.expect(Token::LBrace)?;
                        let body = self.parse_block_contents()?;
                        return Ok(Statement::Event {
                            object,
                            name: function,
                            body,
                        });
                    }
                    let args = self.parse_arguments()?;
                    return Ok(Statement::FunctionCall {
                        object,
                        function,
                        args,
                    });
                }
                Err(format!("Unexpected syntax after identifier: {:?}", self.peek()))
            }
            Some(Token::Eof) => Err("Unexpected end of file".to_string()),
            _ => Err(format!("Unexpected statement start: {:?}", self.peek())),
        }
    }

    fn parse_variable_decl(&mut self) -> Result<Statement, String> {
        self.expect(Token::Var)?;
        self.expect(Token::Local)?;
        let name = self.consume_identifier()?;
        let type_name = self.consume_type_name()?;
        self.expect(Token::Assign)?;
        let value = self.parse_value()?;
        Ok(Statement::VariableDecl {
            name,
            type_name,
            value,
        })
    }

    fn parse_block_contents(&mut self) -> Result<Vec<Statement>, String> {
        let mut statements = Vec::new();
        while !matches!(self.peek(), Some(Token::RBrace) | Some(Token::Eof)) {
            statements.push(self.parse_statement()?);
        }
        self.expect(Token::RBrace)?;
        Ok(statements)
    }

    fn parse_arguments(&mut self) -> Result<Vec<Value>, String> {
        self.expect(Token::LParen)?;
        let mut args = Vec::new();
        if !matches!(self.peek(), Some(Token::RParen)) {
            loop {
                args.push(self.parse_value()?);
                if self.match_token(Token::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(Token::RParen)?;
        Ok(args)
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some(Token::StringLiteral(value)) => {
                let value = value.clone();
                self.index += 1;
                Ok(Value::String(value))
            }
            Some(Token::NumberLiteral(value)) => {
                let value = *value;
                self.index += 1;
                Ok(Value::Number(value))
            }
            Some(Token::BoolLiteral(value)) => {
                let value = *value;
                self.index += 1;
                Ok(Value::Logic(value))
            }
            Some(Token::Null) => {
                self.index += 1;
                Ok(Value::Null)
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.index += 1;
                Ok(Value::Variable(name))
            }
            _ => Err(format!("Expected value, found {:?}", self.peek())),
        }
    }

    fn consume_type_name(&mut self) -> Result<String, String> {
        match self.peek() {
            Some(Token::StringType) => {
                self.index += 1;
                Ok("string".to_string())
            }
            Some(Token::NumberType) => {
                self.index += 1;
                Ok("number".to_string())
            }
            Some(Token::LogicType) => {
                self.index += 1;
                Ok("logic".to_string())
            }
            Some(Token::Null) => {
                self.index += 1;
                Ok("null".to_string())
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.index += 1;
                Ok(name)
            }
            _ => Err(format!("Expected type name, found {:?}", self.peek())),
        }
    }

    fn consume_identifier(&mut self) -> Result<String, String> {
        match self.peek() {
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.index += 1;
                Ok(name)
            }
            _ => Err(format!("Expected identifier, found {:?}", self.peek())),
        }
    }

    fn consume_number_literal(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some(Token::NumberLiteral(value)) => {
                let value = *value;
                self.index += 1;
                Ok(value)
            }
            _ => Err(format!("Expected number literal, found {:?}", self.peek())),
        }
    }

    fn expect_identifier(&mut self, name: &str) -> Result<(), String> {
        match self.peek() {
            Some(Token::Identifier(value)) if value == name => {
                self.index += 1;
                Ok(())
            }
            _ => Err(format!("Expected identifier '{}' but found {:?}", name, self.peek())),
        }
    }

    fn expect(&mut self, token: Token) -> Result<(), String> {
        if self.matches(token.clone()) {
            self.index += 1;
            Ok(())
        } else {
            Err(format!("Expected {:?}, found {:?}", token, self.peek()))
        }
    }

    fn match_token(&mut self, token: Token) -> bool {
        if self.matches(token.clone()) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn matches(&self, expected: Token) -> bool {
        self.peek() == Some(&expected)
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Some(Token::Eof))
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index)
    }
}

pub fn execute(program: &Program) -> Result<(), String> {
    let mut environment: HashMap<String, Value> = HashMap::new();

    for statement in &program.statements {
        execute_statement(statement, &mut environment)?;
    }

    Ok(())
}

fn execute_statement(statement: &Statement, environment: &mut HashMap<String, Value>) -> Result<(), String> {
    match statement {
        Statement::VariableDecl { name, value, .. } => {
            let resolved = resolve_value(value, environment)?;
            environment.insert(name.clone(), resolved);
        }
        Statement::FunctionCall { object, function, args } => {
            if object == "con" && function == "Print" {
                let mut rendered = Vec::new();
                for arg in args {
                    let value = resolve_value(arg, environment)?;
                    rendered.push(format_value(value));
                }
                println!("{}", rendered.join(" "));
            }
        }
        Statement::Event { object, name, body } => {
            if object == "kal" && name == "OnStart" {
                for inner in body {
                    execute_statement(inner, environment)?;
                }
            }
        }
    }

    Ok(())
}

fn resolve_value(value: &Value, environment: &HashMap<String, Value>) -> Result<Value, String> {
    match value {
        Value::Variable(name) => environment
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Unknown variable: {}", name)),
        _ => Ok(value.clone()),
    }
}

fn format_value(value: Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v,
        Value::Logic(v) => v.to_string(),
        Value::Variable(v) => v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
