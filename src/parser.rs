use crate::lexer::{Lexer, Token};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    StrictEqual,
    StrictNotEqual,
    And,
    Or,
    Greater,
    Less,
    GreaterEqual,
    LessEqual,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOperator {
    Not,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Null,
    Number(f64),
    String(String),
    Logic(bool),
    Variable(String),
    Object(HashMap<String, Value>),
    Array(Vec<Value>),
    Index {
        target: Box<Value>,
        index: Box<Value>,
    },
    FunctionCall {
        object: Option<String>,
        module: Option<String>,
        function: String,
        args: Vec<Value>,
    },
    ModuleVar {
        alias: String,
        name: String,
    },
    Binary {
        left: Box<Value>,
        op: BinaryOperator,
        right: Box<Value>,
    },
    Property {
        target: Box<Value>,
        key: String,
    },
    Unary {
        op: UnaryOperator,
        value: Box<Value>,
    },
    Function {
        params: Vec<String>,
        body: Vec<Statement>,
    },
    Error {
        error_type: String,
        message: String,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    VariableDecl {
        name: String,
        type_name: String,
        value: Value,
    },
    GlobalDecl {
        name: String,
        type_name: String,
        value: Value,
    },
    Declare {
        path: String,
        run: bool,
    },
    FunctionCall {
        object: Option<String>,
        module: Option<String>,
        function: String,
        args: Vec<Value>,
    },
    Return {
        value: Box<Value>,
    },
    Event {
        object: String,
        name: String,
        body: Vec<Statement>,
    },
    If {
        condition: Value,
        then_branch: Vec<Statement>,
        else_if_branches: Vec<(Value, Vec<Statement>)>,
        else_branch: Option<Vec<Statement>>,
    },
    While {
        condition: Value,
        body: Vec<Statement>,
    },
    ForIn {
        var: String,
        iterable: Value,
        body: Vec<Statement>,
    },
    Break,
    Continue,
    Throw {
        error_type: String,
        message: Box<Value>,
    },
    Try {
        body: Vec<Statement>,
        catches: Vec<CatchClause>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub struct CatchClause {
    pub error_type: String,
    pub body: Vec<Statement>,
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

        let program = Program { header, statements };
        validate_top_level_declares(&program)?;
        Ok(program)
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
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::Declare) => self.parse_declare_statement(),
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::For) => self.parse_for_statement(),
            Some(Token::Break) => {
                self.index += 1;
                Ok(Statement::Break)
            }
            Some(Token::Continue) => {
                self.index += 1;
                Ok(Statement::Continue)
            }
            Some(Token::Try) => self.parse_try_statement(),
            Some(Token::Throw) => self.parse_throw_statement(),
            Some(Token::Catch) => Err("catch must be inside try".to_string()),
            Some(Token::Identifier(_)) => {
                let name = self.consume_identifier()?;

                if self.matches(Token::Colon) {
                    // Module-qualified statement: `alias:foo(args)` calls a
                    // module global function. Data reads (`alias:var.x`)
                    // are values, not statements.
                    self.expect(Token::Colon)?;
                    if matches!(self.peek(), Some(Token::Var)) {
                        return Err(
                            "module data cannot be used as a statement; use module:foo() to call a module function".to_string(),
                        );
                    }
                    let function = self.consume_identifier()?;
                    if !self.matches(Token::LParen) {
                        return Err(
                            "expected '(' after module function name; use module:foo(...) to call or module:var.foo for the value".to_string(),
                        );
                    }
                    let args = self.parse_arguments()?;
                    return Ok(Statement::FunctionCall {
                        object: None,
                        module: Some(name),
                        function,
                        args,
                    });
                }

                if self.match_token(Token::Dot) {
                    let function = self.consume_identifier()?;
                    if matches!(self.peek(), Some(Token::LBrace)) {
                        self.expect(Token::LBrace)?;
                        let body = self.parse_block_contents()?;
                        return Ok(Statement::Event {
                            object: name,
                            name: function,
                            body,
                        });
                    }
                    let args = self.parse_arguments()?;
                    return Ok(Statement::FunctionCall {
                        object: Some(name),
                        module: None,
                        function,
                        args,
                    });
                }

                if self.matches(Token::LParen) {
                    let args = self.parse_arguments()?;
                    return Ok(Statement::FunctionCall {
                        object: None,
                        module: None,
                        function: name,
                        args,
                    });
                }

                Err(format!("Unexpected syntax after identifier: {:?}", self.peek()))
            }
            Some(Token::Eof) => Err("Unexpected end of file".to_string()),
            _ => Err(format!("Unexpected statement start: {:?}", self.peek())),
        }
    }

    fn parse_return_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::Return)?;
        self.expect(Token::LParen)?;
        let value = self.parse_value()?;
        self.expect(Token::RParen)?;
        Ok(Statement::Return {
            value: Box::new(value),
        })
    }

    fn parse_if_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::If)?;
        self.expect(Token::LParen)?;
        let condition = self.parse_expression()?;
        self.expect(Token::RParen)?;
        self.expect(Token::LBrace)?;
        let then_branch = self.parse_block_contents()?;
        let mut else_if_branches = Vec::new();

        while self.match_token(Token::ElseIf) {
            self.expect(Token::LParen)?;
            let branch_condition = self.parse_expression()?;
            self.expect(Token::RParen)?;
            self.expect(Token::LBrace)?;
            let branch_body = self.parse_block_contents()?;
            else_if_branches.push((branch_condition, branch_body));
        }

        let else_branch = if self.match_token(Token::Else) {
            self.expect(Token::LBrace)?;
            Some(self.parse_block_contents()?)
        } else {
            None
        };

        Ok(Statement::If {
            condition,
            then_branch,
            else_if_branches,
            else_branch,
        })
    }

    fn parse_while_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::While)?;
        self.expect(Token::LParen)?;
        let condition = self.parse_expression()?;
        self.expect(Token::RParen)?;
        self.expect(Token::LBrace)?;
        let body = self.parse_block_contents()?;
        Ok(Statement::While { condition, body })
    }

    fn parse_for_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::For)?;
        self.expect(Token::LParen)?;
        // for (name in iterable) binds var.<name> each iteration.
        // Also accept `for (var.name in ...)`? No - keep bare `name`.
        let loop_var = self.consume_identifier()?;
        self.expect(Token::In)?;
        let iterable = self.parse_expression()?;
        self.expect(Token::RParen)?;
        self.expect(Token::LBrace)?;
        let body = self.parse_block_contents()?;
        Ok(Statement::ForIn {
            var: loop_var,
            iterable,
            body,
        })
    }

    fn parse_throw_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::Throw)?;
        self.expect(Token::LParen)?;
        let error_type = self.consume_identifier()?;
        self.expect(Token::Comma)?;
        let message = self.parse_expression()?;
        self.expect(Token::RParen)?;
        Ok(Statement::Throw {
            error_type,
            message: Box::new(message),
        })
    }

    fn parse_try_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::Try)?;
        self.expect(Token::LBrace)?;
        let mut body = Vec::new();
        let mut catches = Vec::new();
        while !matches!(self.peek(), Some(Token::RBrace) | Some(Token::Eof)) {
            if matches!(self.peek(), Some(Token::Catch)) {
                self.expect(Token::Catch)?;
                self.expect(Token::LParen)?;
                let error_type = self.consume_identifier()?;
                self.expect(Token::RParen)?;
                self.expect(Token::LBrace)?;
                let catch_body = self.parse_block_contents()?;
                catches.push(CatchClause { error_type, body: catch_body });
            } else {
                body.push(self.parse_statement()?);
            }
        }
        self.expect(Token::RBrace)?;
        Ok(Statement::Try { body, catches })
    }

    fn parse_variable_decl(&mut self) -> Result<Statement, String> {
        self.expect(Token::Var)?;
        let is_global = if self.match_token(Token::Local) {
            false
        } else if self.match_token(Token::Global) {
            true
        } else {
            return Err(format!("Expected 'local' or 'global', found {:?}", self.peek()));
        };
        let name = self.consume_identifier()?;
        let type_name = self.consume_type_name()?;
        self.expect(Token::Assign)?;
        let value = if self.matches(Token::LParen) {
            self.parse_function_literal()?
        } else {
            self.parse_value()?
        };
        if is_global {
            Ok(Statement::GlobalDecl {
                name,
                type_name,
                value,
            })
        } else {
            Ok(Statement::VariableDecl {
                name,
                type_name,
                value,
            })
        }
    }

    fn parse_declare_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::Declare)?;
        self.expect(Token::LParen)?;
        let path = match self.peek() {
            Some(Token::StringLiteral(value)) => {
                let path = value.clone();
                self.index += 1;
                path
            }
            _ => return Err(format!("Expected module path string, found {:?}", self.peek())),
        };
        let run = if self.match_token(Token::Comma) {
            match self.peek() {
                Some(Token::BoolLiteral(value)) => {
                    let run = *value;
                    self.index += 1;
                    run
                }
                _ => {
                    return Err(format!(
                        "Expected true or false after module path, found {:?}",
                        self.peek()
                    ))
                }
            }
        } else {
            true
        };
        self.expect(Token::RParen)?;
        Ok(Statement::Declare { path, run })
    }

    fn parse_function_literal(&mut self) -> Result<Value, String> {
        self.expect(Token::LParen)?;
        let params = self.parse_parameter_list()?;
        self.expect(Token::RParen)?;
        self.expect(Token::LBrace)?;
        let body = self.parse_block_contents()?;
        Ok(Value::Function { params, body })
    }

    fn parse_parameter_list(&mut self) -> Result<Vec<String>, String> {
        let mut params = Vec::new();
        if !matches!(self.peek(), Some(Token::RParen)) {
            loop {
                params.push(self.consume_identifier()?);
                if !self.match_token(Token::Comma) {
                    break;
                }
            }
        }
        Ok(params)
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
        self.parse_expression()
    }

    fn parse_expression(&mut self) -> Result<Value, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Value, String> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Token::Or)) {
            self.index += 1;
            let right = self.parse_and()?;
            left = Value::Binary {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Value, String> {
        let mut left = self.parse_unary()?;
        while matches!(self.peek(), Some(Token::And)) {
            self.index += 1;
            let right = self.parse_unary()?;
            left = Value::Binary {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Value, String> {
        if matches!(self.peek(), Some(Token::Not)) {
            self.index += 1;
            let value = self.parse_unary()?;
            return Ok(Value::Unary {
                op: UnaryOperator::Not,
                value: Box::new(value),
            });
        }

        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Value, String> {
        let mut left = self.parse_additive()?;
        while matches!(
            self.peek(),
            Some(Token::Equal)
                | Some(Token::NotEqual)
                | Some(Token::GreaterThan)
                | Some(Token::LessThan)
                | Some(Token::GreaterEqual)
                | Some(Token::LessEqual)
        ) {
            let op = self.parse_comparison_operator()?;
            let right = self.parse_additive()?;
            left = Value::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Value, String> {
        let mut left = self.parse_multiplicative()?;
        while matches!(self.peek(), Some(Token::Plus) | Some(Token::Minus)) {
            let op = if self.matches(Token::Plus) {
                self.index += 1;
                BinaryOperator::Add
            } else if self.matches(Token::Minus) {
                self.index += 1;
                BinaryOperator::Subtract
            } else {
                return Err(format!("Expected operator, found {:?}", self.peek()));
            };
            let right = self.parse_multiplicative()?;
            left = Value::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Value, String> {
        let mut left = self.parse_primary()?;
        while matches!(self.peek(), Some(Token::Asterisk) | Some(Token::Slash)) {
            let op = if self.matches(Token::Asterisk) {
                self.index += 1;
                BinaryOperator::Multiply
            } else if self.matches(Token::Slash) {
                self.index += 1;
                BinaryOperator::Divide
            } else {
                return Err(format!("Expected operator, found {:?}", self.peek()));
            };
            let right = self.parse_primary()?;
            left = Value::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Value, String> {
        let mut value = match self.peek() {
            Some(Token::StringLiteral(value)) => {
                let value = value.clone();
                self.index += 1;
                Value::String(value)
            }
            Some(Token::NumberLiteral(value)) => {
                let value = *value;
                self.index += 1;
                Value::Number(value)
            }
            Some(Token::BoolLiteral(value)) => {
                let value = *value;
                self.index += 1;
                Value::Logic(value)
            }
            Some(Token::Null) => {
                self.index += 1;
                Value::Null
            }
            Some(Token::Var) => {
                self.index += 1;
                self.expect(Token::Dot)?;
                let name = self.consume_identifier()?;
                Value::Variable(format!("var.{}", name))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.index += 1;
                if self.matches(Token::Colon) {
                    // Module-qualified value: `alias:var.x` reads module
                    // global data, `alias:foo(args)` calls a module function.
                    self.expect(Token::Colon)?;
                    if matches!(self.peek(), Some(Token::Var)) {
                        self.index += 1;
                        self.expect(Token::Dot)?;
                        let var_name = self.consume_identifier()?;
                        Value::ModuleVar {
                            alias: name,
                            name: var_name,
                        }
                    } else {
                        let function = self.consume_identifier()?;
                        if self.matches(Token::LParen) {
                            let args = self.parse_arguments()?;
                            return Ok(Value::FunctionCall {
                                object: None,
                                module: Some(name),
                                function,
                                args,
                            });
                        }
                        return Err("expected '(' after module function name; use module:foo(...) to call or module:var.foo for the value".to_string());
                    }
                } else if self.matches(Token::Dot) {
                    self.expect(Token::Dot)?;
                    let scoped_name = self.consume_identifier()?;
                    if self.matches(Token::LParen) {
                        let args = self.parse_arguments()?;
                        return Ok(Value::FunctionCall {
                            object: Some(name),
                            module: None,
                            function: scoped_name,
                            args,
                        });
                    }

                    if name == "arg" {
                        Value::Variable(format!("arg.{}", scoped_name))
                    } else if name == "var" {
                        Value::Variable(format!("var.{}", scoped_name))
                    } else {
                        Value::Variable(format!("{}.", name))
                    }
                } else if self.matches(Token::LParen) {
                    let args = self.parse_arguments()?;
                    Value::FunctionCall {
                        object: None,
                        module: None,
                        function: name,
                        args,
                    }
                } else {
                    Value::Variable(name)
                }
            }
            Some(Token::LBracket) => self.parse_array_literal()?,
            Some(Token::LBrace) => self.parse_object_literal()?,
            Some(Token::LParen) => {
                self.index += 1;
                let value = self.parse_expression()?;
                self.expect(Token::RParen)?;
                value
            }
            _ => return Err(format!("Expected value, found {:?}", self.peek())),
        };

        while matches!(self.peek(), Some(Token::Dot)) {
            self.expect(Token::Dot)?;
            let key = self.consume_identifier()?;
            if self.matches(Token::LParen) {
                let args = self.parse_arguments()?;
                let object = match &value {
                    Value::Variable(name) => Some(name.clone()),
                    _ => None,
                };
                if let Some(object_name) = object {
                    value = Value::FunctionCall {
                        object: Some(object_name),
                        module: None,
                        function: key,
                        args,
                    };
                    break;
                }
                return Err(format!("Method call target is not resolvable: {:?}", value));
            }
            value = Value::Property {
                target: Box::new(value),
                key,
            };
        }

        while matches!(self.peek(), Some(Token::LBracket)) {
            self.expect(Token::LBracket)?;
            let index = self.parse_expression()?;
            self.expect(Token::RBracket)?;
            value = Value::Index {
                target: Box::new(value),
                index: Box::new(index),
            };
        }

        Ok(value)
    }

    fn parse_object_literal(&mut self) -> Result<Value, String> {
        self.expect(Token::LBrace)?;
        let mut object = HashMap::new();

        if !matches!(self.peek(), Some(Token::RBrace)) {
            loop {
                let key = match self.peek() {
                    Some(Token::StringLiteral(value)) => {
                        let key = value.clone();
                        self.index += 1;
                        key
                    }
                    _ => self.consume_identifier()?,
                };
                self.expect(Token::Colon)?;
                let value = self.parse_value()?;
                object.insert(key, value);

                if self.match_token(Token::Comma) {
                    continue;
                }

                if matches!(self.peek(), Some(Token::RBrace)) {
                    break;
                }

                if self.peek().is_some() {
                    continue;
                }
            }
        }

        self.expect(Token::RBrace)?;
        Ok(Value::Object(object))
    }

    fn parse_array_literal(&mut self) -> Result<Value, String> {
        self.expect(Token::LBracket)?;
        let mut items = Vec::new();

        if !matches!(self.peek(), Some(Token::RBracket)) {
            loop {
                items.push(self.parse_value()?);
                if self.match_token(Token::Comma) {
                    continue;
                }
                break;
            }
        }

        self.expect(Token::RBracket)?;
        Ok(Value::Array(items))
    }

    fn parse_comparison_operator(&mut self) -> Result<BinaryOperator, String> {
        match self.peek() {
            Some(Token::Equal) => {
                self.index += 1;
                if self.matches(Token::Equal) {
                    self.index += 1;
                    Ok(BinaryOperator::StrictEqual)
                } else {
                    Ok(BinaryOperator::Equal)
                }
            }
            Some(Token::NotEqual) => {
                self.index += 1;
                if self.matches(Token::Equal) {
                    self.index += 1;
                    Ok(BinaryOperator::StrictNotEqual)
                } else {
                    Ok(BinaryOperator::NotEqual)
                }
            }
            Some(Token::GreaterThan) => {
                self.index += 1;
                Ok(BinaryOperator::Greater)
            }
            Some(Token::LessThan) => {
                self.index += 1;
                Ok(BinaryOperator::Less)
            }
            Some(Token::GreaterEqual) => {
                self.index += 1;
                Ok(BinaryOperator::GreaterEqual)
            }
            Some(Token::LessEqual) => {
                self.index += 1;
                Ok(BinaryOperator::LessEqual)
            }
            _ => Err(format!("Expected comparison operator, found {:?}", self.peek())),
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
            Some(Token::Function) => {
                self.index += 1;
                Ok("function".to_string())
            }
            Some(Token::ArrayType) => {
                self.index += 1;
                Ok("array".to_string())
            }
            Some(Token::ObjectType) => {
                self.index += 1;
                Ok("object".to_string())
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

/// `declare()` is only meaningful before any module code runs, so it must
/// sit at file top level. Nested declares are a parse error.
fn validate_top_level_declares(program: &Program) -> Result<(), String> {
    for stmt in &program.statements {
        match stmt {
            Statement::Declare { .. } => {}
            _ => reject_nested_declare(stmt)?,
        }
    }
    Ok(())
}

fn reject_nested_declare(stmt: &Statement) -> Result<(), String> {
    let nested_err = || Err("declare() must be at file top level".to_string());
    match stmt {
        Statement::Declare { .. } => nested_err(),
        Statement::VariableDecl { value, .. } | Statement::GlobalDecl { value, .. } => {
            reject_nested_declare_in_value(value)
        }
        Statement::Event { body, .. } => reject_nested_declare_in_block(body),
        Statement::If {
            then_branch,
            else_if_branches,
            else_branch,
            ..
        } => {
            reject_nested_declare_in_block(then_branch)?;
            for (_, branch) in else_if_branches {
                reject_nested_declare_in_block(branch)?;
            }
            if let Some(else_body) = else_branch {
                reject_nested_declare_in_block(else_body)?;
            }
            Ok(())
        }
        Statement::While { body, .. } | Statement::ForIn { body, .. } => {
            reject_nested_declare_in_block(body)
        }
        Statement::Try { body, catches } => {
            reject_nested_declare_in_block(body)?;
            for catch in catches {
                reject_nested_declare_in_block(&catch.body)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn reject_nested_declare_in_block(statements: &[Statement]) -> Result<(), String> {
    for stmt in statements {
        reject_nested_declare(stmt)?;
    }
    Ok(())
}

fn reject_nested_declare_in_value(value: &Value) -> Result<(), String> {
    match value {
        Value::Function { body, .. } => reject_nested_declare_in_block(body),
        Value::Array(items) => {
            for item in items {
                reject_nested_declare_in_value(item)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for item in map.values() {
                reject_nested_declare_in_value(item)?;
            }
            Ok(())
        }
        Value::Binary { left, right, .. } => {
            reject_nested_declare_in_value(left)?;
            reject_nested_declare_in_value(right)
        }
        Value::Unary { value, .. } => reject_nested_declare_in_value(value),
        Value::Property { target, .. } | Value::Index { target, .. } => {
            reject_nested_declare_in_value(target)
        }
        Value::FunctionCall { args, .. } => {
            for arg in args {
                reject_nested_declare_in_value(arg)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Flow {
    Normal,
    Break,
    Continue,
    Return(Value),
}

/// Catchable typed error vs fatal (uncatchable) runtime failure.
/// `Fatal` aborts with `Runtime error: ...`; `Throw` carries a
/// `Value::Error` that the nearest enclosing `try` can catch.
#[derive(Debug, PartialEq, Clone)]
enum RuntimeFault {
    Fatal(String),
    Throw(Value),
}

impl From<String> for RuntimeFault {
    fn from(msg: String) -> Self {
        RuntimeFault::Fatal(msg)
    }
}

fn fatal_err(msg: impl Into<String>) -> RuntimeFault {
    RuntimeFault::Fatal(msg.into())
}

fn throw_err(error_type: &str, message: impl Into<String>) -> RuntimeFault {
    RuntimeFault::Throw(Value::Error {
        error_type: error_type.to_string(),
        message: message.into(),
    })
}

fn uncaught_message(err: &Value) -> String {
    match err {
        Value::Error { error_type, message } => format!("Uncaught {}: {}", error_type, message),
        other => format!("Uncaught error: {:?}", other),
    }
}

const MAX_LOOP_ITERS: usize = 1_000_000;

pub const ENTRY_ALIAS: &str = "__main__";

/// A loaded module: file-private top-level locals plus exported globals.
/// Globals are read live from here on every `var.x` / `alias:var.x` access.
#[derive(Debug, Clone)]
struct LoadedModule {
    path: PathBuf,
    top_locals: HashMap<String, Value>,
    globals: HashMap<String, Value>,
    global_names: HashSet<String>,
    program: Option<Program>,
    onstart_executed: bool,
}

impl LoadedModule {
    fn empty() -> Self {
        Self {
            path: PathBuf::new(),
            top_locals: HashMap::new(),
            globals: HashMap::new(),
            global_names: HashSet::new(),
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
}

impl Default for ModuleRuntime {
    fn default() -> Self {
        Self {
            modules: HashMap::new(),
            canonical_to_alias: HashMap::new(),
            loading: Vec::new(),
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

    /// Snapshot file-scope locals after top-level execution: every
    /// `var.*` entry that is not a declared global.
    fn snapshot_top_locals(&mut self, alias: &str, env: &HashMap<String, Value>) {
        self.ensure_module(alias);
        let global_names = self.modules[alias].global_names.clone();
        let module = self.modules.get_mut(alias).unwrap();
        module.top_locals.clear();
        for (k, v) in env {
            if let Some(short) = k.strip_prefix("var.") {
                if !global_names.contains(short) {
                    module.top_locals.insert(k.clone(), v.clone());
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
    for statement in &program.statements {
        if matches!(statement, Statement::Declare { .. }) {
            continue;
        }
        match execute_statement(statement, &mut environment, rt, alias) {
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
    rt.snapshot_top_locals(alias, &environment);
    Ok(())
}

/// Resolve + load one `declare()` import. Cached by canonical path;
/// a later `run=true` upgrades a previously `run=false` load.
fn declare_module(
    rt: &mut ModuleRuntime,
    importer_dir: &Path,
    raw: &str,
    run: bool,
) -> Result<(), String> {
    let joined = if raw.starts_with('/') {
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
    for stmt in &program.statements {
        if matches!(
            stmt,
            Statement::Declare { .. } | Statement::Event { .. }
        ) {
            continue;
        }
        match execute_statement(stmt, &mut env, rt, &alias) {
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
    rt.snapshot_top_locals(&alias, &env);
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
    for event in &events {
        match execute_statement(event, &mut env, rt, alias) {
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
    if let Some(module) = rt.modules.get_mut(alias) {
        for name in module.global_names.clone() {
            let key = format!("var.{}", name);
            if let Some(value) = env.get(&key).cloned() {
                module.globals.insert(key, value);
            }
        }
    }
    rt.snapshot_top_locals(alias, &env);
    if let Some(module) = rt.modules.get_mut(alias) {
        module.onstart_executed = true;
    }
    Ok(())
}

fn execute_block(
    statements: &[Statement],
    environment: &mut HashMap<String, Value>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Flow, RuntimeFault> {
    for stmt in statements {
        match execute_statement(stmt, environment, rt, cur)? {
            Flow::Normal => {}
            other => return Ok(other),
        }
    }
    Ok(Flow::Normal)
}

fn invoke_function(
    object: Option<&str>,
    module: Option<&str>,
    function: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
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
            Value::Function { params, body } => {
                let mut local_env = rt.module_merged_env(alias);
                let argument_array = Value::Array(
                    args.iter()
                        .map(|arg| resolve_value(arg, environment, rt, cur))
                        .collect::<Result<Vec<_>, RuntimeFault>>()?,
                );
                local_env.insert("pass".to_string(), Value::Null);
                local_env.insert("arg".to_string(), argument_array.clone());
                local_env.insert("args".to_string(), argument_array);
                for (param, arg) in params.iter().zip(args.iter()) {
                    let value = resolve_value(arg, environment, rt, cur)?;
                    local_env.insert(format!("arg.{}", param), value);
                }
                let mut result = None;
                for stmt in body {
                    match execute_statement(&stmt, &mut local_env, rt, alias)? {
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
                let value = resolve_value(arg, environment, rt, cur)?;
                rendered.push(format_value(value));
            }
            println!("{}", rendered.join(" "));
            Ok(None)
        }
        Some(obj) if obj == "math" => {
            let value = match args {
                [arg] => resolve_value(arg, environment, rt, cur)?,
                _ => return Err(throw_err("ValueError", format!("{} expects exactly one argument", function))),
            };
            let number = match value {
                Value::Number(value) => value,
                _ => return Err(throw_err("TypeError", format!("{} expects a numeric argument", function))),
            };

            match function {
                "Sin" => Ok(Some(Value::Number(number.sin()))),
                "Cos" => Ok(Some(Value::Number(number.cos()))),
                "Tan" => Ok(Some(Value::Number(number.tan()))),
                _ => Err(throw_err("NameError", format!("Unknown math function: {}", function))),
            }
        }
        _ => {
            let passed_value = match object {
                Some(obj) => resolve_value(&Value::Variable(obj.to_string()), environment, rt, cur)
                    .or_else(|_| lookup_scoped(obj, environment, rt, cur).ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", obj))))?,
                None => Value::Null,
            };

            if function.eq_ignore_ascii_case("EqualsCaseSensitive")
                || function.eq_ignore_ascii_case("caseSensitiveEquals")
                || function.eq_ignore_ascii_case("CaseSensitiveEquals")
            {
                let target = match object {
                    Some(obj) => resolve_value(&Value::Variable(obj.to_string()), environment, rt, cur)
                        .or_else(|_| lookup_scoped(obj, environment, rt, cur).ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", obj))))?,
                    None => return Err(throw_err("ValueError", "case-sensitive equality requires a target value")),
                };

                let rhs = match args {
                    [value] => resolve_value(value, environment, rt, cur)?,
                    _ => return Err(throw_err("ValueError", "case-sensitive equality expects exactly one argument")),
                };

                match (target, rhs) {
                    (Value::String(lhs), Value::String(rhs)) => return Ok(Some(Value::Logic(lhs == rhs))),
                    _ => return Err(throw_err("TypeError", "case-sensitive equality requires two strings")),
                }
            }

            let callee = resolve_value(&Value::Variable(function.to_string()), environment, rt, cur)
                .or_else(|_| {
                    lookup_scoped(function, environment, rt, cur).ok_or_else(|| throw_err("NameError", format!("Unknown variable: {}", function)))
                })?;

            match callee {
                Value::Function { params, body } => {
                    let mut local_env = environment.clone();
                    let argument_array = Value::Array(
                        args.iter()
                            .map(|arg| resolve_value(arg, environment, rt, cur))
                            .collect::<Result<Vec<_>, RuntimeFault>>()?,
                    );
                    local_env.insert("pass".to_string(), passed_value);
                    local_env.insert("arg".to_string(), argument_array.clone());
                    local_env.insert("args".to_string(), argument_array);
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let value = resolve_value(arg, environment, rt, cur)?;
                        let scoped = format!("arg.{}", param);
                        local_env.insert(scoped, value);
                    }

                    let mut result = None;
                    for stmt in body {
                        match execute_statement(&stmt, &mut local_env, rt, cur)? {
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

fn execute_statement(
    statement: &Statement,
    environment: &mut HashMap<String, Value>,
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Flow, RuntimeFault> {
    match statement {
        Statement::VariableDecl { name, value, .. } => {
            // Redeclare-as-assign: `var local x <type> = v` overwrites `var.x`
            // if it already exists, otherwise creates it. Type is not enforced.
            let resolved = match value {
                Value::Function { .. } => value.clone(),
                _ => resolve_value(value, environment, rt, cur)?,
            };
            environment.insert(format!("var.{}", name), resolved);
            Ok(Flow::Normal)
        }
        Statement::FunctionCall { object, module, function, args } => {
            let _ = invoke_function(object.as_deref(), module.as_deref(), function, args, environment, rt, cur)?;
            Ok(Flow::Normal)
        }
        Statement::GlobalDecl { name, value, .. } => {
            // `var global x <type> = v` writes the live module table (and
            // the local scope). Redeclare overwrites. Allowed in any block
            // so module functions can mutate their own globals.
            let resolved = match value {
                Value::Function { .. } => value.clone(),
                _ => resolve_value(value, environment, rt, cur)?,
            };
            environment.insert(format!("var.{}", name), resolved.clone());
            rt.ensure_module(cur);
            let module = rt.modules.get_mut(cur).unwrap();
            module.globals.insert(format!("var.{}", name), resolved);
            module.global_names.insert(name.clone());
            Ok(Flow::Normal)
        }
        Statement::Declare { .. } => {
            // Handled by the loader before execution; no-op here.
            Ok(Flow::Normal)
        }
        Statement::Return { value } => {
            let resolved = resolve_value(value, environment, rt, cur)?;
            Ok(Flow::Return(resolved))
        }
        Statement::Break => Ok(Flow::Break),
        Statement::Continue => Ok(Flow::Continue),
        Statement::Throw { error_type, message } => {
            let resolved_msg = resolve_value(message, environment, rt, cur)?;
            let text = match resolved_msg {
                Value::String(s) => s,
                other => format_value(other),
            };
            Err(throw_err(error_type, text))
        }
        Statement::Try { body, catches } => execute_try(body, catches, environment, rt, cur),
        Statement::Event { object, name, body } => {
            if object == "kal" && name == "OnStart" {
                match execute_block(body, environment, rt, cur)? {
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
            let condition_value = resolve_value(condition, environment, rt, cur)?;
            if is_truthy(&condition_value) {
                return execute_block(then_branch, environment, rt, cur);
            }

            for (else_if_condition, else_if_body) in else_if_branches {
                let branch_value = resolve_value(else_if_condition, environment, rt, cur)?;
                if is_truthy(&branch_value) {
                    return execute_block(else_if_body, environment, rt, cur);
                }
            }

            if let Some(else_body) = else_branch {
                return execute_block(else_body, environment, rt, cur);
            }
            Ok(Flow::Normal)
        }
        Statement::While { condition, body } => {
            for _ in 0..MAX_LOOP_ITERS {
                let condition_value = resolve_value(condition, environment, rt, cur)?;
                if !is_truthy(&condition_value) {
                    return Ok(Flow::Normal);
                }
                match execute_block(body, environment, rt, cur)? {
                    Flow::Normal => {}
                    Flow::Break => return Ok(Flow::Normal),
                    Flow::Continue => {}
                    Flow::Return(value) => return Ok(Flow::Return(value)),
                }
            }
            Err(fatal_err("possible infinite loop: while exceeded iteration limit"))
        }
        Statement::ForIn { var, iterable, body } => {
            let resolved_iterable = resolve_value(iterable, environment, rt, cur)?;
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
                match execute_block(body, environment, rt, cur)? {
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
    rt: &mut ModuleRuntime,
    cur: &str,
) -> Result<Flow, RuntimeFault> {
    match execute_block(body, environment, rt, cur) {
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
                    let result = execute_block(&catch.body, environment, rt, cur);
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
            let result = invoke_function(object.as_deref(), module.as_deref(), function, args, environment, rt, cur)?;
            match result {
                Some(value) => Ok(value),
                None => Ok(Value::Null),
            }
        }
        Value::Array(items) => Ok(Value::Array(
            items
                .iter()
                .map(|item| resolve_value(item, environment, rt, cur))
                .collect::<Result<Vec<_>, RuntimeFault>>()?,
        )),
        Value::Object(map) => {
            let mut resolved = HashMap::new();
            for (key, item) in map {
                resolved.insert(key.clone(), resolve_value(item, environment, rt, cur)?);
            }
            Ok(Value::Object(resolved))
        }
        Value::Property { target, key } => {
            let target_value = resolve_value(target, environment, rt, cur)?;
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
            let target_value = resolve_value(target, environment, rt, cur)?;
            let index_value = resolve_value(index, environment, rt, cur)?;
            match (target_value, index_value) {
                (Value::Array(items), Value::Number(index)) => {
                    let idx = index as usize;
                    items.get(idx)
                        .cloned()
                        .ok_or_else(|| throw_err("IndexError", format!("Index out of bounds: {}", idx)))
                }
                (Value::String(value), Value::Number(index)) => {
                    let idx = index as usize;
                    let ch = value
                        .chars()
                        .nth(idx)
                        .ok_or_else(|| throw_err("IndexError", format!("Index out of bounds: {}", idx)))?;
                    Ok(Value::String(ch.to_string()))
                }
                _ => Err(throw_err("TypeError", "Index requires an array or string with a numeric index")),
            }
        }
        Value::Binary { left, op, right } => {
            let left_value = resolve_value(left, environment, rt, cur)?;
            let right_value = resolve_value(right, environment, rt, cur)?;
            evaluate_binary(left_value, right_value, op)
        }
        Value::Unary { op, value } => {
            let inner = resolve_value(value, environment, rt, cur)?;
            evaluate_unary(inner, op)
        }
        _ => Ok(value.clone()),
    }
}

fn evaluate_unary(value: Value, op: &UnaryOperator) -> Result<Value, RuntimeFault> {
    match op {
        UnaryOperator::Not => Ok(Value::Logic(!is_truthy(&value))),
    }
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

    const TEST_ALIAS: &str = "__test__";

    fn eval(value: &Value, env: &HashMap<String, Value>) -> Result<Value, RuntimeFault> {
        let mut rt = ModuleRuntime::default();
        resolve_value(value, env, &mut rt, TEST_ALIAS)
    }

    fn call(
        object: Option<&str>,
        function: &str,
        args: &[Value],
        env: &HashMap<String, Value>,
    ) -> Result<Option<Value>, RuntimeFault> {
        let mut rt = ModuleRuntime::default();
        invoke_function(object, None, function, args, env, &mut rt, TEST_ALIAS)
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

        execute_statement(&first, &mut environment, &mut ModuleRuntime::default(), TEST_ALIAS).unwrap();
        execute_statement(&second, &mut environment, &mut ModuleRuntime::default(), TEST_ALIAS).unwrap();

        assert_eq!(environment.get("var.score"), Some(&Value::Number(15.0)));
    }

    #[test]
    fn function_call_used_as_value_returns_result() {
        let mut environment = HashMap::new();
        environment.insert(
            "var.add".to_string(),
            Value::Function {
                params: vec!["a".to_string(), "b".to_string()],
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
                globals: HashMap::from([
                    ("var.share".to_string(), Value::Number(53.0)),
                    (
                        "var.double".to_string(),
                        Value::Function {
                            params: vec!["x".to_string()],
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
        let decl = Statement::GlobalDecl {
            name: "share".to_string(),
            type_name: "number".to_string(),
            value: Value::Number(53.0),
        };
        execute_statement(&decl, &mut env, &mut rt, TEST_ALIAS).unwrap();
        // Live store read.
        let via_store = resolve_value(
            &Value::ModuleVar {
                alias: TEST_ALIAS.to_string(),
                name: "share".to_string(),
            },
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
        execute_statement(&redecl, &mut env, &mut rt, TEST_ALIAS).unwrap();
        let updated = resolve_value(
            &Value::Variable("var.share".to_string()),
            &env,
            &mut rt,
            TEST_ALIAS,
        )
        .unwrap();
        assert_eq!(updated, Value::Number(99.0));
    }
}
