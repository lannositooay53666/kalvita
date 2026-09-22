use crate::lexer::{Lexer, Token};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Range,
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
    BitNot,
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
        defaults: HashMap<String, Value>,
        body: Vec<Statement>,
    },
    Error {
        error_type: String,
        message: String,
    },
    File {
        path: String,
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
    Assign {
        target: AssignTarget,
        op: Option<BinaryOperator>,
        value: Box<Value>,
    },
    Switch {
        scrutinee: Box<Value>,
        cases: Vec<(Value, Vec<Statement>)>,
        default: Option<Vec<Statement>>,
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

/// Assignment target: `var.x`, `var.obj.key`, `var.arr[i]` (chains nest).
#[derive(Debug, PartialEq, Clone)]
pub enum AssignTarget {
    Var(String),
    Property { target: Box<AssignTarget>, key: String },
    Index { target: Box<AssignTarget>, index: Box<Value> },
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
            Some(Token::Var) => {
                match self.tokens.get(self.index + 1) {
                    Some(Token::Local) | Some(Token::Global) => self.parse_variable_decl(),
                    _ => self.parse_assign_statement(),
                }
            }
            Some(Token::Switch) => self.parse_switch_statement(),
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
        // `(params) { body }` is a function literal, but `(expr)` is a
        // parenthesized value — disambiguate with lookahead for `{`.
        let value = match self.try_function_header() {
            Some((params, defaults, end)) => {
                self.index = end;
                let body = self.parse_block_contents()?;
                Value::Function {
                    params,
                    defaults,
                    body,
                }
            }
            None => self.parse_value()?,
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

    fn parse_assign_statement(&mut self) -> Result<Statement, String> {
        // `var.x = v`, `var.x += v`, `var.arr[i] = v`, `var.obj.k -= v`.
        // Plain `=` requires the name to exist; compound ops read then write.
        self.expect(Token::Var)?;
        self.expect(Token::Dot)?;
        let name = self.consume_identifier()?;
        let mut target = AssignTarget::Var(name);
        loop {
            if self.matches(Token::Dot) {
                self.expect(Token::Dot)?;
                let key = self.consume_identifier()?;
                target = AssignTarget::Property {
                    target: Box::new(target),
                    key,
                };
            } else if self.matches(Token::LBracket) {
                self.expect(Token::LBracket)?;
                let index = self.parse_expression()?;
                self.expect(Token::RBracket)?;
                target = AssignTarget::Index {
                    target: Box::new(target),
                    index: Box::new(index),
                };
            } else {
                break;
            }
        }
        let op = if self.match_token(Token::Assign) {
            None
        } else if self.match_token(Token::PlusAssign) {
            Some(BinaryOperator::Add)
        } else if self.match_token(Token::MinusAssign) {
            Some(BinaryOperator::Subtract)
        } else if self.match_token(Token::StarAssign) {
            Some(BinaryOperator::Multiply)
        } else if self.match_token(Token::SlashAssign) {
            Some(BinaryOperator::Divide)
        } else {
            return Err(format!("Expected assignment operator, found {:?}", self.peek()));
        };
        let value = self.parse_expression()?;
        Ok(Statement::Assign {
            target,
            op,
            value: Box::new(value),
        })
    }

    fn parse_switch_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::Switch)?;
        self.expect(Token::LParen)?;
        let scrutinee = self.parse_expression()?;
        self.expect(Token::RParen)?;
        self.expect(Token::LBrace)?;
        let mut cases = Vec::new();
        let mut default = None;
        while !matches!(self.peek(), Some(Token::RBrace) | Some(Token::Eof)) {
            if self.match_token(Token::Case) {
                self.expect(Token::LParen)?;
                let value = self.parse_expression()?;
                self.expect(Token::RParen)?;
                self.expect(Token::LBrace)?;
                let body = self.parse_block_contents()?;
                cases.push((value, body));
            } else if self.match_token(Token::Default) {
                if default.is_some() {
                    return Err("switch allows only one default block".to_string());
                }
                self.expect(Token::LBrace)?;
                default = Some(self.parse_block_contents()?);
            } else {
                return Err(format!(
                    "Expected case or default in switch, found {:?}",
                    self.peek()
                ));
            }
        }
        self.expect(Token::RBrace)?;
        Ok(Statement::Switch {
            scrutinee: Box::new(scrutinee),
            cases,
            default,
        })
    }

    /// Lookahead: `(params) {` → function header. Returns params,
    /// defaults, and the index just past `{`. Runs on cloned tokens so a
    /// failed probe (e.g. `(0 - 8) >> 2`) leaves the parser untouched.
    fn try_function_header(&self) -> Option<(Vec<String>, HashMap<String, Value>, usize)> {
        let mut trial = Parser {
            tokens: self.tokens.clone(),
            index: self.index,
        };
        let parsed = (|| -> Result<(Vec<String>, HashMap<String, Value>, usize), String> {
            trial.expect(Token::LParen)?;
            let (params, defaults) = trial.parse_parameter_list()?;
            trial.expect(Token::RParen)?;
            trial.expect(Token::LBrace)?;
            Ok((params, defaults, trial.index))
        })();
        parsed.ok()
    }

    fn parse_parameter_list(&mut self) -> Result<(Vec<String>, HashMap<String, Value>), String> {
        let mut params = Vec::new();
        let mut defaults = HashMap::new();
        if !matches!(self.peek(), Some(Token::RParen)) {
            loop {
                let name = self.consume_identifier()?;
                if self.match_token(Token::Assign) {
                    let default = self.parse_expression()?;
                    defaults.insert(name.clone(), default);
                    params.push(name);
                } else {
                    if !defaults.is_empty() {
                        return Err(format!(
                            "Required parameter '{}' must come before defaulted parameters",
                            name
                        ));
                    }
                    params.push(name);
                }
                if !self.match_token(Token::Comma) {
                    break;
                }
            }
        }
        Ok((params, defaults))
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
        let mut left = self.parse_bitor()?;
        while matches!(self.peek(), Some(Token::And)) {
            self.index += 1;
            let right = self.parse_bitor()?;
            left = Value::Binary {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> Result<Value, String> {
        let mut left = self.parse_bitxor()?;
        while matches!(self.peek(), Some(Token::BitOr)) {
            self.index += 1;
            let right = self.parse_bitxor()?;
            left = Value::Binary {
                left: Box::new(left),
                op: BinaryOperator::BitOr,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<Value, String> {
        let mut left = self.parse_bitand()?;
        while matches!(self.peek(), Some(Token::BitXor)) {
            self.index += 1;
            let right = self.parse_bitand()?;
            left = Value::Binary {
                left: Box::new(left),
                op: BinaryOperator::BitXor,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<Value, String> {
        let mut left = self.parse_unary()?;
        while matches!(self.peek(), Some(Token::BitAnd)) {
            self.index += 1;
            let right = self.parse_unary()?;
            left = Value::Binary {
                left: Box::new(left),
                op: BinaryOperator::BitAnd,
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
        if matches!(self.peek(), Some(Token::BitNot)) {
            self.index += 1;
            let value = self.parse_unary()?;
            return Ok(Value::Unary {
                op: UnaryOperator::BitNot,
                value: Box::new(value),
            });
        }

        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Value, String> {
        let mut left = self.parse_shift()?;
        while matches!(
            self.peek(),
            Some(Token::Equal)
                | Some(Token::StrictEqual)
                | Some(Token::NotEqual)
                | Some(Token::StrictNotEqual)
                | Some(Token::GreaterThan)
                | Some(Token::LessThan)
                | Some(Token::GreaterEqual)
                | Some(Token::LessEqual)
        ) {
            let op = self.parse_comparison_operator()?;
            let right = self.parse_shift()?;
            left = Value::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Value, String> {
        let mut left = self.parse_range()?;
        while matches!(self.peek(), Some(Token::Shl) | Some(Token::Shr)) {
            let op = if self.matches(Token::Shl) {
                self.index += 1;
                BinaryOperator::Shl
            } else if self.matches(Token::Shr) {
                self.index += 1;
                BinaryOperator::Shr
            } else {
                return Err(format!("Expected operator, found {:?}", self.peek()));
            };
            let right = self.parse_range()?;
            left = Value::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_range(&mut self) -> Result<Value, String> {
        let mut left = self.parse_additive()?;
        while matches!(self.peek(), Some(Token::DotDot)) {
            self.index += 1;
            let right = self.parse_additive()?;
            left = Value::Binary {
                left: Box::new(left),
                op: BinaryOperator::Range,
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
                desugar_interpolated_string(&value)?
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
                Ok(BinaryOperator::Equal)
            }
            Some(Token::StrictEqual) => {
                self.index += 1;
                Ok(BinaryOperator::StrictEqual)
            }
            Some(Token::NotEqual) => {
                self.index += 1;
                Ok(BinaryOperator::NotEqual)
            }
            Some(Token::StrictNotEqual) => {
                self.index += 1;
                Ok(BinaryOperator::StrictNotEqual)
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
            // `file` stays a plain identifier (so `file.Read(...)` keeps
            // parsing as a namespaced call) and is only special here.
            Some(Token::Identifier(name)) if name == "file" => {
                self.index += 1;
                Ok("file".to_string())
            }
            Some(Token::Identifier(name)) => {
                return Err(format!(
                    "Expected type name (string, number, logic, null, array, object, file, function), found '{}'",
                    name
                ));
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

/// `"hello ${var.name}!"` desugars to `"hello " + var.name + "!"`.
/// Uses `+` concat (string + number supported). Unclosed `${` is an error.
fn desugar_interpolated_string(raw: &str) -> Result<Value, String> {
    if !raw.contains("${") {
        return Ok(Value::String(raw.to_string()));
    }
    let chars: Vec<char> = raw.chars().collect();
    let mut parts: Vec<Value> = Vec::new();
    let mut literal = String::new();
    let mut i = 0;
    let flush = |literal: &mut String, parts: &mut Vec<Value>| {
        if !literal.is_empty() {
            parts.push(Value::String(std::mem::take(literal)));
        }
    };
    while i < chars.len() {
        if chars[i] == '$' && chars.get(i + 1) == Some(&'{') {
            flush(&mut literal, &mut parts);
            // Find the matching close brace (nesting-aware for `{...}`).
            let mut depth = 1;
            let mut j = i + 2;
            while j < chars.len() && depth > 0 {
                if chars[j] == '{' {
                    depth += 1;
                } else if chars[j] == '}' {
                    depth -= 1;
                }
                j += 1;
            }
            if depth != 0 {
                return Err("Unclosed ${ in string interpolation".to_string());
            }
            let inner: String = chars[i + 2..j - 1].iter().collect();
            if inner.trim().is_empty() {
                return Err("Empty ${} in string interpolation".to_string());
            }
            let mut inner_lexer = Lexer::new(&inner);
            let tokens = inner_lexer.tokenize();
            let mut inner_parser = Parser {
                tokens,
                index: 0,
            };
            let expr = inner_parser.parse_expression()?;
            if !matches!(inner_parser.peek(), Some(Token::Eof)) {
                return Err(format!(
                    "Invalid expression in string interpolation: '{}'",
                    inner
                ));
            }
            parts.push(expr);
            i = j;
        } else {
            literal.push(chars[i]);
            i += 1;
        }
    }
    flush(&mut literal, &mut parts);
    if parts.is_empty() {
        return Ok(Value::String(String::new()));
    }
    let mut iter = parts.into_iter();
    let mut acc = iter.next().unwrap();
    for part in iter {
        acc = Value::Binary {
            left: Box::new(acc),
            op: BinaryOperator::Add,
            right: Box::new(part),
        };
    }
    Ok(acc)
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
        Statement::Assign { target, value, .. } => {
            reject_nested_declare_in_target(target)?;
            reject_nested_declare_in_value(value)
        }
        Statement::Switch {
            scrutinee,
            cases,
            default,
            ..
        } => {
            reject_nested_declare_in_value(scrutinee)?;
            for (case_value, body) in cases {
                reject_nested_declare_in_value(case_value)?;
                reject_nested_declare_in_block(body)?;
            }
            if let Some(default_body) = default {
                reject_nested_declare_in_block(default_body)?;
            }
            Ok(())
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

fn reject_nested_declare_in_target(target: &AssignTarget) -> Result<(), String> {
    match target {
        AssignTarget::Var(_) => Ok(()),
        AssignTarget::Property { target, .. } => reject_nested_declare_in_target(target),
        AssignTarget::Index { target, index } => {
            reject_nested_declare_in_target(target)?;
            reject_nested_declare_in_value(index)
        }
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

