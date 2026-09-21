use crate::lexer::{Lexer, Token};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
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
        function: String,
        args: Vec<Value>,
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
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    VariableDecl {
        name: String,
        type_name: String,
        value: Value,
    },
    FunctionCall {
        object: Option<String>,
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
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::Identifier(_)) => {
                let name = self.consume_identifier()?;

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
                        function,
                        args,
                    });
                }

                if self.matches(Token::LParen) {
                    let args = self.parse_arguments()?;
                    return Ok(Statement::FunctionCall {
                        object: None,
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

    fn parse_variable_decl(&mut self) -> Result<Statement, String> {
        self.expect(Token::Var)?;
        self.expect(Token::Local)?;
        let name = self.consume_identifier()?;
        let type_name = self.consume_type_name()?;
        self.expect(Token::Assign)?;
        let value = if self.matches(Token::LParen) {
            self.parse_function_literal()?
        } else {
            self.parse_value()?
        };
        Ok(Statement::VariableDecl {
            name,
            type_name,
            value,
        })
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
                if self.matches(Token::Dot) {
                    self.expect(Token::Dot)?;
                    let scoped_name = self.consume_identifier()?;
                    if self.matches(Token::LParen) {
                        let args = self.parse_arguments()?;
                        return Ok(Value::FunctionCall {
                            object: Some(name),
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
            Some(Token::NotEqual) => {
                self.index += 1;
                Ok(BinaryOperator::NotEqual)
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

pub fn execute(program: &Program) -> Result<(), String> {
    let mut environment: HashMap<String, Value> = HashMap::new();

    for statement in &program.statements {
        execute_statement(statement, &mut environment)?;
    }

    Ok(())
}

fn invoke_function(
    object: Option<&str>,
    function: &str,
    args: &[Value],
    environment: &HashMap<String, Value>,
) -> Result<Option<Value>, String> {
    match object {
        Some(obj) if obj == "con" && function == "Print" => {
            let mut rendered = Vec::new();
            for arg in args {
                let value = resolve_value(arg, environment)?;
                rendered.push(format_value(value));
            }
            println!("{}", rendered.join(" "));
            Ok(None)
        }
        Some(obj) if obj == "math" => {
            let value = match args {
                [arg] => resolve_value(arg, environment)?,
                _ => return Err(format!("{} expects exactly one argument", function)),
            };
            let number = match value {
                Value::Number(value) => value,
                _ => return Err(format!("{} expects a numeric argument", function)),
            };

            match function {
                "Sin" => Ok(Some(Value::Number(number.sin()))),
                "Cos" => Ok(Some(Value::Number(number.cos()))),
                "Tan" => Ok(Some(Value::Number(number.tan()))),
                _ => Err(format!("Unknown math function: {}", function)),
            }
        }
        _ => {
            let passed_value = match object {
                Some(obj) => resolve_value(&Value::Variable(obj.to_string()), environment)
                    .or_else(|_| environment.get(&format!("var.{}", obj)).cloned().ok_or_else(|| format!("Unknown variable: {}", obj)))?,
                None => Value::Null,
            };

            let callee = resolve_value(&Value::Variable(function.to_string()), environment)
                .or_else(|_| {
                    let prefixed = format!("var.{}", function);
                    environment.get(&prefixed).cloned().ok_or_else(|| format!("Unknown variable: {}", function))
                })?;

            match callee {
                Value::Function { params, body } => {
                    let mut local_env = environment.clone();
                    let argument_array = Value::Array(
                        args.iter()
                            .map(|arg| resolve_value(arg, environment))
                            .collect::<Result<Vec<_>, String>>()?,
                    );
                    local_env.insert("pass".to_string(), passed_value);
                    local_env.insert("arg".to_string(), argument_array.clone());
                    local_env.insert("args".to_string(), argument_array);
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let value = resolve_value(arg, environment)?;
                        let scoped = format!("arg.{}", param);
                        local_env.insert(scoped, value);
                    }

                    let mut result = None;
                    for stmt in body {
                        match stmt {
                            Statement::Return { value } => {
                                result = Some(resolve_value(&value, &local_env)?);
                                break;
                            }
                            _ => execute_statement(&stmt, &mut local_env)?,
                        }
                    }
                    Ok(result)
                }
                _ => Err(format!("{} is not callable", function)),
            }
        }
    }
}

fn execute_statement(statement: &Statement, environment: &mut HashMap<String, Value>) -> Result<(), String> {
    match statement {
        Statement::VariableDecl { name, value, .. } => {
            let resolved = match value {
                Value::Function { .. } => value.clone(),
                _ => resolve_value(value, environment)?,
            };
            environment.insert(format!("var.{}", name), resolved);
        }
        Statement::FunctionCall { object, function, args } => {
            let _ = invoke_function(object.as_deref(), function, args, environment)?;
        }
        Statement::Return { value } => {
            return Err("return can only be used inside a function body".to_string());
        }
        Statement::Event { object, name, body } => {
            if object == "kal" && name == "OnStart" {
                for inner in body {
                    execute_statement(inner, environment)?;
                }
            }
        }
        Statement::If {
            condition,
            then_branch,
            else_if_branches,
            else_branch,
        } => {
            let condition_value = resolve_value(condition, environment)?;
            if is_truthy(&condition_value) {
                for stmt in then_branch {
                    execute_statement(stmt, environment)?;
                }
                return Ok(());
            }

            for (else_if_condition, else_if_body) in else_if_branches {
                let branch_value = resolve_value(else_if_condition, environment)?;
                if is_truthy(&branch_value) {
                    for stmt in else_if_body {
                        execute_statement(stmt, environment)?;
                    }
                    return Ok(());
                }
            }

            if let Some(else_body) = else_branch {
                for stmt in else_body {
                    execute_statement(stmt, environment)?;
                }
            }
        }
    }

    Ok(())
}

fn resolve_variable_name(name: &str, environment: &HashMap<String, Value>) -> Option<Value> {
    if name == "pass" {
        return environment.get("pass").cloned();
    }

    if name == "arg" || name == "args" {
        return environment.get(name).cloned();
    }

    if name.starts_with("var.") || name.starts_with("arg.") {
        return environment.get(name).cloned();
    }

    None
}

fn resolve_value(value: &Value, environment: &HashMap<String, Value>) -> Result<Value, String> {
    match value {
        Value::Variable(name) => resolve_variable_name(name, environment)
            .ok_or_else(|| format!("Unknown variable: {}", name)),
        Value::FunctionCall { object, function, args } => {
            let result = invoke_function(object.as_deref(), function, args, environment)?;
            match result {
                Some(value) => Ok(value),
                None => Ok(Value::Null),
            }
        }
        Value::Array(items) => Ok(Value::Array(
            items
                .iter()
                .map(|item| resolve_value(item, environment))
                .collect::<Result<Vec<_>, String>>()?,
        )),
        Value::Object(map) => {
            let mut resolved = HashMap::new();
            for (key, item) in map {
                resolved.insert(key.clone(), resolve_value(item, environment)?);
            }
            Ok(Value::Object(resolved))
        }
        Value::Property { target, key } => {
            let target_value = resolve_value(target, environment)?;
            match target_value {
                Value::Object(map) => map.get(key).cloned().ok_or_else(|| format!("Unknown property: {}", key)),
                Value::Null => Ok(Value::Null),
                other => Err(format!("Property access requires an object, got {:?}", other)),
            }
        }
        Value::Index { target, index } => {
            let target_value = resolve_value(target, environment)?;
            let index_value = resolve_value(index, environment)?;
            match (target_value, index_value) {
                (Value::Array(items), Value::Number(index)) => {
                    let idx = index as usize;
                    items.get(idx)
                        .cloned()
                        .ok_or_else(|| format!("Index out of bounds: {}", idx))
                }
                (Value::String(value), Value::Number(index)) => {
                    let idx = index as usize;
                    let ch = value
                        .chars()
                        .nth(idx)
                        .ok_or_else(|| format!("Index out of bounds: {}", idx))?;
                    Ok(Value::String(ch.to_string()))
                }
                _ => Err("Index requires an array or string with a numeric index".to_string()),
            }
        }
        Value::Binary { left, op, right } => {
            let left_value = resolve_value(left, environment)?;
            let right_value = resolve_value(right, environment)?;
            evaluate_binary(left_value, right_value, op)
        }
        Value::Unary { op, value } => {
            let inner = resolve_value(value, environment)?;
            evaluate_unary(inner, op)
        }
        _ => Ok(value.clone()),
    }
}

fn evaluate_unary(value: Value, op: &UnaryOperator) -> Result<Value, String> {
    match op {
        UnaryOperator::Not => Ok(Value::Logic(!is_truthy(&value))),
    }
}

fn evaluate_binary(left: Value, right: Value, op: &BinaryOperator) -> Result<Value, String> {
    match op {
        BinaryOperator::Add => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
            (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            (Value::String(a), Value::Number(b)) => Ok(Value::String(format!("{}{}", a, b))),
            (Value::Number(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            _ => Err("Addition requires numbers or strings".to_string()),
        },
        BinaryOperator::Subtract => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a - b)),
            _ => Err("Subtraction requires numbers".to_string()),
        },
        BinaryOperator::Multiply => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a * b)),
            _ => Err("Multiplication requires numbers".to_string()),
        },
        BinaryOperator::Divide => match (left, right) {
            (Value::Array(left_items), Value::Array(right_items)) => apply_array_array_op(left_items, right_items, op),
            (Value::Array(items), scalar) => apply_array_scalar_op(items, scalar, op),
            (scalar, Value::Array(items)) => apply_array_scalar_op(items, scalar, op),
            (Value::Number(a), Value::Number(b)) if b != 0.0 => Ok(Value::Number(a / b)),
            _ => Err("Division requires non-zero numbers".to_string()),
        },
        BinaryOperator::Equal => Ok(Value::Logic(left == right)),
        BinaryOperator::NotEqual => Ok(Value::Logic(left != right)),
        BinaryOperator::And => Ok(Value::Logic(is_truthy(&left) && is_truthy(&right))),
        BinaryOperator::Or => Ok(Value::Logic(is_truthy(&left) || is_truthy(&right))),
        BinaryOperator::Greater => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a > b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a > b)),
            _ => Err("Greater-than requires comparable values".to_string()),
        },
        BinaryOperator::Less => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a < b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a < b)),
            _ => Err("Less-than requires comparable values".to_string()),
        },
        BinaryOperator::GreaterEqual => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a >= b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a >= b)),
            _ => Err("Greater-or-equal requires comparable values".to_string()),
        },
        BinaryOperator::LessEqual => match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Logic(a <= b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Logic(a <= b)),
            _ => Err("Less-or-equal requires comparable values".to_string()),
        },
    }
}

fn apply_array_scalar_op(items: Vec<Value>, scalar: Value, op: &BinaryOperator) -> Result<Value, String> {
    let scalar_number = match scalar {
        Value::Number(value) => value,
        _ => return Err(format!("Array math requires a numeric scalar, got {:?}", scalar)),
    };

    let mut result = Vec::new();
    for item in items {
        let current = match item {
            Value::Number(value) => value,
            _ => return Err("Array math only works on numeric arrays".to_string()),
        };

        let transformed = match op {
            BinaryOperator::Add => current + scalar_number,
            BinaryOperator::Subtract => current - scalar_number,
            BinaryOperator::Multiply => current * scalar_number,
            BinaryOperator::Divide if scalar_number != 0.0 => current / scalar_number,
            BinaryOperator::Divide => return Err("Division by zero in array math".to_string()),
            _ => return Err("Unsupported array operation".to_string()),
        };
        result.push(Value::Number(transformed));
    }

    Ok(Value::Array(result))
}

fn apply_array_array_op(left_items: Vec<Value>, right_items: Vec<Value>, op: &BinaryOperator) -> Result<Value, String> {
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
        Value::Array(items) => {
            let rendered: Vec<String> = items.into_iter().map(format_value).collect();
            format!("[{}]", rendered.join(", "))
        }
        Value::FunctionCall { .. } => "<function-call>".to_string(),
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

        execute_statement(&first, &mut environment).unwrap();
        execute_statement(&second, &mut environment).unwrap();

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

        let result = resolve_value(
            &Value::FunctionCall {
                object: None,
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

        let resolved = resolve_value(&Value::Variable("var.score".to_string()), &environment).unwrap();
        assert_eq!(resolved, Value::Number(15.0));
        let arg_a = resolve_value(&Value::Variable("arg.a".to_string()), &environment).unwrap();
        assert_eq!(arg_a, Value::Number(7.0));
    }

    #[test]
    fn naked_variables_are_rejected() {
        let mut environment = HashMap::new();
        environment.insert("var.score".to_string(), Value::Number(15.0));

        let result = resolve_value(&Value::Variable("score".to_string()), &environment);
        assert!(result.is_err());
    }

    #[test]
    fn logic_gate_operations_work() {
        let mut environment = HashMap::new();
        environment.insert("var.active".to_string(), Value::Logic(true));
        environment.insert("var.ready".to_string(), Value::Logic(true));
        environment.insert("var.blocked".to_string(), Value::Logic(false));

        let and_result = resolve_value(
            &Value::Binary {
                left: Box::new(Value::Variable("var.active".to_string())),
                op: BinaryOperator::And,
                right: Box::new(Value::Variable("var.ready".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(and_result, Value::Logic(true));

        let or_result = resolve_value(
            &Value::Binary {
                left: Box::new(Value::Variable("var.blocked".to_string())),
                op: BinaryOperator::Or,
                right: Box::new(Value::Variable("var.active".to_string())),
            },
            &environment,
        )
        .unwrap();
        assert_eq!(or_result, Value::Logic(true));

        let not_result = resolve_value(
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

        let sin_result = invoke_function(Some("math"), "Sin", &[Value::Number(0.0)], &environment).unwrap().unwrap();
        match sin_result {
            Value::Number(value) => assert!((value - 0.0).abs() < 1e-9),
            _ => panic!("Sin should return a number"),
        }

        let cos_result = invoke_function(Some("math"), "Cos", &[Value::Number(0.0)], &environment).unwrap().unwrap();
        match cos_result {
            Value::Number(value) => assert!((value - 1.0).abs() < 1e-9),
            _ => panic!("Cos should return a number"),
        }

        let tan_result = invoke_function(Some("math"), "Tan", &[Value::Number(0.0)], &environment).unwrap().unwrap();
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

        let result = invoke_function(Some("enemy"), "attack", &[], &environment).unwrap().unwrap();
        assert_eq!(result, Value::Number(40.0));

        let null_result = invoke_function(None, "attack", &[], &environment).unwrap().unwrap();
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

        let result = resolve_value(
            &Value::FunctionCall {
                object: None,
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

        let first = resolve_value(&Value::Index {
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
}
