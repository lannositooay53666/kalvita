// src/tools.rs
pub fn hurt() {
    println!("owuch!!");
}

pub fn heal() {
    println!("yeagh!!");
}

pub fn greet(name: String) { // Forces the caller to allocate a heap String
    println!("Hello, {}!", name);
}