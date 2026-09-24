# Kalvita

A small array-first, expression-driven scripting language, implemented in Rust
as a tree-walk interpreter: lexer → parser → interpreter.

```kal
[SCRIPTTYPE KALVITA VERSION 1]
kal.OnStart {
    var local names array = ["nova", "kael", "lyra"]
    for (name in var.names) {
        con.Print("hi ${var.name}!")
    }
}
```

## Quickstart

```sh
cargo run -- run sample.kal     # run a script
cargo run -- check file.kal     # parse only
cargo run -- fmt file.kal       # print formatted source (or --write)
cargo run -- test tests         # golden tests (*.kal vs *.expected)
cargo run -- repl               # interactive session (:help, :clear, :quit)
cargo run -- <script>           # kal.toml [scripts] shortcut
cargo test                      # Rust unit tests
```

Bare `cargo run` uses `kal.toml`'s `main`, else `sample.kal`.
`kalvita file.kal` also works.

## The language in 60 seconds

* `var local x number = 7` (private) and `var global y string = "hi"`
  (exported). Reads use the `var.` prefix: `var.x + 1`.
* Assignment without redeclaring: `var.x = 10`, `var.x += 2`,
  `var.arr[0] = 99`.
* Functions are values: `var local add function = (a, b) { return(arg.a +
  arg.b) }`, called bare — `add(1, 2)`. Defaults allowed: `(who = "World")`.
* Modules: `declare("./math.kal")` (alias = file stem), then
  `math:var.share` and `math:double(21)`. Second arg `false` loads globals
  without running `kal.OnStart`.
* Control flow: `if/elseif/else`, `while`, `for (x in arr|string|range)`,
  `switch/case/default`, `break`, `continue`.
* Typed errors: `try { ... catch (IndexError) { } }` with `var.err`
  (`{type, message}`) and `throw(MyErr, "boom")`.
* Bitwise `& | ^ ~ << >>`, ranges `0..10`, `"hi ${var.name}"` interpolation,
  strict `===` alongside case-insensitive `==`.
* Builtins: `con.Print/Input`, `math.*`, `str.*` (incl. `Match` glob,
  `Lines`, `ParseInt`), `arr.*`, `time.Now/Format`, `file.Read/Write/Append/
  Exists/ListDir/MkDir/Remove`, `sys.Args/Getenv/Cwd`, `json.Parse/Stringify`,
  `http.Get/Post`, `db.Open/Exec/Query/Close`, `gui.PickFile/Message`,
  bare `assert(cond[, msg])` (catchable `AssertError`).
* GUI widgets: `var local w window = "Title"`, `var local b button = "Go"`,
  `b.AttachToWindow(var.w)`, `b.SetPos(x, y)`, `w.OnStart/w.OnExit` +
  `b.OnClick` event blocks, `w.Run()`. See `examples/gui_demo.kal`.

The full reference lives in [SPEC.md](SPEC.md); runnable examples in
`sample.kal` and `tests/`.

## Project layout

```
src/lexer.rs         tokens (keywords, numbers, strings, comments)
src/parser.rs        AST + parser (precedence, desugar, validation)
src/interpreter.rs   tree-walk runtime (modules, builtins, unit tests)
src/error.rs         RuntimeFault: fatal vs catchable errors
src/main.rs          CLI: run/check/fmt/test/repl
tests/               golden scripts (*.kal vs *.expected)
SPEC.md              normative language reference
kal.toml             project file (name, version, main)
sample.kal           demo script
```

## License

Dual-licensed under MIT or Apache-2.0 — see `LICENSE-MIT` and
`LICENSE-APACHE`. Same terms as the Rust ecosystem itself.
