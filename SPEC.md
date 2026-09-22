# Kalvita Language Spec (v0.1.0, `VERSION 1`)

Array-first, expression-driven scripting language. Interpreter: lexer → parser → tree-walk runtime (`src/`).

## 1. Files

Every file starts with a header:

```kal
[SCRIPTTYPE KALVITA VERSION 1]
```

`kal.OnStart { ... }` is the entry event. Comments: `// line` and `.// block //.`
are stripped by the lexer.

## 2. Variables

```kal
var local score number = 7       // file/function-private
var global share number = 53     // exported to importers
var.score = 15                   // assignment (must already exist)
var.score += 5                   // compound: += -= *= /=
var.hero.health = 50             // nested property/index assignment
var.pair[0] = 99
```

Types (`string number logic null array object file function`) are enforced at
write time: declaring, redeclaring, or assigning a mismatched value is a
catchable `TypeError` (`expected number for var.n, got string`). `null`
fits every slot. Redeclaring with a new type resets the slot. Anything else
as a type name is a parse error. Loop variables, `catch (var.err)`, and
function params are untyped; array elements and object properties are
unchecked (collections are unparameterized). Reads require the `var.`
prefix; bare names are `NameError`. Redeclaring still overwrites values
(redeclare-as-assign).

## 3. Functions

```kal
var local add function = (a, b) { return(arg.a + arg.b) }
var local greet function = (who = "World") { return("hey ${arg.who}") }
add(1, 2)                        // bare call, own file only
```

Params bind to `arg.<name>`; all args also arrive as the `arg`/`args` array
(extras allowed). Defaults fill missing args; missing required args are
`ValueError`. No `return` → `null`. Max call depth: 64 (fatal).

## 4. Modules

```kal
declare("./math.kal")            // alias = file stem; cached; top-level only
declare("./quiet.kal", false)    // load globals, skip its kal.OnStart
declare("@/lib/x.kal")           // project-root (kal.toml) relative
con.Print(math:var.share)        // data read
con.Print(math:double(21))       // function call (no `var.`)
```

`var global` is the export list. Reads/calls are live: module functions
mutating their own globals are visible to importers. Plain `var.x` never
falls through into imports. Load runs the whole file (depth-first);
`run=false` skips only `kal.OnStart` (a later `run=true` upgrades).
Cycles, duplicate stems, bad stems (`[A-Za-z_][A-Za-z0-9_]*`), and missing
files are fatal `Module error`s. Unknown alias/name at runtime is a
catchable `NameError`.

## 5. Control flow

```kal
if (a > 1) { } elseif (b) { } else { }
while (var.i < 10) { }
for (x in var.items) { }         // arrays, strings (chars), ranges
for (i in 0..5) { }              // 0..=4, end-exclusive, empty if start >= end
switch (var.v) {
    case (1) { }                 // first strict (===) match wins, no fallthrough
    default { }
}
break                            // loops only
continue
```

`break`/`continue`/`return` misuse is fatal. Infinite `while` aborts after
1M iterations (fatal).

## 6. Errors

```kal
try {
    var local x number = var.arr[99]
    throw(MyErr, "boom")
    catch (IndexError) { con.Print(var.err.message) }
    catch (MyErr) { con.Print(var.err.type) }
    catch (Error) { con.Print("catch-all") }   // Error matches any type
}
```

`catch` blocks nest **inside** `try`, first match wins; `var.err` (`{type,
message}`) is bound per handler. Runtime faults: `IndexError`, `DivZero`,
`TypeError`, `ValueError`, `NameError`, `IOError` + custom `throw` types.
Uncaught → `Runtime error: Uncaught Type: msg`.

## 7. Operators (precedence, loose → tight)

`or → and → | → ^ → & → unary (not ! ~) → comparison (== === != !== > <
>= <=) → shift (<< >>) → range (..) → additive → multiplicative`.
`==` on strings is case-**insensitive**;
`===` is strict. Numbers are f64; bitwise truncates toward zero to i64
(NaN/∞/out-of-range → `ValueError`; shifts need `0..64`).
`+ - * /` and bitwise broadcast over arrays (scalar + array↔array recycle).
`..` outside `for-in` is a `TypeError`. `~[1,2]` maps elementwise.
Indexing needs non-negative integers (`arr[1.9]` → `TypeError`).

Strings interpolate: `"hi ${var.name}!"` (parsed as `+` concat).

## 8. Standard library

`con.Print(...)`, `con.Input([prompt])`, `math.Sin/Cos/Tan/Sqrt/Floor/Ceil/
Abs/Pow/Min/Max/Clamp/Random`, `str.Len/Upper/Lower/Split/Join/Contains/
Replace/Trim/Sub/From/ToNum`, `arr.Len/Push/Pop/Reverse/Sort/Join/Keys/Has/
Get/Slice`, `time.Now()` (unix ms), `file.Read/Write`. Array builtins return
new arrays (`var.xs = arr.Push(var.xs, 4)`).

File variables hold paths selected up front — no I/O happens at select
time, so missing files surface later as catchable `IOError`:

```kal
var local myfile file = selectFile("/tmp/notes.txt")
file.Write(var.myfile, "hello file")
con.Print(file.Read(var.myfile))
```

`file.Read`/`file.Write` also accept plain path strings
(`file.Read("/tmp/notes.txt")`), and a user-defined `selectFile` function
takes precedence over the builtin.

## 9. CLI

`kalvita run [file]`, `check`, `fmt [--write]`, `test [dir]` (golden
`<name>.kal` vs `<name>.expected`), `repl` (`:help :clear :quit`).
Bare `kalvita` uses `kal.toml` `main`, else `sample.kal`.
