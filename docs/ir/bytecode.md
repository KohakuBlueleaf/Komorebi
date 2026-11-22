# The IR's byte code (being built)

!!! note "Being built"
    The Komorebi IR and its command `komo` are being built. This page gives the layout of a byte-code
    program as the build writes it today; it changes with each version.

A byte-code program (`.komc`) is the stored form of a module's program structure: what program
compile gives after its checks. `komo compile x.kom -o x.komc` writes one; `komo print x.komc` prints
the structure it holds, and `komo hashes x.komc` its three hashes. A CPU interprets it as it is: the
reader gives back the program structure the interpreter runs, unchanged.

## The file

A header, then three parts, in this order:

| Bytes | Holds |
|---|---|
| 4 | The magic `KOMC` |
| 4 | The version, a u32: `6` |
| 8, 8, 8 | The length in bytes of each part, S, H and P, each a u64 |
| S | The program: everything but the notes |
| H | The hints |
| P | The source positions, the names the source gives values, slots and parameters, and the origin of a frontend's module |

A reader can hash a part, or skip the positions, without reading the others.

Version 3 gave a block, a stream, a sink and a reduction's values the rows and the columns of their
tiles (a block is a matrix of tiles): each writes its rows before its columns. Version 4 gives
`ForZip` its streams as a list (a count, then the values), so a `zip` walks two streams or more.
Version 5 gives each call of a chain its part of the chain's group (R47) after its arguments:
`Whole`, or `Part` and its two ends, each a `Count`. Version 6 gives a message its body: each
message type writes the bytes of its body (a u32, `0` for none) after its fields; `Send` and `Offer`
write where their body comes from as an option of `ScalarsFrom` (`Value` and a value, or `Member`
and a state member), `Send` after its rank; `Forward` writes the type it goes on as (`as`) as an
option of a message id after its message; the operation `Body` and the type `Body` are new. The operation `AllGather`
(number 89) is new too: its operand, then its `publish` list; and the operation `StoreScalars`
(number 90): its view, then where its scalars come from as a `ScalarsFrom`. Version 6 also gives a
module several tile shapes: the module writes the list of its shapes (a count, then each shape's rows
and columns, the default first) where it wrote its one shape, and each function, state member,
handler and run writes, last, its shape's place in that list (a u32). A reader refuses a version 5 program
with `K952`, naming both versions.

## How values are written

Every integer is fixed width and little-endian, so one module has one byte string.

| Kind | Written as |
|---|---|
| An id (`FuncId`, `ValueId`, ...), a position, an operand index | u32 |
| A count of a list, a length of a string | u32, then the items |
| A string | its length in bytes (u32), then UTF-8 |
| A number the module states (`Count::Const`, a grid's shape, a tile's shape, an integer constant's bits, an amount of `requires`) | u64 |
| A float constant | the bits of its `f64`, as a u64 |
| `bool` | one byte, `0` or `1` |
| An option | one byte, `0` for none, or `1` and the value |
| An enum | its number (one byte, from the tables below), then its fields in the order declared |
| A struct | its fields, in the order the walk below gives |

Names are written where they stand; there is no table of names, so that S does not depend on P.

## The walk

Each part is one walk of the program structure, each list in its one order:

- **S**: the module's name; the tile shapes (a count, then each shape's rows and cols, the default
  first); the formats; the parameters (name, scalar
  type, binding); the records and the messages (name, the program it is declared in as an option,
  the fields as name and scalar type; a message then the bytes of its body); the functions; the
  programs; the control programs; the requirements.
    - A function: name, kind (`Plain`; `Combine` and whether commutative; `Extern` with its
      implementation name, the positions it reads, the positions it writes), the parameter types,
      the result type, the body as an option, the effects, the tile shape (its place in the
      module's list).
    - A program: name, the state members, the handlers, the runs, the chains.
    - A state member: name, kind, count, scope, lifetime, `@export`, `@filled(element)`, the tile
      shape of its `state` block.
    - A handler: name, parameter types, the result as an option, `@on`, `@exclusive`, body,
      effects, the tile shape.
    - A run: name, parameter types, `@peers`, the seed, the `on M` bodies (message id, body),
      effects, the tile shape.
    - A chain: name, parameter types, `@on`, the calls (handler id, arguments, the part of the
      group as `Part`), the dependency edges (writer, reader, parameter, write, read, `past`).
    - A control program: name, parameter types, the result as an option, body.
    - A requirement: the ability, then its causes.
    - A body: the type of each value, the type of each slot, the instructions (the operation, then
      the values it defines). An operation is its number, then its operands in the order the
      variant declares them.
    - Effects: the tensor accesses (parameter, body, view, access, instruction), the state uses
      (member, body, use, instruction), the collectives (body, instruction), the functions called.
- **H**: the hints, each its anchor, its dotted name, its arguments.
- **P**: the positions (anchor, first byte, end byte); the names of values and slots; the names of
  parameters (anchor, name); the origin, an option: the source files (strings), the lines a place
  covers (file, line number, the line as written), and the places (anchor, file, first line, last
  line). A module read from `.kom` text has no origin; a frontend's module (today's Python classes)
  records where each handler and statement was written, for a backend that prints it.

## The hashes

Each is SHA-256 over a tag, the version (u32, little-endian) and its parts:

| Hash | Tag | Parts | What it means |
|---|---|---|---|
| `semantic` | `komorebi-ir semantic` and a zero byte | S | The same program |
| `build` | `komorebi-ir build` and a zero byte | S, H | The program and what may change how it is built: the key of the cache of a backend that reads S and H only (the CPU backend) |
| `full` | `komorebi-ir full` and a zero byte | S, H, P | This exact file: the key of the cache of a backend that also reads the positions and names |

A comment added changes only `full`; a hint changed changes `build` and `full`; a declaration
moved changes all three. A hash is printed as 64 lowercase hex digits.

## What the reader refuses

The reader trusts nothing: it refuses any byte the writer would not have written, then validates
the module it read. No input makes it panic.

| Code | Refused |
|---|---|
| `K950` | Not a byte-code program: shorter than the magic and the version, or another magic |
| `K951` | A newer version than the build reads, by name: both versions are given |
| `K952` | An older version, by name: compile it again from its `.kom` |
| `K953` | Bytes that are not a byte-code program of this version: the parts' lengths do not sum to the file; a part cut short, or whose walk does not end at its length; a number that is no variant of its enum; a `bool` or an option byte other than `0` and `1`; a string that is not UTF-8; a count above the bytes left; a type or a hint nested deeper than 256; an integer constant with bits above its type's width; a float constant that is a NaN |
| `K954` | A module that reads whole and does not validate, its notes included: an origin's place on an anchor that names nothing, or on a file past its files, on line 0, from a line after its last, on a line its lines do not hold; places not in the order of their anchors or one twice; lines not in (file, number) order or one twice |
| `K955` | A module whose derived parts (a body's effects, a chain's dependencies, the requirement set) are not the ones Derive gives it again |

Loading a byte-code program to run it goes further: the module is accepted only where its text would
check. After `K955`, Verify's rules run on it, and a module that breaks one is refused with that
rule's code (`K205`, `K701`, ...), giving the source bytes it is about.

A refusal gives the byte it is about: `x.komc: byte 32: K953: ...`.

## The numbers

Every enum the format writes, each variant with its number. A number is never reused or moved: a
new variant takes a new number, a removed one leaves its number unused, and either change is a new
version.

### `BodyRef`

| Number | Variant |
|---|---|
| 0 | `Func` |
| 1 | `Handler` |
| 2 | `Seed` |
| 3 | `On` |
| 4 | `Control` |

### `FuncKind`

| Number | Variant |
|---|---|
| 0 | `Plain` |
| 1 | `Combine` |
| 2 | `Extern` |

### `MemberKind`

| Number | Variant |
|---|---|
| 0 | `Tiles` |
| 1 | `Scalars` |

### `Scope`

| Number | Variant |
|---|---|
| 0 | `Worker` |

### `Lifetime`

| Number | Variant |
|---|---|
| 0 | `Program` |

### `CallArg`

| Number | Variant |
|---|---|
| 0 | `Param` |
| 1 | `Past` |
| 2 | `Const` |

### `Op`

| Number | Variant |
|---|---|
| 0 | `Const` |
| 1 | `Arg` |
| 2 | `Param` |
| 3 | `Get` |
| 4 | `Set` |
| 5 | `Binary` |
| 6 | `Compare` |
| 7 | `LogicAnd` |
| 8 | `LogicOr` |
| 9 | `If` |
| 10 | `Else` |
| 11 | `ForRange` |
| 12 | `ForStream` |
| 13 | `ForZip` |
| 14 | `ForArrive` |
| 15 | `While` |
| 16 | `WhileTest` |
| 17 | `End` |
| 18 | `Return` |
| 19 | `Convert` |
| 20 | `Unary` |
| 21 | `Select` |
| 22 | `Reduce` |
| 23 | `SpreadCol` |
| 24 | `SpreadRow` |
| 25 | `Matmul` |
| 26 | `Transpose` |
| 27 | `Zeros` |
| 28 | `CallFunc` |
| 29 | `Tiles` |
| 30 | `Rect` |
| 31 | `Turn` |
| 32 | `Items` |
| 33 | `ReadTile` |
| 34 | `Words` |
| 35 | `Word` |
| 36 | `Records` |
| 37 | `RecordAt` |
| 38 | `RecordField` |
| 39 | `Put` |
| 40 | `Partition` |
| 41 | `Stream` |
| 42 | `Sink` |
| 43 | `Push` |
| 44 | `Store` |
| 45 | `BlockAt` |
| 46 | `Fill` |
| 47 | `ZeroState` |
| 48 | `StateSet` |
| 49 | `StateGet` |
| 50 | `Bump` |
| 51 | `Rank` |
| 52 | `IsLeader` |
| 53 | `WorkerId` |
| 54 | `Row` |
| 55 | `Col` |
| 56 | `GroupSize` |
| 57 | `GroupId` |
| 58 | `Split` |
| 59 | `Leaders` |
| 60 | `ReduceGroup` |
| 61 | `AllReduce` |
| 62 | `Gather` |
| 63 | `Broadcast` |
| 64 | `IsSender` |
| 65 | `Cast` |
| 66 | `Sync` |
| 67 | `Arrived` |
| 68 | `MakeMessage` |
| 69 | `Send` |
| 70 | `Forward` |
| 71 | `Consume` |
| 72 | `Offer` |
| 73 | `Field` |
| 74 | `MakeGroup` |
| 75 | `Invoke` |
| 76 | `Await` |
| 77 | `Outcome` |
| 78 | `Failed` |
| 79 | `Reason` |
| 80 | `OutcomeValue` |
| 81 | `FailureName` |
| 82 | `IslandCount` |
| 83 | `IslandOf` |
| 84 | `IslandCrosses` |
| 85 | `IslandHops` |
| 86 | `ReduceTo` |
| 87 | `BroadcastFrom` |
| 88 | `Body` |
| 89 | `AllGather` |
| 90 | `StoreScalars` |

### `Access`

| Number | Variant |
|---|---|
| 0 | `Stream` |
| 1 | `Read` |
| 2 | `Sink` |
| 3 | `Store` |
| 4 | `Put` |
| 5 | `ExternRead` |
| 6 | `ExternWrite` |

### `StateUse`

| Number | Variant |
|---|---|
| 0 | `Read` |
| 1 | `WriteElement` |
| 2 | `FillWhole` |

### `Cause`

| Number | Variant |
|---|---|
| 0 | `Stated` |
| 1 | `GridHandler` |
| 2 | `DynamicRun` |
| 3 | `ExternWithoutFallback` |

### `Anchor`

| Number | Variant |
|---|---|
| 0 | `Module` |
| 1 | `Func` |
| 2 | `Program` |
| 3 | `Handler` |
| 4 | `Run` |
| 5 | `Member` |
| 6 | `Control` |
| 7 | `Chain` |
| 8 | `Call` |
| 9 | `Param` |
| 10 | `Message` |
| 11 | `Record` |
| 12 | `ParamOf` |
| 13 | `Inst` |

### `Owner`

| Number | Variant |
|---|---|
| 0 | `Func` |
| 1 | `Handler` |
| 2 | `Run` |
| 3 | `Chain` |
| 4 | `Control` |

### `HintArg`

| Number | Variant |
|---|---|
| 0 | `Number` |
| 1 | `Text` |
| 2 | `Name` |
| 3 | `Named` |
| 4 | `List` |

### `Named`

| Number | Variant |
|---|---|
| 0 | `Value` |
| 1 | `Local` |

### `Type`

| Number | Variant |
|---|---|
| 0 | `Scalar` |
| 1 | `Bool` |
| 2 | `Tile` |
| 3 | `Reduced` |
| 4 | `Block` |
| 5 | `Tensor` |
| 6 | `View` |
| 7 | `Items` |
| 8 | `Stream` |
| 9 | `Zip` |
| 10 | `Sink` |
| 11 | `Words` |
| 12 | `Records` |
| 13 | `RecordValue` |
| 14 | `Message` |
| 15 | `SubGroups` |
| 16 | `Leaders` |
| 17 | `Broadcast` |
| 18 | `Group` |
| 19 | `Handle` |
| 20 | `Outcome` |
| 21 | `Reason` |
| 22 | `Body` |

### `ConstValue`

| Number | Variant |
|---|---|
| 0 | `Int` |
| 1 | `Float` |
| 2 | `Bool` |

### `Binding`

| Number | Variant |
|---|---|
| 0 | `Build` |
| 1 | `Load` |
| 2 | `Invoke` |

### `On`

| Number | Variant |
|---|---|
| 0 | `Group` |
| 1 | `Grid` |

### `Part`

| Number | Variant |
|---|---|
| 0 | `Whole` |
| 1 | `Part` |

### `ScalarsFrom`

| Number | Variant |
|---|---|
| 0 | `Value` |
| 1 | `Member` |

### `Peers`

| Number | Variant |
|---|---|
| 0 | `None` |
| 1 | `Offsets` |
| 2 | `Pairs` |

### `Ability`

| Number | Variant |
|---|---|
| 0 | `LocalState` |
| 1 | `Grid` |
| 2 | `Dynamic` |
| 3 | `Islands` |
| 4 | `Impl` |

### `Count`

| Number | Variant |
|---|---|
| 0 | `Const` |
| 1 | `Param` |
| 2 | `GroupSize` |

### `Axis`

| Number | Variant |
|---|---|
| 0 | `Row` |
| 1 | `Col` |

### `Format`

| Number | Variant |
|---|---|
| 0 | `U8` |
| 1 | `U16` |
| 2 | `U32` |
| 3 | `U64` |
| 4 | `I8` |
| 5 | `I16` |
| 6 | `I32` |
| 7 | `I64` |
| 8 | `F16` |
| 9 | `F32` |
| 10 | `F64` |
| 11 | `Bf16` |
| 12 | `Bfp8` |
| 13 | `Bfp4` |
| 14 | `Bfp2` |

### `ScalarType`

| Number | Variant |
|---|---|
| 0 | `U8` |
| 1 | `U16` |
| 2 | `U32` |
| 3 | `U64` |
| 4 | `I8` |
| 5 | `I16` |
| 6 | `I32` |
| 7 | `I64` |
| 8 | `F16` |
| 9 | `F32` |
| 10 | `F64` |

### `BinOp`

| Number | Variant |
|---|---|
| 0 | `Add` |
| 1 | `Sub` |
| 2 | `Mul` |
| 3 | `Div` |
| 4 | `Rem` |
| 5 | `And` |
| 6 | `Or` |
| 7 | `Xor` |
| 8 | `Shl` |
| 9 | `Shr` |
| 10 | `Min` |
| 11 | `Max` |

### `CmpOp`

| Number | Variant |
|---|---|
| 0 | `Gt` |
| 1 | `Lt` |
| 2 | `Ge` |
| 3 | `Le` |
| 4 | `Eq` |
| 5 | `Ne` |

### `UnaryOp`

| Number | Variant |
|---|---|
| 0 | `Neg` |
| 1 | `Abs` |
| 2 | `Relu` |
| 3 | `Square` |
| 4 | `Floor` |
| 5 | `Exp` |
| 6 | `Exp2` |
| 7 | `Tanh` |
| 8 | `Erf` |
| 9 | `Log` |
| 10 | `Sigmoid` |
| 11 | `Silu` |
| 12 | `Sqrt` |
| 13 | `Rsqrt` |
| 14 | `Reciprocal` |

### `ReduceOp`

| Number | Variant |
|---|---|
| 0 | `Sum` |
| 1 | `Max` |
| 2 | `Min` |

### `SplitBy`

| Number | Variant |
|---|---|
| 0 | `Parts` |
| 1 | `Size` |

### `Combiner`

| Number | Variant |
|---|---|
| 0 | `Func` |
| 1 | `Sum` |
| 2 | `Max` |
| 3 | `Min` |
| 4 | `And` |
| 5 | `Or` |

### `Failure`

| Number | Variant |
|---|---|
| 0 | `ViewEnd` |
| 1 | `EmptyState` |
| 2 | `SendRefused` |
| 3 | `Capacity` |
| 4 | `GivenUp` |
| 5 | `Dependency` |
| 6 | `NotAPeer` |
| 7 | `OverBound` |
| 8 | `DivideByZero` |
| 9 | `OutOfRange` |

### `Entry`

| Number | Variant |
|---|---|
| 0 | `Handler` |
| 1 | `Run` |
| 2 | `Chain` |
