; Highlights for the text form of a Komorebi IR module (`.kom`).
; Capture names follow the common tree-sitter set. A word the checker gives a meaning
; (an operation, a value of the language) is matched by name; a pattern for a built-in
; and the one for any other name never match the same node, so the order of patterns
; does not matter.

(comment) @comment

(string_literal) @string
(int_literal) @number
(float_literal) @number.float
[(true) (false)] @boolean
(fit) @constant.builtin

; ---- Words that declare ----------------------------------------------------------

[
  "module" "tile" "format" "requires" "param" "message" "record" "func" "extern"
  "combine" "fallback" "program" "control" "state" "handler" "run" "seed" "on"
  "chain" "call"
] @keyword

["local_state" "group" "islands" "impl" "dynamic"] @keyword
(requirement_unit) @constant.builtin
(requirement_shape) @constant.builtin

(module name: (identifier) @module)
(program_decl name: (identifier) @type)
(message_decl name: (identifier) @type)
(record_decl name: (identifier) @type)
(format_decl (identifier) @type)
(param_decl name: (identifier) @constant)

(func_decl name: (identifier) @function)
(combine_decl name: (identifier) @function)
(extern_decl name: (identifier) @function)
(handler_decl name: (identifier) @function)
(control_decl name: (identifier) @function)
(run_decl name: (identifier) @function)
(chain_decl name: (identifier) @function)
(on_block name: (identifier) @function)
(on_block binding: (identifier) @variable.parameter)
(call_statement name: (identifier) @function.call)

(parameter name: (identifier) @variable.parameter)
(field name: (identifier) @property)
(state_member name: (identifier) @property)
(init name: (identifier) @property)

; ---- Statements ------------------------------------------------------------------

["let" "var"] @keyword
["if" "else" "for" "in" "while" "return" "await"] @keyword.control
["store" "consume" "past"] @keyword

(let_statement name: (identifier) @variable)
(var_statement name: (identifier) @variable)
(for_statement variable: (identifier) @variable)

; ---- Types -----------------------------------------------------------------------

(scalar_type) @type.builtin
(view_type) @type.builtin
(bool_type) @type.builtin
(tile_type "tile" @type.builtin)
(tensor_type "tensor" @type.builtin)
(tile_type format: (identifier) @type)
(tensor_type format: (identifier) @type)
(named_type (identifier) @type)
(tile_type "<" @punctuation.bracket ">" @punctuation.bracket)
(tensor_type "<" @punctuation.bracket ">" @punctuation.bracket)

; ---- Attributes ------------------------------------------------------------------

(attribute_name) @attribute

; ---- Names and operations --------------------------------------------------------

(argument name: (identifier) @variable.parameter)

(call_expression
  function: (identifier) @function.builtin
  (#any-of? @function.builtin
    "min" "max" "range" "where" "maximum" "minimum" "matmul" "reduce_sum" "reduce_max"
    "reduce_min" "neg" "abs" "relu" "square" "floor" "exp" "exp2" "tanh" "erf" "log"
    "sigmoid" "silu" "sqrt" "rsqrt" "reciprocal" "partition" "stream" "sink" "zip"
    "reduce" "gather" "broadcast" "sync" "all_reduce" "all_gather" "send" "forward"
    "offer" "invoke" "outcome" "group" "reduce_to" "broadcast_from"
    "u8" "u16" "u32" "u64" "i8" "i16" "i32" "i64" "f16" "f32" "f64"))

(call_expression
  function: (identifier) @function.call
  (#not-any-of? @function.call
    "min" "max" "range" "where" "maximum" "minimum" "matmul" "reduce_sum" "reduce_max"
    "reduce_min" "neg" "abs" "relu" "square" "floor" "exp" "exp2" "tanh" "erf" "log"
    "sigmoid" "silu" "sqrt" "rsqrt" "reciprocal" "partition" "stream" "sink" "zip"
    "reduce" "gather" "broadcast" "sync" "all_reduce" "all_gather" "send" "forward"
    "offer" "invoke" "outcome" "group" "reduce_to" "broadcast_from"
    "u8" "u16" "u32" "u64" "i8" "i16" "i32" "i64" "f16" "f32" "f64"))

(member_expression
  . (identifier) @variable.builtin
  (#any-of? @variable.builtin "worker" "group" "island"))

(call_expression
  function: (member_expression member: (identifier) @function.method.builtin)
  (#any-of? @function.method.builtin
    "id" "tiles" "items" "words" "records" "put" "push" "fill" "zero" "bump" "split"
    "leaders" "of" "crosses" "hops" "cast" "arrive" "row" "col"))

(call_expression
  function: (member_expression member: (identifier) @function.method.call)
  (#not-any-of? @function.method.call
    "id" "tiles" "items" "words" "records" "put" "push" "fill" "zero" "bump" "split"
    "leaders" "of" "crosses" "hops" "cast" "arrive" "row" "col"))

(member_expression member: (identifier) @property)

((identifier) @constant.builtin
  (#any-of? @constant.builtin "zeros" "row" "col" "sum" "and" "or"))

(argument
  value: ((identifier) @constant.builtin
    (#any-of? @constant.builtin "max" "min")))

((identifier) @constant.builtin
  (#any-of? @constant.builtin
    "view_end" "empty_state" "send_refused" "capacity" "given_up" "dependency"
    "not_a_peer" "over_bound" "divide_by_zero" "out_of_range"))

; ---- Operators and punctuation ---------------------------------------------------

(binary_expression operator: _ @operator)
(unary_expression operator: _ @operator)
(assign_operator) @operator
["->" "<-" ">=" "="] @operator

["(" ")" "[" "]" "{" "}"] @punctuation.bracket
["," ":"] @punctuation.delimiter
"." @punctuation.delimiter
