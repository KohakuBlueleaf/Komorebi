// A tree-sitter grammar for the text form of a Komorebi IR module (a `.kom` file),
// written rule by rule from `src/komorebi-ir/syntax/komorebi.pest`: the rule names
// follow that file's. A line break separates the parts of a module and the statements
// of a block, so it is a token here (`_nl`) and not whitespace; inside parentheses,
// brackets and braces the grammar allows it where the pest file names `ogap`.
// This grammar is looser than the checker in two places, on purpose: the target of an
// assignment is any postfix expression, and a state written without its block (which
// the checker refuses by name) is not parsed.

const PREC = {
  or: 1,
  and: 2,
  compare: 3,
  bit_or: 4,
  bit_xor: 5,
  bit_and: 6,
  shift: 7,
  add: 8,
  mul: 9,
  neg: 10,
};

// One or more of `rule`, separated by `sep`.
function sep1(rule, sep) {
  return seq(rule, repeat(seq(sep, rule)));
}

module.exports = grammar({
  name: 'komorebi_ir',

  extras: $ => [/[ \t\r]/, $.comment],

  word: $ => $.identifier,

  conflicts: $ => [
    [$.if_statement],
    [$._primary_expression, $.struct_init],
  ],

  rules: {
    // module = SOI ogap kw_module ident (gap item)* ogap EOI
    module: $ => seq(
      optional($._gap),
      'module',
      field('name', $.identifier),
      repeat(seq($._gap, $._item)),
      optional($._gap),
    ),

    comment: _ => token(seq('//', /[^\n]*/)),

    // A line break (the comments and the blank lines between are extras and blanks).
    _nl: _ => /\n/,
    _gap: $ => repeat1($._nl),

    // ---- Items ---------------------------------------------------------------

    _item: $ => choice(
      $.tile_decl,
      $.format_decl,
      $.requires_decl,
      $.param_decl,
      $.message_decl,
      $.record_decl,
      $.func_decl,
      $.extern_decl,
      $.combine_decl,
      $.program_decl,
      $.control_decl,
    ),

    tile_decl: $ => seq('tile', '[', $.int_literal, ',', $.int_literal, ']'),

    format_decl: $ => seq('format', sep1($.identifier, ',')),

    requires_decl: $ => seq(
      'requires', '{',
      optional($._gap),
      optional(seq(sep1($.requirement, $._gap), optional($._gap))),
      '}',
    ),

    requirement: $ => choice(
      seq('local_state', '>=', $.requirement_unit, '(', $.int_literal, ')'),
      seq('group', '=', $.requirement_shape),
      'dynamic',
      seq('islands', '>=', $.int_literal),
      seq('impl', '(', $.string_literal, ')'),
    ),
    requirement_unit: _ => choice('tiles', 'bytes'),
    requirement_shape: _ => choice('line', 'grid'),

    param_decl: $ => seq(
      'param', field('name', $.identifier), ':', field('type', $._type), '=',
      choice($.fit, $._expression),
      repeat($.attribute),
    ),
    fit: _ => 'fit',

    message_decl: $ => seq(
      'message', field('name', $.identifier), repeat($.attribute), $.fields,
    ),
    record_decl: $ => seq('record', field('name', $.identifier), $.fields),
    fields: $ => comma_list($, '{', $.field, '}'),
    field: $ => seq(field('name', $.identifier), ':', field('type', $._type)),

    func_decl: $ => seq(
      'func', field('name', $.identifier), $.parameters, '->', field('type', $._type),
      repeat($.attribute), $.block,
    ),
    combine_decl: $ => seq(
      'combine', field('name', $.identifier), $.parameters, '->', field('type', $._type),
      repeat($.attribute), $.block,
    ),
    extern_decl: $ => seq(
      'extern', 'func', field('name', $.identifier), $.parameters, '->',
      field('type', $._type), repeat($.attribute), optional($.fallback),
    ),
    fallback: $ => seq('fallback', $.block),

    parameters: $ => comma_list($, '(', $.parameter, ')', true),
    parameter: $ => seq(
      field('name', $.identifier), ':', field('type', $._type), repeat($.attribute),
    ),

    program_decl: $ => seq(
      'program', field('name', $.identifier), '{',
      optional($._gap),
      optional(seq(sep1($._program_item, $._gap), optional($._gap))),
      '}',
    ),
    _program_item: $ => choice(
      $.message_decl, $.record_decl, $.state_decl, $.handler_decl, $.run_decl, $.chain_decl,
    ),

    state_decl: $ => seq(
      'state', repeat($.attribute), '{',
      optional($._gap),
      optional(seq(sep1($.state_member, $._gap), optional($._gap))),
      '}',
    ),
    state_member: $ => seq(
      field('name', $.identifier), ':', field('type', $._type), repeat($.attribute),
    ),

    handler_decl: $ => seq(
      'handler', field('name', $.identifier), $.parameters,
      optional(seq('->', field('type', $.scalar_type))),
      repeat($.attribute), $.block,
    ),

    run_decl: $ => seq(
      'run', field('name', $.identifier), $.parameters, repeat($.attribute), '{',
      optional($._gap),
      $.seed_block,
      repeat(seq($._gap, $.on_block)),
      optional($._gap),
      '}',
    ),
    seed_block: $ => seq('seed', $.block),
    on_block: $ => seq(
      'on', field('name', $.identifier), '(', field('binding', $.identifier), ')', $.block,
    ),

    chain_decl: $ => seq(
      'chain', field('name', $.identifier), $.parameters, repeat($.attribute), '{',
      optional($._gap),
      optional(seq(sep1($.call_statement, $._gap), optional($._gap))),
      '}',
    ),
    call_statement: $ => seq('call', field('name', $.identifier), $.arguments),

    control_decl: $ => seq(
      'control', field('name', $.identifier), $.parameters,
      optional(seq('->', field('type', $.scalar_type))),
      repeat($.attribute), $.block,
    ),

    // ---- Attributes ----------------------------------------------------------

    attribute: $ => seq($.attribute_name, optional($.arguments)),
    attribute_name: _ => /@[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*/,

    // ---- Types ---------------------------------------------------------------

    _type: $ => choice(
      $.tile_type,
      $.tensor_type,
      $.view_type,
      $.bool_type,
      $.scalar_array_type,
      $.named_type,
    ),
    tile_type: $ => seq(
      'tile', '<', field('format', $.identifier), '>', optional(choice($.shape, $.array)),
    ),
    tensor_type: $ => seq('tensor', '<', field('format', $.identifier), '>'),
    view_type: _ => 'view',
    bool_type: _ => 'bool',
    scalar_array_type: $ => seq($.scalar_type, optional($.array)),
    scalar_type: _ => choice(
      'u8', 'u16', 'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'f16', 'f32', 'f64',
    ),
    named_type: $ => $.identifier,
    array: $ => seq('[', $._expression, ']'),
    // a block of r rows of c tiles
    shape: $ => seq('[', $._expression, ',', $._expression, ']'),

    // ---- Statements ----------------------------------------------------------

    block: $ => seq(
      '{',
      optional($._gap),
      optional(seq(sep1($._statement, $._gap), optional($._gap))),
      '}',
    ),

    _statement: $ => choice(
      $.let_statement,
      $.var_statement,
      $.for_statement,
      $.while_statement,
      $.if_statement,
      $.store_statement,
      $.consume_statement,
      $.return_statement,
      $.assign_statement,
      $.expression_statement,
    ),

    let_statement: $ => seq(
      'let', field('name', $.identifier), optional(seq(':', field('type', $._type))),
      '=', field('value', $._expression), repeat($.attribute),
    ),
    var_statement: $ => seq(
      'var', field('name', $.identifier), ':', field('type', $._type),
      '=', field('value', $._expression),
    ),
    for_statement: $ => seq(
      'for', sep1(field('variable', $.identifier), ','), 'in',
      field('iterable', $._expression), repeat($.attribute), $.block,
    ),
    while_statement: $ => seq('while', field('condition', $._expression), $.block),
    if_statement: $ => seq(
      'if', field('condition', $._expression), $.block,
      optional(seq(optional($._gap), 'else', $.block)),
    ),
    store_statement: $ => seq('store', field('view', $._expression), '<-', field('value', $._expression)),
    consume_statement: $ => seq('consume', field('message', $.identifier)),
    return_statement: $ => seq('return', optional($._expression)),
    assign_statement: $ => seq(
      field('target', $._postfix_expression), $.assign_operator, field('value', $._expression),
    ),
    assign_operator: _ => choice('+=', '-=', '='),
    expression_statement: $ => $._expression,

    // ---- Expressions ---------------------------------------------------------

    _expression: $ => choice($.binary_expression, $.unary_expression, $._postfix_expression),

    binary_expression: $ => {
      const table = [
        [PREC.or, '||'],
        [PREC.and, '&&'],
        [PREC.compare, choice('==', '!=', '<=', '>=', '<', '>')],
        [PREC.bit_or, '|'],
        [PREC.bit_xor, '^'],
        [PREC.bit_and, '&'],
        [PREC.shift, choice('<<', '>>')],
        [PREC.add, choice('+', '-')],
        [PREC.mul, choice('*', '/', '%')],
      ];
      return choice(...table.map(([p, op]) => prec.left(p, seq(
        field('left', $._expression), field('operator', op), field('right', $._expression),
      ))));
    },
    unary_expression: $ => prec(PREC.neg, seq(field('operator', '-'), field('operand', $._expression))),

    _postfix_expression: $ => choice(
      $._primary_expression,
      $.member_expression,
      $.call_expression,
      $.index_expression,
    ),
    member_expression: $ => seq($._postfix_expression, '.', field('member', $.identifier)),
    call_expression: $ => seq(field('function', $._postfix_expression), $.arguments),
    index_expression: $ => seq($._postfix_expression, $.index),

    index: $ => comma_list($, '[', $._index_item, ']'),
    _index_item: $ => choice($.slice, $._expression),
    slice: $ => seq($._expression, ':', $._expression, optional(seq(':', $._expression))),

    arguments: $ => comma_list($, '(', $.argument, ')', true),
    argument: $ => seq(optional(seq(field('name', $.identifier), '=')), field('value', $._expression)),

    _primary_expression: $ => choice(
      $.parenthesized_expression,
      $.list,
      $.anonymous_init,
      $.struct_init,
      $.float_literal,
      $.int_literal,
      $.string_literal,
      $.true,
      $.false,
      $.await_expression,
      $.past_expression,
      $.identifier,
    ),

    // `(e)` is a parenthesis and `(e, f)` a tuple.
    parenthesized_expression: $ => comma_list($, '(', $._expression, ')'),
    list: $ => comma_list($, '[', $._expression, ']', true),
    anonymous_init: $ => $.inits,
    struct_init: $ => seq(field('type', $.identifier), $.inits),
    inits: $ => comma_list($, '{', $.init, '}'),
    init: $ => seq(field('name', $.identifier), ':', field('value', $._expression)),
    await_expression: $ => seq('await', field('handle', $.identifier)),
    past_expression: $ => seq('past', '(', field('name', $.identifier), ')'),

    // ---- Literals ------------------------------------------------------------

    float_literal: _ => token(/[0-9]+(\.[0-9]+([eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)/),
    int_literal: _ => token(/0x[0-9A-Fa-f]+|[0-9]+/),
    string_literal: _ => token(/"([^"\\\n]|\\.)*"/),
    true: _ => 'true',
    false: _ => 'false',

    identifier: _ => /[A-Za-z_][A-Za-z0-9_]*/,
  },
});

// `open item (, item)* ,? close` with a line break allowed beside every comma and every
// delimiter (the pest file's `ogap`); `maybe_empty` lets the list be empty.
function comma_list($, open, item, close, maybe_empty = false) {
  const gap = optional($._gap);
  const body = seq(
    item,
    repeat(seq(gap, ',', gap, item)),
    optional(seq(gap, ',')),
  );
  return maybe_empty
    ? seq(open, gap, optional(seq(body, gap)), close)
    : seq(open, gap, body, gap, close);
}
