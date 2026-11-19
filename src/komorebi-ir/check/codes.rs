//! The diagnostic codes of the checks program compile makes, one a rule. The page
//! `docs/ir/checks.md` gives each with the sentence that states it.

/// An undotted attribute that is not one of the language's own.
pub const UNKNOWN_ATTRIBUTE: &str = "K003";
/// `while` outside a control body.
pub const WHILE_OUTSIDE_CONTROL: &str = "K005";
/// A known attribute where it has no meaning.
pub const ATTRIBUTE_MISPLACED: &str = "K006";
/// A declaration without the attribute every example gives it.
pub const ATTRIBUTE_MISSING: &str = "K007";
/// `= fit` on a parameter not bound at load.
pub const FIT_NOT_AT_LOAD: &str = "K008";
/// An ability the content uses and `requires` leaves out.
pub const REQUIREMENT_MISSING: &str = "K009";
/// Named in the language, not built yet.
pub const NOT_BUILT: &str = "K010";
/// A name that is not declared.
pub const UNKNOWN_NAME: &str = "K101";
/// An operand of a type its operation does not take.
pub const WRONG_TYPE: &str = "K102";
/// An operand the operation takes is not given.
pub const MISSING_OPERAND: &str = "K103";
/// A `let` (or anything but a `var` or an element of state) assigned.
pub const ASSIGN_LET: &str = "K104";
/// `worker.row` or `worker.col` outside a handler on a grid.
pub const ROW_OUTSIDE_GRID: &str = "K105";
/// A format used and not declared in the module's `format` line.
pub const FORMAT_NOT_DECLARED: &str = "K106";
/// A tensor indexed directly, not through a view.
pub const TENSOR_INDEXED: &str = "K107";
/// A message field that is not a scalar.
pub const MESSAGE_FIELD: &str = "K108";
/// `bool` as the type of state, a field, a parameter or a result.
pub const BOOL_PLACE: &str = "K109";
/// A format name outside the language's set.
pub const UNKNOWN_FORMAT: &str = "K110";
/// An operand in another form than its own (by position or by name).
pub const OPERAND_FORM: &str = "K111";
/// One name declared twice.
pub const NAME_TWICE: &str = "K112";
/// A message made without one of its fields.
pub const MESSAGE_FIELD_MISSING: &str = "K113";
/// An integer `/` or `%` by a zero the text shows.
pub const DIVIDE_BY_ZERO: &str = "K114";
/// An index or a count that the text shows does not fit what it addresses.
pub const OUT_OF_RANGE: &str = "K115";
/// A reduction's values used where a tile is asked, or mixed with a tile.
pub const REDUCED_VALUE: &str = "K116";
/// An integer tile with a float scalar or a float tile.
pub const INTEGER_WITH_FLOAT: &str = "K117";
/// A tile, a tensor or a view in a module with no `tile` line.
pub const TILE_NOT_DECLARED: &str = "K118";
/// Two scalars of different types in one operation or comparison.
pub const MIXED_SCALARS: &str = "K119";
/// A `func` that takes a view.
pub const FUNC_TAKES_VIEW: &str = "K201";
/// A function body that uses more than the compute family.
pub const FUNC_NOT_COMPUTE: &str = "K202";
/// A block of several tiles given to a function on one tile.
pub const BLOCK_TO_TILE: &str = "K203";
/// An `extern func` that takes a view and states neither `@reads` nor `@writes`.
pub const EXTERN_VIEW_UNSTATED: &str = "K204";
/// A `fallback` body that touches what `@reads` and `@writes` do not state.
pub const FALLBACK_TOUCHES: &str = "K205";
/// A stream that writes a view overlapping the one it reads, other than exactly.
pub const STREAM_OVERLAP: &str = "K206";
/// `.words()` or `.records()` on a view shown to hold more than one tile.
pub const WORDS_MANY_TILES: &str = "K207";
/// `items`, `records` or `words` on a tensor whose format has no stated bytes.
pub const NO_STATED_BYTES: &str = "K208";
/// The block of a `stream` or a `sink` that is neither a number written in the text
/// nor a module parameter.
pub const STREAM_BLOCK: &str = "K209";
/// A state member that some body reads and no body of the program fills.
pub const EMPTY_STATE: &str = "K301";
/// A reduce whose `with` is not a `combine` or a provided one.
pub const REDUCE_WITH: &str = "K401";
/// A broadcast in a handler that is not on a grid.
pub const BROADCAST_ON_LINE: &str = "K402";
/// Two casts on one broadcast that send tiles of different formats.
pub const BROADCAST_FORMATS: &str = "K403";
/// An arrived message not forwarded or consumed exactly once.
pub const MESSAGE_ONCE: &str = "K501";
/// An `offer` to a rank that is not a declared peer.
pub const OFFER_NOT_PEER: &str = "K502";
/// `send`, `forward`, `consume` or `offer` outside a dynamic run.
pub const SEND_OUTSIDE_RUN: &str = "K503";
/// A `past` the build cannot prove (waits for the derived effects).
pub const PAST_UNPROVEN: &str = "K601";
/// `past` outside a chain's call.
pub const PAST_OUTSIDE_CHAIN: &str = "K602";
/// An operand of a chain's call other than a chain parameter, `past(y)` or a constant.
pub const CHAIN_OPERAND: &str = "K603";
/// A race that program compile can show (waits for the derived effects).
pub const RACE: &str = "K701";
/// A control body that uses more than values, control flow, `invoke`, `await` and
/// results.
pub const CONTROL_NOT_VALUES: &str = "K702";
/// `invoke`, `await` or `outcome` outside a control program.
pub const INVOKE_OUTSIDE_CONTROL: &str = "K703";
/// A failure's name outside the ten the language names.
pub const UNKNOWN_REASON: &str = "K704";
/// `reduce_to` or `broadcast_from` outside a control program.
pub const ISLANDS_IN_HANDLER: &str = "K801";
