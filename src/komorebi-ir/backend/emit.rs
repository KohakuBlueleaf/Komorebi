//! The emitter's shared walk: a body as events in program order, each instruction one
//! event, with the depth of the constructs it is inside. Every backend prints its own
//! text from these (the Tensix backend must give today's text exactly, so there is no
//! shared tree of the output); what is shared is the walk and its test.
//!
//! A construct opens with its instruction (`If`, a loop, `While`) and closes with its
//! `End`; `Else` turns an `If` to its second arm; `WhileTest` is where a `while` tests
//! its condition, after the instructions that compute it.

use crate::program::body::{Body, Inst, InstId, Op};

/// One instruction of a body, as the walk gives it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event<'b> {
    /// An instruction that opens a construct: `If`, `ForRange`, `ForStream`, `ForZip`,
    /// `ForArrive`, `While`.
    Open(InstId, &'b Inst),
    /// `Else`: the `If` open at this depth goes on with its second arm.
    Else(InstId),
    /// `WhileTest`: the `While` open at this depth tests its condition here.
    Test(InstId, &'b Inst),
    /// `End`: the construct open at this depth closes.
    Close(InstId),
    /// Any other instruction.
    Plain(InstId, &'b Inst),
}

/// An event and its depth: the constructs it is inside (an `Open` and its `Close` have
/// the depth of the construct's outside; `Else` and `Test` that of its inside).
pub type Step<'b> = (usize, Event<'b>);

/// Which construct is open.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    If { seen_else: bool },
    Loop,
    While { tested: bool },
}

/// The events of a body, in program order; or, for a body whose constructs do not
/// balance (which `program::validate` refuses), what is wrong and where.
pub fn walk(body: &Body) -> Result<Vec<Step<'_>>, String> {
    let mut out = Vec::with_capacity(body.insts.len());
    let mut open: Vec<Open> = Vec::new();
    for (i, inst) in body.insts.iter().enumerate() {
        let id = InstId(i as u32);
        let depth = open.len();
        match &inst.op {
            Op::If { .. } => {
                out.push((depth, Event::Open(id, inst)));
                open.push(Open::If { seen_else: false });
            }
            Op::ForRange { .. }
            | Op::ForStream { .. }
            | Op::ForZip { .. }
            | Op::ForArrive { .. } => {
                out.push((depth, Event::Open(id, inst)));
                open.push(Open::Loop);
            }
            Op::While => {
                out.push((depth, Event::Open(id, inst)));
                open.push(Open::While { tested: false });
            }
            Op::Else => match open.last_mut() {
                Some(Open::If { seen_else }) if !*seen_else => {
                    *seen_else = true;
                    out.push((depth, Event::Else(id)));
                }
                _ => return Err(format!("i{i}: `Else` outside an `If` or after its `Else`")),
            },
            Op::WhileTest { .. } => match open.last_mut() {
                Some(Open::While { tested }) if !*tested => {
                    *tested = true;
                    out.push((depth, Event::Test(id, inst)));
                }
                _ => return Err(format!("i{i}: `WhileTest` outside a `While`")),
            },
            Op::End => match open.pop() {
                Some(Open::While { tested: false }) => {
                    return Err(format!("i{i}: a `While` closed without its `WhileTest`"))
                }
                Some(_) => out.push((depth - 1, Event::Close(id))),
                None => return Err(format!("i{i}: `End` with nothing open")),
            },
            _ => out.push((depth, Event::Plain(id, inst))),
        }
    }
    if open.is_empty() {
        Ok(out)
    } else {
        Err(format!(
            "{} constructs left open at the body's end",
            open.len()
        ))
    }
}
