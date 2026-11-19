//! The stack the depth limit needs. Every entry point of the crate runs its work
//! through `on_deep_stack`, so a tree as deep as the parser accepts is parsed,
//! written, cloned, compared and dropped there safely, whatever thread called.

use std::cell::Cell;

/// The stack of the thread the work runs on: four times what the deepest accepted
/// text needs to be parsed and written in a debug build.
pub const STACK_BYTES: usize = 64 << 20;

thread_local! {
    /// Whether this thread is one `on_deep_stack` made.
    static DEEP: Cell<bool> = const { Cell::new(false) };
}

/// Runs `work` on a thread with a stack of `STACK_BYTES` and gives its result. On a
/// thread this function made, or when no thread can be made, it runs `work` in
/// place. A panic inside `work` goes on in the caller.
pub fn on_deep_stack<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    if DEEP.with(Cell::get) {
        return work();
    }
    let mut work = Some(work);
    let mut slot = None;
    let joined = std::thread::scope(|scope| {
        let thread = std::thread::Builder::new().stack_size(STACK_BYTES);
        let spawned = thread.spawn_scoped(scope, || {
            DEEP.with(|deep| deep.set(true));
            if let Some(work) = work.take() {
                slot = Some(work());
            }
        });
        spawned.ok().map(|handle| handle.join())
    });
    if let Some(Err(panic)) = joined {
        std::panic::resume_unwind(panic);
    }
    match (slot, work) {
        (Some(result), _) => result,
        (None, Some(work)) => work(),
        (None, None) => unreachable!("the work ran and gave its result"),
    }
}
