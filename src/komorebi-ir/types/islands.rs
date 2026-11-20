//! A machine's islands as a graph, as a run and a backend's load both take them: which
//! workers each island holds and the connections between islands (any connected set:
//! nothing names a shape or a count), and why a graph is refused (`K966`).

use std::collections::VecDeque;
use std::fmt;

/// The islands of a machine, as a graph: the islands, which workers each holds, and
/// the connections between them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IslandGraph {
    /// How many workers each island holds, in order: island 0 holds the first
    /// `sizes[0]` workers, island 1 the next `sizes[1]`, and so on.
    pub sizes: Vec<usize>,
    /// The connections between islands, by number, each both ways. None: every pair
    /// of islands is connected (a host that does not care about distance). An empty
    /// list with more than one island is not connected.
    pub links: Option<Vec<(usize, usize)>>,
}

/// The code of a machine given at load that is not a machine (its islands not a graph
/// of its workers), the same from a run's load and a backend's.
pub const NOT_A_MACHINE: &str = "K966";

/// Why a machine's islands are refused at load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IslandRefusal {
    /// The sizes add up to another number than the loaded workers.
    NotTheWorkers {
        /// The sum of the sizes.
        held: usize,
        /// The loaded workers.
        workers: usize,
    },
    /// An island that holds no worker.
    Empty(usize),
    /// A connection that names no island.
    NoSuchIsland(usize),
    /// Some two islands have no path between them.
    NotConnected,
}

impl fmt::Display for IslandRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IslandRefusal::NotTheWorkers { held, workers } => {
                write!(
                    f,
                    "the islands hold {held} workers, not the {workers} loaded"
                )
            }
            IslandRefusal::Empty(i) => write!(f, "island {i} holds no worker"),
            IslandRefusal::NoSuchIsland(i) => {
                write!(f, "a connection to island {i}, which is not one")
            }
            IslandRefusal::NotConnected => write!(f, "the islands are not connected"),
        }
    }
}

impl IslandRefusal {
    /// The refusal's sentence, as every load gives it under `NOT_A_MACHINE`.
    pub fn sentence(&self) -> String {
        format!("the machine given at load is not a machine: {self}")
    }
}

impl IslandGraph {
    /// One island holding `workers` workers.
    pub fn one(workers: usize) -> IslandGraph {
        IslandGraph {
            sizes: vec![workers],
            links: None,
        }
    }

    /// How many islands.
    pub fn count(&self) -> usize {
        self.sizes.len()
    }

    /// The hops of a shortest path between every two islands, by a breadth-first walk
    /// from each, of a graph that holds `workers` workers; or why it is refused.
    pub fn hops(&self, workers: usize) -> Result<Vec<Vec<u64>>, IslandRefusal> {
        let held: usize = self.sizes.iter().sum();
        if held != workers {
            return Err(IslandRefusal::NotTheWorkers { held, workers });
        }
        if let Some(i) = self.sizes.iter().position(|s| *s == 0) {
            return Err(IslandRefusal::Empty(i));
        }
        let n = self.sizes.len();
        let mut next: Vec<Vec<usize>> = vec![Vec::new(); n];
        match &self.links {
            None => {
                for (a, out) in next.iter_mut().enumerate() {
                    out.extend((0..n).filter(|b| *b != a));
                }
            }
            Some(links) => {
                for &(a, b) in links {
                    if a >= n || b >= n {
                        return Err(IslandRefusal::NoSuchIsland(a.max(b)));
                    }
                    if let Some(x) = next.get_mut(a) {
                        x.push(b);
                    }
                    if let Some(x) = next.get_mut(b) {
                        x.push(a);
                    }
                }
            }
        }
        let mut hops = Vec::with_capacity(n);
        for from in 0..n {
            let mut dist: Vec<Option<u64>> = vec![None; n];
            let mut queue = VecDeque::from([from]);
            if let Some(d) = dist.get_mut(from) {
                *d = Some(0);
            }
            while let Some(a) = queue.pop_front() {
                let here = dist.get(a).copied().flatten().unwrap_or(0);
                for &b in next.get(a).into_iter().flatten() {
                    if let Some(d @ None) = dist.get_mut(b) {
                        *d = Some(here + 1);
                        queue.push_back(b);
                    }
                }
            }
            let row: Option<Vec<u64>> = dist.into_iter().collect();
            hops.push(row.ok_or(IslandRefusal::NotConnected)?);
        }
        Ok(hops)
    }
}
