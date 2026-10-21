//! Union-by-rank disjoint set matching upstream `ds_union_by_rank`.
//!
//! On equal ranks the first argument's root becomes the parent. Upstream's `ds_find` never
//! compresses paths; compression does not change roots, so path halving is observationally
//! identical.

/// Rank disjoint set over `0..n`.
#[derive(Debug, Clone)]
pub struct RankDsu {
    parent: Vec<usize>,
    rank: Vec<u32>,
}

impl RankDsu {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Representative of `x`, halving the path.
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    /// Union two roots; `x` wins rank ties.
    pub fn union_roots(&mut self, mut x: usize, mut y: usize) {
        if x == y {
            return;
        }
        if self.rank[x] < self.rank[y] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent[y] = x;
        if self.rank[x] == self.rank[y] {
            self.rank[x] += 1;
        }
    }

    /// `ds_union_by_rank(x, y)`.
    pub fn union(&mut self, x: usize, y: usize) {
        let (a, b) = (self.find(x), self.find(y));
        self.union_roots(a, b);
    }
}
