//! Literal argsort port of Numba's quicksort implementation used by epsilon search.

/// Return the Numba quicksort permutation for an integer slice.
pub(crate) fn argsort(values: &[usize]) -> Vec<usize> {
    const SMALL_QUICKSORT: usize = 15;
    const MAX_STACK: usize = 100;
    let mut order: Vec<usize> = (0..values.len()).collect();
    if order.len() < 2 {
        return order;
    }
    let mut stack = [(0usize, 0usize); MAX_STACK];
    stack[0] = (0, order.len() - 1);
    let mut depth = 1;
    while depth > 0 {
        depth -= 1;
        let (mut low, mut high) = stack[depth];
        while high - low >= SMALL_QUICKSORT {
            debug_assert!(depth < MAX_STACK);
            let mid = (low + high) >> 1;
            if values[order[mid]] < values[order[low]] {
                order.swap(low, mid);
            }
            if values[order[high]] < values[order[mid]] {
                order.swap(high, mid);
            }
            if values[order[mid]] < values[order[low]] {
                order.swap(low, mid);
            }
            let pivot = values[order[mid]];
            order.swap(high, mid);
            let (mut i, mut j) = (low, high - 1);
            loop {
                while i < high && values[order[i]] < pivot {
                    i += 1;
                }
                while j >= low && pivot < values[order[j]] {
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                order.swap(i, j);
                i += 1;
                j -= 1;
            }
            order.swap(i, high);
            if high - i > i - low {
                if high > i {
                    stack[depth] = (i + 1, high);
                    depth += 1;
                }
                if i == low {
                    low = high;
                    break;
                }
                high = i - 1;
            } else {
                if i > low {
                    stack[depth] = (low, i - 1);
                    depth += 1;
                }
                if i == high {
                    low = high;
                    break;
                }
                low = i + 1;
            }
        }
        for i in low.saturating_add(1)..=high {
            let key = order[i];
            let mut j = i;
            while j > low && values[key] < values[order[j - 1]] {
                order[j] = order[j - 1];
                j -= 1;
            }
            order[j] = key;
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::argsort;

    #[test]
    fn argsort_matches_literal_insertion_and_quicksort_shapes() {
        assert_eq!(argsort(&[4, 1, 1, 3]), vec![1, 2, 3, 0]);
        let values: Vec<_> = (0..40).map(|i| (i * 17 + 3) % 11).collect();
        let result = argsort(&values);
        assert!(
            result
                .windows(2)
                .all(|pair| values[pair[0]] <= values[pair[1]])
        );
        let mut sorted = result.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..values.len()).collect::<Vec<_>>());
    }

    #[test]
    fn tied_permutations_match_pinned_numba() {
        // Recorded from the pinned environment with @numba.njit np.argsort(int64_values).
        let values: Vec<_> = (0..40).map(|i| (i * 17 + 3) % 11).collect();
        assert_eq!(
            argsort(&values[..16]),
            vec![5, 7, 9, 11, 0, 2, 13, 15, 4, 6, 8, 10, 1, 12, 3, 14]
        );
        assert_eq!(
            argsort(&values),
            vec![
                38, 27, 16, 5, 18, 29, 7, 20, 9, 31, 22, 11, 0, 33, 35, 13, 2, 24, 15, 4, 37, 26,
                28, 39, 6, 17, 8, 19, 30, 32, 10, 21, 23, 34, 1, 12, 14, 25, 36, 3
            ]
        );
    }
}
