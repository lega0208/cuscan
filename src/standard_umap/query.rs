//! Bounded best-first search over a retained training proximity graph.

pub(crate) const TRANSFORM_SCRATCH_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryScratchLayout {
    pub(crate) per_row_bytes: usize,
    pub(crate) max_chunk_rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryOutputAllocation {
    pub(crate) output_indices_bytes: usize,
    pub(crate) output_distances_bytes: usize,
    pub(crate) stats_bytes: usize,
}

pub(crate) fn approximate_query_scratch_layout(
    query_rows: usize,
    candidate_capacity: usize,
    visit_capacity: usize,
    graph_width: usize,
) -> crate::standard_umap::Result<QueryScratchLayout> {
    approximate_query_scratch_layout_with_limit(
        query_rows,
        candidate_capacity,
        visit_capacity,
        graph_width,
        TRANSFORM_SCRATCH_BUDGET_BYTES,
    )
}

fn approximate_query_scratch_layout_with_limit(
    query_rows: usize,
    candidate_capacity: usize,
    visit_capacity: usize,
    graph_width: usize,
    scratch_limit_bytes: usize,
) -> crate::standard_umap::Result<QueryScratchLayout> {
    use crate::standard_umap::UmapError;

    if query_rows == 0 || candidate_capacity == 0 || visit_capacity == 0 || graph_width == 0 {
        return Err(UmapError::InvalidInput(
            "approximate-query scratch dimensions must be nonzero".into(),
        ));
    }

    let int_bytes = std::mem::size_of::<i32>();
    let float_bytes = std::mem::size_of::<f32>();
    let candidate_ids = candidate_capacity
        .checked_mul(int_bytes)
        .ok_or(UmapError::Overflow("approximate-query candidate scratch"))?;
    let candidate_distances = candidate_capacity
        .checked_mul(float_bytes)
        .ok_or(UmapError::Overflow("approximate-query candidate scratch"))?;
    let candidate_expanded = candidate_capacity
        .checked_mul(int_bytes)
        .ok_or(UmapError::Overflow("approximate-query candidate scratch"))?;
    let candidate_bytes = candidate_ids
        .checked_add(candidate_distances)
        .and_then(|bytes| bytes.checked_add(candidate_expanded))
        .ok_or(UmapError::Overflow("approximate-query candidate scratch"))?;

    let visit_ids = visit_capacity
        .checked_mul(int_bytes)
        .ok_or(UmapError::Overflow("approximate-query visit scratch"))?;
    let visit_distances = visit_capacity
        .checked_mul(float_bytes)
        .ok_or(UmapError::Overflow("approximate-query visit scratch"))?;
    let visit_bytes = visit_ids
        .checked_add(visit_distances)
        .ok_or(UmapError::Overflow("approximate-query visit scratch"))?;

    // Include conservative fuzzy-graph proposal and reduction storage even
    // though the serial L3 query kernel does not allocate either buffer.
    let proposal_ids = graph_width
        .checked_mul(int_bytes)
        .ok_or(UmapError::Overflow("approximate-query proposal scratch"))?;
    let proposal_weights = graph_width
        .checked_mul(float_bytes)
        .ok_or(UmapError::Overflow("approximate-query proposal scratch"))?;
    let proposal_bytes = proposal_ids
        .checked_add(proposal_weights)
        .ok_or(UmapError::Overflow("approximate-query proposal scratch"))?;
    let reduction_bytes = 32usize
        .checked_mul(float_bytes)
        .ok_or(UmapError::Overflow("approximate-query reduction scratch"))?;

    let per_row_bytes = candidate_bytes
        .checked_add(visit_bytes)
        .and_then(|bytes| bytes.checked_add(proposal_bytes))
        .and_then(|bytes| bytes.checked_add(reduction_bytes))
        .ok_or(UmapError::Overflow("approximate-query scratch total"))?;
    if per_row_bytes > scratch_limit_bytes {
        return Err(UmapError::InvalidInput(format!(
            "approximate-query per-row scratch ({per_row_bytes} bytes) exceeds the transform scratch budget ({scratch_limit_bytes} bytes)"
        )));
    }
    let max_chunk_rows = (scratch_limit_bytes / per_row_bytes).min(query_rows);
    debug_assert!(max_chunk_rows > 0);
    Ok(QueryScratchLayout {
        per_row_bytes,
        max_chunk_rows,
    })
}

pub(crate) fn query_output_allocation(
    query_rows: usize,
    k: usize,
) -> crate::standard_umap::Result<QueryOutputAllocation> {
    use crate::standard_umap::UmapError;

    let output_elements = query_rows
        .checked_mul(k)
        .ok_or(UmapError::Overflow("approximate-query output elements"))?;
    let stats_elements = query_rows
        .checked_mul(2)
        .ok_or(UmapError::Overflow("approximate-query stats elements"))?;
    let output_indices_bytes = output_elements
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(UmapError::Overflow(
            "approximate-query index output allocation",
        ))?;
    let output_distances_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(UmapError::Overflow(
            "approximate-query distance output allocation",
        ))?;
    let _output_bytes = output_indices_bytes
        .checked_add(output_distances_bytes)
        .ok_or(UmapError::Overflow("approximate-query output allocation"))?;
    let stats_bytes = stats_elements
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or(UmapError::Overflow("approximate-query stats allocation"))?;
    Ok(QueryOutputAllocation {
        output_indices_bytes,
        output_distances_bytes,
        stats_bytes,
    })
}

#[cfg(any(feature = "cpu", feature = "cuda", test))]
pub(crate) fn query_chunk_rows(
    query_rows: usize,
    row_start: usize,
    max_chunk_rows: usize,
) -> usize {
    assert!(row_start < query_rows && max_chunk_rows > 0);
    (query_rows - row_start).min(max_chunk_rows)
}

#[cfg(test)]
mod scratch_tests {
    use super::{
        approximate_query_scratch_layout_with_limit, query_chunk_rows, query_output_allocation,
    };
    use crate::standard_umap::UmapError;

    #[test]
    fn approximate_query_scratch_budget_checks_overflow() {
        assert!(matches!(
            approximate_query_scratch_layout_with_limit(1, usize::MAX, 1, 1, usize::MAX),
            Err(UmapError::Overflow(_))
        ));
        assert!(matches!(
            query_output_allocation(usize::MAX, 2),
            Err(UmapError::Overflow(_))
        ));
        assert!(matches!(
            query_output_allocation(usize::MAX / 2 + 1, 0),
            Err(UmapError::Overflow(_))
        ));
    }

    #[test]
    fn approximate_query_scratch_budget_rejects_overbudget_row() {
        let error = approximate_query_scratch_layout_with_limit(1, 3, 5, 2, 219).unwrap_err();
        assert!(matches!(error, UmapError::InvalidInput(message) if message.contains("scratch")));
    }

    #[test]
    fn approximate_query_scratch_budget_accepts_exact_limit() {
        let layout = approximate_query_scratch_layout_with_limit(3, 3, 5, 2, 220).unwrap();
        assert_eq!(layout.per_row_bytes, 220);
        assert_eq!(layout.max_chunk_rows, 1);
    }

    #[test]
    fn approximate_query_chunk_math_keeps_tail_rows() {
        let layout = approximate_query_scratch_layout_with_limit(5, 3, 5, 2, 440).unwrap();
        assert_eq!(layout.max_chunk_rows, 2);
        let lengths = (0..5)
            .step_by(layout.max_chunk_rows)
            .map(|start| query_chunk_rows(5, start, layout.max_chunk_rows))
            .collect::<Vec<_>>();
        assert_eq!(lengths, [2, 2, 1]);
    }
}
