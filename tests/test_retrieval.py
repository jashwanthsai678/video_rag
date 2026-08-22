"""The ranking logic: keyword scoring, fusion, aggregation, clustering."""

import pytest

from video_rag.retrieval import (
    BM25,
    cluster_by_time,
    confidence,
    group_by,
    rank_score,
    reciprocal_rank_fusion,
    span_of,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("New > Patient Record!") == ["new", "patient", "record"]

    def test_drops_question_scaffolding(self):
        assert tokenize("How do I add a new patient?") == ["add", "new", "patient"]

    def test_keeps_digits(self):
        assert tokenize("Form 1099 field") == ["form", "1099", "field"]

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("the a an of") == []


class TestBM25:
    @pytest.fixture
    def corpus(self):
        return [
            tokenize("Click New then Patient Record to create a patient"),
            tokenize("Export the monthly report as a PDF file"),
            tokenize("Patient billing settings and insurance details"),
            tokenize("Save and close the window"),
        ]

    def test_ranks_the_matching_document_first(self, corpus):
        bm25 = BM25(corpus)
        top = bm25.top_n(tokenize("How do I add a new patient?"), 2)
        assert top[0][0] == 0

    def test_only_scores_documents_containing_a_query_term(self, corpus):
        """The inverted index must not fan out to the whole corpus."""
        scores = BM25(corpus).scores(tokenize("insurance"))
        assert set(scores) == {2}

    def test_unknown_terms_score_nothing(self, corpus):
        assert BM25(corpus).scores(tokenize("helicopter")) == {}

    def test_rarer_terms_outweigh_common_ones(self, corpus):
        bm25 = BM25(corpus)
        # "patient" appears in 2 of 4 docs, "insurance" in 1 — so the doc
        # matched on the rarer term should win.
        assert bm25.idf["insurance"] > bm25.idf["patient"]

    def test_longer_documents_are_not_unfairly_favoured(self):
        short = tokenize("patient record")
        padded = tokenize("patient record " + "filler word here " * 20)
        bm25 = BM25([short, padded])
        scores = bm25.scores(tokenize("patient record"))
        assert scores[0] > scores[1]

    def test_empty_corpus_and_empty_query(self, corpus):
        assert BM25([]).scores(tokenize("anything")) == {}
        assert BM25(corpus).scores([]) == {}

    def test_top_n_is_deterministic_on_ties(self, corpus):
        bm25 = BM25(corpus)
        assert bm25.top_n(tokenize("patient"), 3) == bm25.top_n(tokenize("patient"), 3)


class TestReciprocalRankFusion:
    def test_agreement_between_retrievers_wins(self):
        fused = reciprocal_rank_fusion([[5, 1, 2], [5, 3, 4]])
        assert max(fused, key=fused.get) == 5

    def test_promotes_a_document_only_one_retriever_found(self):
        fused = reciprocal_rank_fusion([[1, 2], [9]])
        assert 9 in fused

    def test_weights_shift_the_balance(self):
        vector_only, keyword_only = [1], [2]
        keyword_heavy = reciprocal_rank_fusion(
            [vector_only, keyword_only], weights=[0.1, 1.0]
        )
        assert keyword_heavy[2] > keyword_heavy[1]

    def test_zero_weight_excludes_a_retriever_entirely(self):
        fused = reciprocal_rank_fusion([[1], [2]], weights=[1.0, 0.0])
        assert set(fused) == {1}

    def test_larger_k_flattens_rank_differences(self):
        tight = reciprocal_rank_fusion([[1, 2]], k=1)
        loose = reciprocal_rank_fusion([[1, 2]], k=1000)
        assert (tight[1] - tight[2]) > (loose[1] - loose[2])

    def test_rejects_mismatched_weights(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[1], [2]], weights=[1.0])

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[], []]) == {}


class TestAggregation:
    def test_concentrated_evidence_beats_a_single_mediocre_hit(self):
        """Regression: averaging made this backwards — a video with three
        strong hits lost to a video with one weaker hit."""
        strong = [0.85, 0.80, 0.75, 0.40, 0.35]
        single = [0.70]
        assert sum(strong) / len(strong) < sum(single) / len(single)  # the old bug
        assert rank_score(strong) > rank_score(single)                # the fix

    def test_uses_only_the_best_n(self):
        assert rank_score([1.0, 1.0, 1.0, 1.0, 1.0], top_n=3) == pytest.approx(3.0)

    def test_a_long_tail_of_weak_hits_cannot_dominate(self):
        assert rank_score([0.9, 0.9, 0.9]) > rank_score([0.3] * 50)

    def test_confidence_is_the_single_best_score(self):
        assert confidence([0.2, 0.91, 0.4]) == pytest.approx(0.91)

    def test_empty_inputs_are_zero(self):
        assert rank_score([]) == 0.0
        assert confidence([]) == 0.0


class TestClusterByTime:
    def test_splits_distant_matches_into_separate_clusters(self):
        """Regression: min-to-max over all matches turned a 10-second answer
        into a 14-minute 'clip' whenever the query also hit elsewhere."""
        spans = [(12.0, 14.0), (18.0, 20.0), (870.0, 875.0)]
        groups = cluster_by_time(spans, max_gap_seconds=30.0)
        assert [len(g) for g in groups] == [2, 1]

    def test_keeps_a_contiguous_run_together(self):
        spans = [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]
        assert len(cluster_by_time(spans, max_gap_seconds=10.0)) == 1

    def test_gap_is_measured_from_the_end_of_the_run(self):
        # A long segment bridges what would otherwise be a large gap.
        spans = [(0.0, 100.0), (105.0, 106.0)]
        assert len(cluster_by_time(spans, max_gap_seconds=10.0)) == 1

    def test_input_order_does_not_change_the_grouping(self):
        spans = [(0.0, 1.0), (2.0, 3.0), (100.0, 101.0)]
        forward = cluster_by_time(spans, 10.0)
        shuffled = cluster_by_time([spans[2], spans[0], spans[1]], 10.0)
        # Same partition of the same moments, regardless of input order.
        as_times = lambda groups, source: [  # noqa: E731
            sorted(source[i] for i in group) for group in groups
        ]
        assert as_times(forward, spans) == as_times(
            shuffled, [spans[2], spans[0], spans[1]]
        )

    def test_returns_indices_into_the_original_list(self):
        spans = [(500.0, 501.0), (1.0, 2.0)]
        groups = cluster_by_time(spans, max_gap_seconds=5.0)
        assert groups == [[1], [0]]

    def test_gap_exactly_at_the_threshold_stays_together(self):
        assert len(cluster_by_time([(0.0, 0.0), (30.0, 30.0)], 30.0)) == 1

    def test_zero_gap_isolates_every_instant(self):
        assert len(cluster_by_time([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], 0.0)) == 3

    def test_empty_input(self):
        assert cluster_by_time([], 30.0) == []

    def test_single_span(self):
        assert cluster_by_time([(4.0, 9.0)], 30.0) == [[0]]


class TestSpanOf:
    def test_bounding_span(self):
        spans = [(10.0, 12.0), (5.0, 6.0), (20.0, 21.0)]
        result = span_of(spans, [0, 1, 2])
        assert (result.start, result.end) == (5.0, 21.0)
        assert result.duration == pytest.approx(16.0)

    def test_only_the_selected_indices_count(self):
        spans = [(10.0, 12.0), (900.0, 901.0)]
        result = span_of(spans, [0])
        assert (result.start, result.end) == (10.0, 12.0)

    def test_empty_selection(self):
        assert span_of([(1.0, 2.0)], []).duration == 0.0


class TestGroupBy:
    def test_groups_and_preserves_insertion_order(self):
        grouped = group_by(["apple", "avocado", "banana"], lambda s: s[0])
        assert list(grouped) == ["a", "b"]
        assert grouped["a"] == ["apple", "avocado"]

    def test_empty(self):
        assert group_by([], lambda x: x) == {}
