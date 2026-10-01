"""Contract tests for the C0/C1/C2/C3 cache experiment harness; synthetic rows only."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/cache_experiment"))
import cache_experiment as ce  # noqa: E402
import latency  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/cache_question_hash_java.json"


def unit_vector(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def pool_frame(rows):
    """rows: (sample_id, group, text, label, quality, cost, family, gold, answer, options)."""
    records = []
    for sample, group, text, label, quality, cost, family, gold, answer, options in rows:
        records.append({"sample_id": sample, "duplicate_group_id": group, "request_text": text,
                        "key_hash": ce.question_hash(text), "source_dataset": "routerbench", "task_family": family,
                        "gold": gold, "options_json": json.dumps(options), "routed_label": label,
                        "routed_tier": ce.LABEL_TO_TIER[label], "routed_answer": answer, "routed_quality": quality,
                        "routed_success": quality, "routed_actual_cost": cost, "routed_latency_s": np.nan,
                        "source_task": "mmlu-x"})
    return pd.DataFrame(records)


def trace_of(samples):
    seen, rows = {}, []
    for i, s in enumerate(samples):
        rows.append((i, s, s in seen, seen.setdefault(s, i)))
    return pd.DataFrame(rows, columns=["position", "sample_id", "is_repeat", "first_position"])


class JavaParityTest(unittest.TestCase):
    def test_question_hash_matches_backend_java(self):
        cases = json.loads(FIXTURE.read_text())["cases"]
        self.assertGreaterEqual(len(cases), 18)
        for case in cases:
            raw = bytes.fromhex(case["utf16_hex"]).decode("utf-16-be", "surrogatepass")
            self.assertEqual(ce.question_hash(raw), case["sha256"], repr(raw))

    def test_normalization_keeps_case_and_nbsp(self):
        self.assertEqual(ce.normalize_question("  a \t b\n"), "a b")
        self.assertNotEqual(ce.question_hash("Paris"), ce.question_hash("paris"))
        self.assertEqual(ce.normalize_question("a b"), "a b")
        self.assertEqual(ce.normalize_question("a b"), "a b")


class OperationalSqlTest(unittest.TestCase):
    def test_sql_is_read_from_cache_service(self):
        sql = latency.operational_sql()
        self.assertIn("ORDER BY embedding <=> %s::vector LIMIT %s", sql["SELECT_SQL"])
        self.assertIn("WHERE t.similarity >= t.threshold", sql["SELECT_SQL"])
        self.assertIn("WHERE question_hash = %s ORDER BY updated_at DESC LIMIT 1", sql["SELECT_EXACT_SQL"])
        self.assertEqual(sql["INSERT_SQL"].count("%s"), 8)
        self.assertIn("hit_count = hit_count + 1", sql["RECORD_HIT"])

    def test_cache_entry_ddl_is_extracted(self):
        ddl = latency.cache_entry_ddl()
        self.assertTrue(any("vector(1024)" in d for d in ddl))
        self.assertTrue(any("USING hnsw (embedding vector_cosine_ops)" in d for d in ddl))
        self.assertTrue(any("question_hash" in d and "CREATE INDEX" in d for d in ddl))


class AnswerScoringTest(unittest.TestCase):
    def test_letter_extraction(self):
        self.assertEqual(ce.rb_letter("D"), ("D", "ok"))
        self.assertEqual(ce.rb_letter("\nA)"), ("A", "ok"))
        self.assertEqual(ce.rb_letter("B) because"), ("B", "ok"))
        self.assertEqual(ce.rb_letter("The answer is C."), ("C", "ok"))
        self.assertEqual(ce.rb_letter("I think 42 apples"), (None, "none"))
        self.assertEqual(ce.rb_letter("A cat sat"), (None, "none"))
        self.assertEqual(ce.rb_letter("(B) since"), ("B", "ok"))
        self.assertEqual(ce.xrb_letter("so (B) then (B)"), ("B", "ok"))
        self.assertEqual(ce.xrb_letter("(A) or (C)")[1], "ambiguous")
        self.assertEqual(ce.choice_letter("C"), ("C", "ok"))

    def test_options_and_numbers(self):
        rb = "Q?\nA) red\nB) Blue.\nPrint only a single choice"
        self.assertEqual(ce.parse_options(rb, "routerbench"), {"A": "red", "B": "blue"})
        xrb = "Question: q\n(A) one\n(B) two"
        self.assertEqual(ce.parse_options(xrb, "xroutebench"), {"A": "one", "B": "two"})
        self.assertEqual(ce.parse_options("A) x\nC) y", "routerbench"), {})
        self.assertEqual(ce.numeric_answer("so \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")
        self.assertEqual(ce.as_number("\\frac{1}{2}"), 0.5)
        self.assertEqual(ce.numeric_answer("... #### 1,234"), "1,234")
        self.assertEqual(ce.xrb_gold("GSM8K", "work\n#### 18"), ("numeric", "18"))
        self.assertEqual(ce.xrb_gold("em_mc", "C"), ("mcq", "C"))
        self.assertEqual(ce.xrb_gold("f1_score", "Latin")[0], "unscorable")

    def test_reuse_needs_letter_and_option_text_to_agree(self):
        entry = {"duplicate_group_id": "g1", "key_hash": "k1", "routed_answer": "B",
                 "options_json": json.dumps({"A": "paris", "B": "rome"})}
        same_options = {"duplicate_group_id": "g2", "key_hash": "k2", "task_family": "mcq", "gold": "B",
                        "source_dataset": "routerbench", "options_json": json.dumps({"A": "paris", "B": "rome"})}
        self.assertEqual(ce.reuse_outcome(entry, same_options), ce.REUSE_CORRECT)
        coincidence = dict(same_options, options_json=json.dumps({"A": "rome", "B": "oslo"}))
        self.assertEqual(ce.reuse_outcome(entry, coincidence), ce.REUSE_UNKNOWN)
        wrong = dict(same_options, gold="A", options_json=json.dumps({"A": "oslo", "B": "lima"}))
        self.assertEqual(ce.reuse_outcome(entry, wrong), ce.REUSE_INCORRECT)
        essay = dict(entry, routed_answer="Let me compute 12 * 3 = 36")
        self.assertEqual(ce.reuse_outcome(essay, same_options), ce.REUSE_INCORRECT)
        self.assertEqual(ce.reuse_outcome(entry, dict(same_options, duplicate_group_id="g1")), ce.REUSE_RECORDED)
        self.assertEqual(ce.reuse_outcome(entry, dict(same_options, task_family="unscorable")), ce.REUSE_UNKNOWN)

    def test_numeric_reuse(self):
        entry = {"duplicate_group_id": "g1", "key_hash": "k1", "routed_answer": "thus \\boxed{18}", "options_json": "{}"}
        target = {"duplicate_group_id": "g2", "key_hash": "k2", "task_family": "numeric", "gold": "18",
                  "source_dataset": "xroutebench", "options_json": "{}"}
        self.assertEqual(ce.reuse_outcome(entry, target), ce.REUSE_CORRECT)
        self.assertEqual(ce.reuse_outcome(entry, dict(target, gold="19")), ce.REUSE_INCORRECT)
        self.assertEqual(ce.reuse_outcome(dict(entry, routed_answer="no number"), target), ce.REUSE_INCORRECT)


class TraceTest(unittest.TestCase):
    def test_repeat_count_prefix_and_causality(self):
        ids = np.array([f"q{i}" for i in range(500)])
        base = ce.make_trace(ids, 200, 0.0, 7)
        for r in (0.1, 0.3, 0.5):
            t = ce.make_trace(ids, 200, r, 7)
            self.assertEqual(int(t.is_repeat.sum()), round(r * 200))
            self.assertFalse(bool(t.is_repeat.iat[0]))
            new = t[~t.is_repeat].sample_id.tolist()
            self.assertEqual(new, base.sample_id.tolist()[: len(new)])
            first = dict(zip(t[~t.is_repeat].sample_id, t[~t.is_repeat].position))
            for row in t[t.is_repeat].itertuples():
                self.assertLess(first[row.sample_id], row.position)
            pd.testing.assert_frame_equal(t, ce.make_trace(ids, 200, r, 7))
        self.assertFalse(ce.make_trace(ids, 200, 0.3, 8).equals(ce.make_trace(ids, 200, 0.3, 7)))

    def test_trace_needs_enough_questions(self):
        with self.assertRaises(ValueError):
            ce.make_trace(np.array(["a", "b"]), 10, 0.0, 1)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        mcq = {"A": "x", "B": "y"}
        self.pool = pool_frame([
            ("s0", "g0", "What is the capital of France?", "easy", 1.0, 0.001, "mcq", "A", "A", mcq),
            ("s1", "g1", "Which city is the capital of France?", "hard", 1.0, 0.1, "mcq", "A", "A", mcq),
            ("s2", "g2", "What is the capital of Spain?", "medium", 0.0, 0.01, "mcq", "B", "A", mcq),
            ("s3", "g3", "Unrelated question about cooking", "easy", 1.0, 0.001, "unscorable", None, "stir", {}),
        ])
        self.vectors = np.stack([unit_vector(1, 0, 0), unit_vector(0.95, 0.312, 0), unit_vector(0.85, 0.527, 0),
                                 unit_vector(0, 0, 1)])
        self.policies = ce.config_policies(ce.load_config())

    def rerank(self, table):
        texts = self.pool.request_text.tolist()
        return ce.PairScorer(lambda q, cands: [table.get((q, c), 0.0) for c in cands], texts)

    def test_c0_calls_every_request(self):
        log = ce.replay(trace_of(["s0", "s0", "s1"]), self.pool, self.policies["C0"])
        self.assertTrue(log.llm_called.all())
        self.assertAlmostEqual(log.llm_cost.sum(), 0.001 + 0.001 + 0.1)

    def test_c1_hits_exact_repeats_only(self):
        log = ce.replay(trace_of(["s0", "s1", "s0"]), self.pool, self.policies["C1"])
        self.assertEqual(log.decision.tolist(), ["miss", "miss", "exact"])
        self.assertEqual(log.reuse.iat[2], ce.REUSE_RECORDED)
        self.assertEqual(log.embeds.sum(), 0)
        ce.exact_arm_guard(log)

    def test_c2_reuses_across_questions_and_flags_relation_error(self):
        log = ce.replay(trace_of(["s0", "s1", "s2"]), self.pool, self.policies["C2"], self.vectors)
        self.assertEqual(log.decision.tolist(), ["miss", "semantic", "semantic"])
        self.assertTrue(log.relation_error.iat[1])
        self.assertEqual(log.reuse.iat[1], ce.REUSE_CORRECT)
        self.assertEqual(log.reuse.iat[2], ce.REUSE_INCORRECT)
        self.assertEqual(log.llm_called.sum(), 1)

    def test_c3_rejects_low_scores_and_falls_back_to_miss(self):
        texts = self.pool.request_text.tolist()
        scorer = self.rerank({(texts[1], texts[0]): 0.9, (texts[2], texts[0]): 0.001, (texts[2], texts[1]): 0.001})
        log = ce.replay(trace_of(["s0", "s1", "s2"]), self.pool, self.policies["C3"], self.vectors, scorer)
        self.assertEqual(log.decision.tolist(), ["miss", "semantic", "miss"])
        self.assertEqual(log.rerank_reason.iat[2], "rejected")
        self.assertTrue(log.inserted.iat[2])

    def test_c3_requires_rerank_cosine_even_with_high_score(self):
        texts = self.pool.request_text.tolist()
        scorer = self.rerank({(texts[2], texts[0]): 0.99})
        log = ce.replay(trace_of(["s0", "s2"]), self.pool, self.policies["C3"], self.vectors, scorer)
        # cosine .85 clears the .80 floor but not the .8774 reranker cosine
        self.assertEqual(log.decision.iat[1], "miss")
        self.assertEqual(log.candidate_count.iat[1], 1)

    def test_rerank_can_reorder_candidates(self):
        pool = self.pool.copy()
        vectors = np.stack([unit_vector(1, 0.02, 0), unit_vector(1, 0.05, 0), unit_vector(1, 0, 0), unit_vector(0, 0, 1)])
        texts = pool.request_text.tolist()
        table = {(texts[1], texts[0]): 0.0, (texts[2], texts[0]): 0.2, (texts[2], texts[1]): 0.8}
        scorer = ce.PairScorer(lambda q, c: [table[(q, x)] for x in c], texts)
        log = ce.replay(trace_of(["s0", "s1", "s2"]), pool, self.policies["C3"], vectors, scorer)
        self.assertEqual(log.source_sample_id.iat[2], "s1")
        self.assertEqual(log.rerank_reason.iat[2], "accepted_reordered")

    def test_threshold_is_stored_as_float32(self):
        cache = ce.MemoryCache(3)
        cache.insert(0, "k", unit_vector(1, 0, 0), "easy", 0.80)
        self.assertEqual(cache.thresholds[0], float(np.float32(0.80)))
        at = np.float32(0.80)
        self.assertEqual(len(cache.candidates(np.array([at, 0, 0], dtype=np.float32), 5)), 1)
        below = np.nextafter(at, np.float32(0))
        self.assertEqual(cache.candidates(np.array([below, 0, 0], dtype=np.float32), 5), [])

    def test_exact_prefers_most_recently_updated_and_lru_evicts(self):
        cache = ce.MemoryCache(0, capacity=2)
        a = cache.insert(0, "k", None, "easy", 0.8)
        b = cache.insert(1, "k", None, "easy", 0.8)
        self.assertEqual(cache.exact("k"), b)
        cache.touch(a)
        self.assertEqual(cache.exact("k"), a)
        cache.insert(2, "z", None, "easy", 0.8)
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(cache.exact("k"), a)

    def test_capacity_changes_hits(self):
        policy = ce.Policy(**{**self.policies["C1"].__dict__, "capacity": 1, "name": "C1-cap1"})
        log = ce.replay(trace_of(["s0", "s3", "s0"]), self.pool, policy)
        self.assertEqual(log.decision.iat[2], "miss")
        self.assertEqual(log.attrs["evictions"], 2)

    def test_metrics_and_bootstrap(self):
        trace = trace_of(["s0", "s1", "s0", "s2", "s3", "s0"])
        logs = {a: ce.replay(trace, self.pool, self.policies[a], self.vectors) for a in ("C0", "C1", "C2")}
        sums = {a: ce.group_sums(l).sum() for a, l in logs.items()}
        m = ce.ratios(sums["C1"], sums["C0"])
        self.assertAlmostEqual(m["hit_rate"], 2 / 6)
        self.assertAlmostEqual(m["call_reduction"], 2 / 6)
        self.assertEqual(m["false_reuse_among_hits"], 0.0)
        c2 = ce.ratios(sums["C2"], sums["C0"])
        self.assertGreater(c2["false_reuse_per_request"], 0)
        boot = ce.paired_bootstrap(logs, "C0", 200, 1)
        row = boot[(boot.arm == "C1") & (boot.metric == "hit_rate")].iloc[0]
        self.assertLessEqual(row.ci_low, row.point)
        self.assertGreaterEqual(row.ci_high, row.point)
        with self.assertRaises(ValueError):
            ce.paired_bootstrap({"C0": logs["C0"], "C1": logs["C1"].iloc[:3]}, "C0", 10, 1)

    def test_c1_guard_catches_cross_question_exact_hit(self):
        bad = pd.DataFrame({"decision": ["exact"], "relation_error": [True]})
        with self.assertRaises(AssertionError):
            ce.exact_arm_guard(bad)

    def test_rerank_memo_is_per_batch(self):
        calls = []
        scorer = ce.PairScorer(lambda q, c: calls.append(tuple(c)) or [0.5] * len(c), ["q", "a", "b"])
        scorer(0, [1, 2]); scorer(0, [1, 2]); scorer(0, [1])
        self.assertEqual(calls, [("a", "b"), ("a",)])


class D2Test(unittest.TestCase):
    def setUp(self):
        self.config = ce.load_config()

    def test_selection_respects_quotas_and_is_order_independent(self):
        rows = []
        for i in range(600):
            source = "xroutebench" if i % 5 == 0 else "routerbench"
            family = ["mcq", "numeric", "unscorable"][i % 3] if source == "xroutebench" else ["mcq", "unscorable"][i % 2]
            task = "grade-school-math" if (source == "routerbench" and i % 4 == 1) else "mmlu-x"
            rows.append({"sample_id": f"s{i:03d}", "source_dataset": source, "task_family": family, "source_task": task,
                         "routed_answer": "A", "request_text": f"q{i}"})
        pool = pd.DataFrame(rows)
        tokens = {s: 50 for s in pool.sample_id}
        config = json.loads(json.dumps(self.config))
        config["d2"]["strata"] = {"rb_mcq": 9, "rb_math": 6, "xrb_mcq": 3}
        a = ce.select_d2_originals(pool, config, tokens)
        b = ce.select_d2_originals(pool.sample(frac=1, random_state=3), config, tokens)
        self.assertEqual(sorted(a.sample_id), sorted(b.sample_id))
        self.assertEqual(a.stratum.value_counts().to_dict(), {"rb_mcq": 9, "rb_math": 6, "xrb_mcq": 3})
        self.assertEqual(int(a.d2_split.eq("calibration").sum()), 3 + 2 + 1)
        config["d2"]["strata"]["rb_math"] = 1000
        with self.assertRaises(ValueError):
            ce.select_d2_originals(pool, config, tokens)
        with self.assertRaises(ValueError):
            ce.select_d2_originals(pool, self.config, dict(tokens, s000=513))

    def test_authoring_validation(self):
        originals = pd.DataFrame({"group_id": ["d2-000"], "request_text": ["What is 2+2?"], "stratum": ["rb_math"]})
        sheet = pd.DataFrame({"group_id": ["d2-000"], "paraphrase_text": ["What is 2+2? "],
                              "hard_negative_text": [""], "author": [""]})
        problems = ce.validate_authoring(sheet, originals)
        self.assertEqual(set(problems.problem), {"identical_after_whitespace_normalization", "empty", "missing"})

    def test_validation_accepts_the_generated_sheet_shape(self):
        originals = pd.DataFrame({"group_id": ["d2-000"], "d2_split": ["test"], "stratum": ["rb_mcq"],
                                  "negative_type": ["negation"], "sample_id": ["s"], "source_task": ["t"],
                                  "request_text": ["a b c d"], "routed_answer": ["A"], "routed_quality": [1.0], "gold": ["A"]})
        sheet = ce.authoring_sheet(originals).assign(paraphrase_text="a b c e", hard_negative_text="a b c d e",
                                                      author="x")
        words = lambda text: len(text.split())
        self.assertTrue(ce.validate_authoring(sheet, originals, words, 5).empty)
        bad = sheet.assign(hard_negative_text="a b c d e f")
        self.assertEqual(ce.validate_authoring(bad, originals, words, 5).problem.tolist(),
                         ["exceeds_service_input_limit_5"])

    def test_annotation_sheet_hides_role_and_merge(self):
        originals = pd.DataFrame({"group_id": ["g"], "sample_id": ["s"], "request_text": ["q"], "routed_answer": ["A"],
                                  "gold": ["A"], "d2_split": ["test"], "stratum": ["rb_mcq"], "negative_type": ["negation"]})
        sheet = pd.DataFrame({"group_id": ["g"], "paraphrase_text": ["q2"], "hard_negative_text": ["not q"], "author": ["x"]})
        variants = ce.variants_from(sheet, originals)
        blank = ce.annotation_sheet(variants, originals, 1)
        self.assertNotIn("role", blank.columns)
        self.assertFalse(blank.item_id.str.contains(r"-[pn]$").any())
        self.assertEqual(sorted(ce.unblind(blank, variants, 1).item_id), sorted(variants.item_id))
        self.assertTrue(ce.unblind(variants[["item_id"]], variants, 1).equals(variants[["item_id"]]))
        with self.assertRaises(ValueError):
            ce.unblind(blank.assign(item_id="vbogus"), variants, 1)
        blank = ce.unblind(blank, variants, 1)
        a = blank.assign(semantic_equivalent=["Y", "N"], cached_answer_correct_for_target=["Y", "N"], annotator="ann1")
        b = blank.assign(semantic_equivalent=["Y", "Y"], cached_answer_correct_for_target=["Y", "N"], annotator="ann2")
        merged, pending, report = ce.merge_annotations(a, b)
        self.assertEqual(len(pending), 1)
        self.assertLess(report["semantic_equivalent"]["agreement"], 1.0)
        item = pending.item_id.iat[0]
        adj = pd.DataFrame({"item_id": [item], "field": ["semantic_equivalent"], "final": ["N"], "reason": ["changes the answer"]})
        merged, pending, _ = ce.merge_annotations(a, b, adj)
        self.assertTrue(pending.empty)
        self.assertFalse(merged.semantic_equivalent.isna().any())
        with self.assertRaises(ValueError):
            ce.merge_annotations(a, a)

    def test_quality_flags_reach_variants(self):
        originals = pd.DataFrame({"group_id": ["g1", "g2"], "sample_id": ["s1", "s2"], "d2_split": ["test"] * 2,
                                  "stratum": ["rb_mcq"] * 2, "negative_type": ["negation"] * 2})
        sheet = pd.DataFrame({"group_id": ["g1", "g2"], "paraphrase_text": ["a", "b"], "hard_negative_text": ["c", "d"],
                              "author": ["x", "x"], "author_notes": ["함정 목표 정답: B / low-quality trap", ""]})
        variants = ce.variants_from(sheet, originals)
        self.assertEqual(variants.set_index("item_id").quality_flags.to_dict(),
                         {"g1-p": "low-quality trap", "g1-n": "low-quality trap", "g2-p": "", "g2-n": ""})
        self.assertEqual(ce.quality_flags("source anomaly: x"), ["source anomaly"])

    def test_kappa(self):
        self.assertAlmostEqual(ce.cohen_kappa(["Y", "N", "Y", "N"], ["Y", "N", "Y", "N"]), 1.0)
        self.assertLess(ce.cohen_kappa(["Y", "N", "Y", "N"], ["N", "Y", "N", "Y"]), 0)

    def _d2_fixture(self):
        cache = pd.DataFrame({"sample_id": ["o1", "o2", "bg"], "duplicate_group_id": ["o1", "o2", "bg"],
                              "key_hash": ["h1", "h2", "h3"], "routed_answer": ["A", "B", "C"], "task_family": "mcq",
                              "gold": ["A", "B", "C"], "options_json": json.dumps({"A": "a", "B": "b", "C": "c"}),
                              "source_dataset": "routerbench"})
        items = pd.DataFrame({"item_id": ["i1p", "i1n", "i2p", "i2n"], "group_id": ["g1", "g1", "g2", "g2"],
                              "role": ["paraphrase", "hard_negative"] * 2, "stratum": "rb_mcq",
                              "negative_type": [None, "negation", None, "entity_swap"],
                              "original_sample_id": ["o1", "o1", "o2", "o2"], "d2_split": "test",
                              "key_hash": ["x1", "x2", "x3", "x4"]})
        labels = items.assign(semantic_equivalent=["Y", "N", "Y", "N"], cached_answer_correct_for_target=["Y", "N", "Y", "N"])
        sims = np.array([[0.95, 0.5, 0.1], [0.93, 0.6, 0.2], [0.85, 0.84, 0.3], [0.97, 0.4, 0.1]])
        order = np.array([[0, 1, 2], [0, 1, 2], [1, 2, 0], [1, 0, 2]])
        scores = np.array([[0.9, 0.1, 0.0], [0.01, 0.1, 0.0], [0.7, 0.2, 0.0], [0.5, 0.0, 0.0]])
        relation, status = ce.candidate_status(items, order, cache, labels, cache)
        return items, labels, order, sims, scores, relation, status, cache

    def test_production_specs_match_semantics(self):
        items, labels, order, sims, scores, relation, status, cache = self._d2_fixture()
        specs = ce.production_specs(self.config)
        c2 = ce.d2_metrics(ce.evaluate_spec(specs["C2"], items, labels, order, sims, scores, relation, status, cache))
        self.assertEqual((c2["hits"], c2["false_reuse"]), (4, 2))
        c3 = ce.evaluate_spec(specs["C3"], items, labels, order, sims, scores, relation, status, cache)
        # i1n: score .01 < .0358 -> miss; i2p: .85 < .8774 -> miss; i2n: accepted -> relation error
        self.assertEqual(c3.hit.tolist(), [True, False, False, True])
        self.assertEqual(c3.relation_error.tolist(), [False, False, False, True])
        self.assertEqual(c3.answer_status.iat[0], ce.REUSE_CORRECT)

    def test_calibration_respects_limit(self):
        items, labels, order, sims, scores, relation, status, cache = self._d2_fixture()
        config = json.loads(json.dumps(self.config))
        config["calibration"].update(cosine_start=0.8, cosine_stop=0.99, cosine_step=0.01, candidate_limits=[1, 3],
                                     rerank_scores=[0.05, 0.6])
        spec, table = ce.calibrate(items, labels, order, sims, scores, relation, status, cache, config)
        self.assertIsNotNone(spec)
        chosen = ce.d2_metrics(ce.evaluate_spec(spec, items, labels, order, sims, scores, relation, status, cache))
        self.assertEqual(chosen["false_reuse"], 0)
        self.assertEqual(chosen["reusable_hits"], table[table.false_reuse_among_hits <= 0.01].reusable_hits.max())
        op = ce.calibrated_policy_spec(spec)
        ce.policy_from("C_cal", op)
        self.assertEqual(ce.calibrated_policy_spec(None), {"exact": True, "semantic": False})

    def test_clopper_pearson(self):
        lo, hi = ce.clopper_pearson(0, 300)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0122, places=3)


class SealTest(unittest.TestCase):
    def test_exact_json_allows_rerun_but_not_changes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            ce.exact_json(path, {"created_at": "t1", "rows": 3})
            ce.exact_json(path, {"created_at": "t2", "rows": 3})
            with self.assertRaises(ValueError):
                ce.exact_json(path, {"created_at": "t3", "rows": 4})
            ce.exact_json(Path(tmp) / "n.json", {"ratio": float("nan"), "n": np.int64(2)})
            self.assertEqual(ce.read(Path(tmp) / "n.json"), {"ratio": None, "n": 2})


class ConfigTest(unittest.TestCase):
    def test_operational_values(self):
        config = ce.load_config()
        c3 = config["policies"]["C3"]
        self.assertEqual((c3["candidate_limit"], c3["rerank_similarity"], c3["rerank_score"]), (5, 0.8774, 0.03581769))
        self.assertEqual(set(config["policies"]["C2"]["thresholds"].values()), {0.80})
        self.assertEqual(sum(config["d2"]["strata"].values()), 300)
        self.assertEqual(config["service_input_tokens"], 512)
        for stratum in config["d2"]["strata"]:
            self.assertIn(stratum, config["d2"]["negative_types"])
        for spec in config["inputs"].values():
            self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")

    def test_config_matches_backend_application_yml(self):
        text = (ROOT.parent / "backend/src/main/resources/application.yml").read_text()
        for needle in ("candidate-limit: 5", "easy: 0.80", "medium: 0.80", "hard: 0.80",
                       "similarity-threshold: 0.8774", "score-threshold: 0.03581769", "mode: ${RERANK_MODE:enforce}"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
