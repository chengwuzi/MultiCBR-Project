"""Meal-H round-two plan and inheritance checks; no formal training."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as runner


class MealHRoundTwoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = runner.load_plan(runner.MODEL_DIR / 'Meal-H/搜参配置.yaml')

    def fixture(self):
        plan = copy.deepcopy(self.plan)
        return plan, plan['rounds'][-1]

    def test_actual_reference_grids_and_fixed_settings(self):
        plan, round_ = self.fixture()
        self.assertEqual(round_['id'], 'meal_h_round_02')
        self.assertEqual(len(round_['experiments']), 35)
        expected = [
            ('l2_reg', 'l2_regs', [[1e-6], [5e-6], [5e-5], [1e-4]]),
            ('diffusion_beta_end', 'latent_rebuild', [
                {'latent_diffusion_beta_end': v} for v in [.002, .008, .05, .1]]),
            ('fusion_low_ub', 'fusion_weights', [{'modal_weight': w} for w in [
                [.025, .375, .6], [0., .4, .6], [.025, .325, .65],
                [0., .35, .65], [.025, .3, .675], [0., .3, .7]]]),
            ('bi_layer', 'fusion_weights', [{'BI_layer': w} for w in [
                [.7, .3, 0.], [.3, .7, 0.], [.3, .4, .3], [.1, .3, .6]]]),
            ('bi_noise', 'BI_ratios', [[0.], [.1], [.3], [.5]]),
            ('ui_layer', 'fusion_weights', [{'UI_layer': w} for w in [
                [.6, .3, .1], [.2, .6, .2], [.2, .3, .5], [1/3]*3]]),
            ('ui_noise', 'UI_ratios', [[.025], [.05], [.1], [.2]]),
            ('temperature', 'c_temps', [[.05], [.1], [.3], [.5]]),
        ]
        parents = ['baseline_best']
        by_id = {e['id']: e for e in round_['experiments']}
        for stage, (sid, key, values) in zip(round_['stages'], expected):
            self.assertEqual(stage['id'], sid)
            new_ids = stage['candidates'][len(parents):]
            self.assertTrue(4 <= len(new_ids) <= 6)
            self.assertEqual([by_id[eid]['overrides'] for eid in new_ids], [{key: v} for v in values])
            for eid in new_ids:
                self.assertEqual(by_id[eid]['inherit_best']['candidates'], parents)
            parents += new_ids
            self.assertEqual(stage['candidates'], parents)
        self.assertEqual(round_['final_candidates'], parents)
        runner.resolve_dependencies(plan, round_, runner.empty_state())
        baseline = round_['experiments'][0]
        self.assertEqual(baseline['blocked_by'], [])
        record = baseline['_reference_record']
        self.assertEqual(runner.validation_key(record['training_record']),
                         (.3785844258140611, .24390730448890638))
        self.assertFalse(record['reference_evidence']['artifacts_verified'])
        for item in round_['experiments']:
            c = item['template_config']
            self.assertEqual((c['seed'], c['epochs']), (2026, 150))
            self.assertEqual((c['c_lambdas'], c['latent_rebuild']['rebuild_k']), ([.1], 13))
            self.assertEqual(c['latent_rebuild']['latent_diffusion_set_loss_weight'], .05)
            self.assertTrue(c['latent_rebuild']['enabled'])
            self.assertFalse(c['anchor_cl']['enabled'])
            self.assertEqual(c['anchor_cl']['anchor_cl_lambda'], 0)
            self.assertEqual((c['ui_bundle_user_agg_beta'], c['bi_user_bundle_agg_beta']), (0, 0))

    def test_all_stages_inherit_full_winner_and_keep_control(self):
        # Exercise real config merging/selection with synthetic metrics, including
        # a zero-UB winner, an exact tie, and stages with no improvement.
        plan, round_ = self.fixture()
        state = runner.empty_state()
        native_check = runner.check_success
        def valid(record, item, rd, p):
            return native_check(record, item, rd, p) if item.get('reference') else record is not None
        baseline_score = .3785844258140611
        with patch.object(runner, 'check_success', side_effect=valid):
            runner.resolve_dependencies(plan, round_, state)
            for item in round_['experiments'][1:]:
                runner.resolve_dependencies(plan, round_, state)
                self.assertEqual(item['blocked_by'], [])
                candidates = item['inherit_best']['candidates']
                pairs = [(parent, runner.latest_execution(state, round_['id'], parent))
                         for parent in round_['experiments'] if parent['id'] in candidates]
                winner, record = max(pairs, key=lambda p: runner.validation_key(p[1]['training_record']))
                self.assertEqual(item['dependency_selection']['winner_experiment_id'], winner['id'])
                expected = runner.normalize_config(runner.merge_config(winner['config'], item['overrides']), 'Meal-H')
                self.assertEqual(item['config'], expected)
                self.assertEqual(item['config']['latent_rebuild']['latent_diffusion_set_loss_weight'], .05)
                score = .2
                if item['id'] == 'l2_5e6':
                    score = .4
                elif item['id'] == 'beta_end_0008':
                    score = .41
                elif item['id'] == 'fusion_ub0_ui040':
                    score = .45
                elif item['id'] == 'bi_layer_mixed':
                    score = .46
                elif item['id'] == 'ui_layer_deep':
                    score = .47
                elif item['id'] == 'ui_noise_0025':
                    score = .48
                ndcg = score / 2
                if item['id'] == 'l2_1e6':
                    score, ndcg = baseline_score, .24390730448890638
                if item['id'].startswith('beta_end_'):
                    self.assertEqual(item['config']['fusion_weights']['modal_weight'][0], .05)
                if item['id'].startswith(('bi_', 'ui_', 'temp_')):
                    self.assertEqual(item['config']['fusion_weights']['modal_weight'], [0., .4, .6])
                state['executions'].append({
                    'round_id': round_['id'], 'experiment_id': item['id'], 'execution_id': 'synthetic_' + item['id'],
                    'fingerprint': item['fingerprint'], 'status': 'SUCCESS',
                    'training_record': {'result': {'metrics': {
                        'val': {'recall': {'20': score}, 'ndcg': {'20': ndcg}},
                        'test': {'recall': {'20': 1-score}, 'ndcg': {'20': 1-ndcg}},
                    }}, 'artifacts': {'checkpoint_model': 'synthetic.pt'}},
                })
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                plan['work_dir'] = Path(temp) / 'work'
                plan['result_file'] = Path(temp) / 'results.txt'
                summary = runner.write_round_summary(plan, round_, state)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['training_planned'], 34)
        self.assertEqual(summary['best']['experiment_id'], 'ui_noise_0025')
        self.assertEqual(summary['best']['config']['l2_regs'], [5e-6])
        self.assertEqual(summary['best']['config']['latent_rebuild']['latent_diffusion_beta_end'], .008)
        self.assertEqual(summary['stages'][4]['best']['experiment_id'], 'bi_layer_mixed')
        self.assertEqual(summary['stages'][-1]['best']['experiment_id'], 'ui_noise_0025')

    def test_text_reference_migration_preserves_content_checks(self):
        plan, round_ = self.fixture()
        item = round_['experiments'][0]
        record = runner.read_result_reference(item, round_, plan)
        self.assertTrue(item['reference']['allow_data_path_relocation'])
        self.assertEqual(record['execution_id'], '20261026T172912_81c8dcda')
        self.assertEqual(runner.validation_key(record['training_record']),
                         (.3785844258140611, .24390730448890638))
        # Binary pretrained inputs still require exact SHA256 equality.
        changed = copy.deepcopy(round_)
        changed['data_files'][runner.EMBEDDING_KEYS[0]] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'pretrained embedding mismatch'):
            runner.read_result_reference(changed['experiments'][0], changed, plan)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.py'
            path.write_bytes(b'original\n')
            historical = runner.hashlib.sha256(b'original\r\n').hexdigest()
            self.assertIsNotNone(runner.reference_hash_match(plan, path, historical, runner.file_hash(path)))
            path.write_bytes(b'actual content change\n')
            with self.assertRaisesRegex(ValueError, 'content mismatch beyond LF/CRLF'):
                runner.reference_hash_match(plan, path, historical, runner.file_hash(path))


if __name__ == '__main__':
    unittest.main()
