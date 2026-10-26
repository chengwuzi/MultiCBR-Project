"""Only the new plan, historical baseline and stage inheritance are checked."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiments as runner

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dcbr_coverage_training', HERE / 'train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class CoverageTests(unittest.TestCase):
    def load(self):
        plan = runner.load_plan(HERE / '搜参配置.yaml')
        round_ = next(r for r in plan['rounds'] if r['id'] == 'youshu_round_04_coverage')
        return plan, round_

    def test_baseline_grids_and_actual_training_validation(self):
        plan, round_ = self.load()
        state = runner.empty_state()
        selected = runner.select_experiments(plan, round_, state, 'resume', [])
        self.assertEqual(len(selected), 49)
        baseline = round_['experiments'][0]
        record = runner.latest_execution(state, round_['id'], baseline)
        self.assertTrue(runner.check_success(record, baseline, round_, plan))
        self.assertEqual(runner.validation_key(record['training_record']),
                         (.27404046711304497, .13345297207176463))
        self.assertEqual(record['execution_id'], '20261025T115614_4738c53b')
        self.assertEqual(baseline['config'], runner.read_yaml(HERE / 'config.yaml')['Youshu'] |
                         {'data_path': str(runner.PROJECT_ROOT / 'datasets')})
        expected = {
            'lambda_1': [.001, .005, .01, .03, .05, .1, .2, .4],
            'omega': [0., .1, .2, .4, .6, .8, 1.],
            'lambda_2': [1e-6, 1e-7],
            'upsilon_UI': [.025, .05, .1, .2],
            'upsilon_BI': [0., .05, .2, .3],
            'upsilon_UB': [0., .01, .1, .2],
            'xi_UI': [[.6,.3,.1], [.3,.4,.3], [.2,.3,.5], [1/3]*3],
            'xi_BI': [[.6,.3,.1], [.3,.4,.3], [.2,.3,.5], [1/3]*3],
            'xi_UB': [[.5,.5,0.], [.3,.4,.3], [.2,.3,.5], [1/3]*3],
            'steps': [2, 10, 20, 50],
            'noise_scale': [.1, .3, .5, 2.],
        }
        actual = {}
        for item in selected:
            self.assertEqual(len(item['overrides']), 1)
            key, value = next(iter(item['overrides'].items()))
            actual.setdefault(key, []).append(value)
            conf = item['template_config']
            training.validate_config(dict(conf, dataset='Youshu', model='DCBR'))
            self.assertEqual((conf['seed'], conf['epochs']), (0, 90))
            if key.startswith('xi_'):
                self.assertAlmostEqual(sum(value), 1)
        self.assertEqual(actual, expected)
        self.assertEqual(len(round_['stages']), 11)

    def test_complete_chain_inherits_only_previous_stages_and_keeps_controls(self):
        plan, round_ = self.load()
        state = runner.empty_state()
        real_check = runner.check_success
        def valid(record, item, current_round, current_plan):
            if item.get('reference'):
                return real_check(record, item, current_round, current_plan)
            return bool(record and record.get('_synthetic'))
        with patch.object(runner, 'check_success', side_effect=valid):
            pool = ['baseline_best']
            offset = 1
            for stage_index, stage in enumerate(round_['stages']):
                ids = stage['candidates'][len(pool):]
                items = round_['experiments'][offset:offset+len(ids)]
                self.assertEqual([e['id'] for e in items], ids)
                for trial_index, item in enumerate(items):
                    self.assertEqual(item['inherit_best']['candidates'], pool)
                    runner.resolve_dependencies(plan, round_, state)
                    winner = item['dependency_selection']['winner_experiment_id']
                    parent = next(e for e in round_['experiments'] if e['id'] == winner)
                    expected = runner.merge_config(parent['config'], item['overrides'])
                    self.assertEqual(item['config'], expected)
                    training.validate_config(dict(expected, dataset='Youshu', model='DCBR'))
                    parent_record = runner.latest_execution(state, round_['id'], parent)
                    prior_score = runner.validation_key(parent_record['training_record'])[0]
                    # Some stages have no improvement, and must retain the control.
                    score = prior_score - .001 if stage_index % 3 == 0 else prior_score + trial_index * .0001
                    state['executions'].append({
                        'execution_id': 'synthetic_' + item['id'], 'round_id': round_['id'],
                        'experiment_id': item['id'], 'fingerprint': item['fingerprint'],
                        'status': 'SUCCESS', '_synthetic': True,
                        'training_record': {'result': {'metrics': {
                            'val': {'recall': {'20': score}, 'ndcg': {'20': .13}},
                            'test': {'recall': {'20': 1 - score}}}},
                            'artifacts': {'checkpoint_model': 'synthetic.pt'}}})
                pool.extend(ids)
                offset += len(ids)
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                plan['work_dir'] = Path(temp) / 'work'
                plan['result_file'] = Path(temp) / 'results.txt'
                summary = runner.write_round_summary(plan, round_, state)
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['planned'], 50)
            self.assertEqual(summary['training_planned'], 49)
            self.assertEqual(summary['reference_count'], 1)
            scores = [s['best']['validation']['recall@20'] for s in summary['stages']]
            self.assertEqual(scores, sorted(scores))
            self.assertEqual(summary['stages'][0]['best']['experiment_id'], 'baseline_best')


if __name__ == '__main__':
    unittest.main()
