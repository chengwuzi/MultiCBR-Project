"""Historical TXT references work without old history/checkpoints."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiments as runner


class ResultReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = runner.load_plan(runner.MODEL_DIR / '搜参配置.yaml')
        cls.text = (runner.MODEL_DIR / '搜参结果.txt').read_text(encoding='utf-8')
        start = cls.text.index('EXECUTION_ID: 20261025T121254_2673b424\n')
        end = cls.text.index('END_EXECUTION: 20261025T121254_2673b424', start)
        cls.block = cls.text[start:end] + 'END_EXECUTION: 20261025T121254_2673b424\n'

    def fixture(self, directory):
        plan = copy.deepcopy(self.plan)
        round_ = next(r for r in plan['rounds'] if r['id'] == 'youshu_round_03_k')
        path = Path(directory) / 'results.txt'
        path.write_text(self.block, encoding='utf-8')
        plan['work_dir'] = Path(directory) / 'work'
        plan['result_file'] = path
        round_['experiments'][0]['reference']['result_file'] = str(path)
        return plan, round_, path

    def test_no_history_or_artifacts_needed_and_validation_keeps_full_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, round_, path = self.fixture(directory)
            before = path.read_bytes()
            state = runner.empty_state()
            selected = runner.select_experiments(plan, round_, state, 'resume', [])
            item = round_['experiments'][0]
            record = runner.latest_execution(state, round_['id'], item)
            self.assertTrue(runner.check_success(record, item, round_, plan))
            self.assertEqual(runner.validation_key(record['training_record']),
                             (0.2693115420017584, 0.13257031635724037))
            self.assertEqual(record['training_record']['result']['best_epoch'], 79)
            self.assertEqual([x['config']['rebuild_k'] for x in selected], [3, 5, 7, 9, 11])
            self.assertEqual(state['executions'], [])
            self.assertFalse(plan['work_dir'].exists())
            self.assertEqual(path.read_bytes(), before)
            evidence = item['dependency_selection']['reference_evidence']
            self.assertFalse(evidence['artifacts_verified'])
            self.assertEqual(evidence['kind'], 'result_text')
            # Appending a later report must not invalidate earlier child fingerprints.
            fingerprints = [x['fingerprint'] for x in selected]
            with path.open('a', encoding='utf-8') as stream:
                stream.write('\nUnrelated later experiment\n')
            runner.resolve_dependencies(plan, round_, state)
            self.assertEqual(fingerprints, [x['fingerprint'] for x in selected])
            tampered = copy.deepcopy(record)
            tampered['training_record']['result']['metrics']['val']['recall']['20'] = .9
            self.assertFalse(runner.check_reference(tampered, item, round_, plan))

    def test_incomplete_duplicate_wrong_config_data_and_rounded_metrics_rejected(self):
        changes = [
            lambda s: s.replace('END_EXECUTION:', 'INCOMPLETE:'),
            lambda s: s + s,
            lambda s: s.replace('状态: SUCCESS', '状态: FAILED'),
            lambda s: s.replace('轮次: youshu_round_02', '轮次: wrong_round'),
            lambda s: s.replace('lambda_0: 2000.0', 'lambda_0: 1000.0'),
            lambda s: s.replace('seed: 0', 'seed: 1'),
            lambda s: s.replace('epochs: 90', 'epochs: 40'),
            lambda s: s.replace('efc0c8a83909e2ba8aadc6c06fa3a55b5f81cc2244593eef54266ddf83237379', '0' * 64),
            lambda s: s.replace('4f370f871ed707e9505430b8ca12462f2faee526b98f7539fde7e4d4c71ccb8f', '0' * 64),
            lambda s: s.replace('验证选模指标:', 'MISSING_SELECTION:'),
            lambda s: s.replace('0.2693115420017584', 'NaN'),
            lambda s: s.replace('0.2693115420017584', '0.5'),
            lambda s: s.replace('best_epoch: 79', 'best_epoch: 78'),
            lambda s: s.replace('val TOP 40: Recall=0.36338330', 'val TOP 40: Recall=nan'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, change in enumerate(changes):
                with self.subTest(index=index):
                    plan, round_, path = self.fixture(directory)
                    path.write_text(change(self.block), encoding='utf-8')
                    with self.assertRaises((ValueError, KeyError)):
                        runner.resolve_dependencies(plan, round_, runner.empty_state())

    def test_baseline_can_win_summary_without_claiming_an_available_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, round_, path = self.fixture(directory)
            state = runner.empty_state()
            real_check = runner.check_success
            def check(record, item, rd, p):
                if item.get('reference'):
                    return real_check(record, item, rd, p)
                return record is not None and record['status'] == 'SUCCESS'
            with patch.object(runner, 'check_success', side_effect=check), contextlib.redirect_stdout(io.StringIO()):
                runner.resolve_dependencies(plan, round_, state)
                for item in round_['experiments'][1:]:
                    baseline = round_['experiments'][0]['_reference_record']
                    training = copy.deepcopy(baseline['training_record'])
                    training['result']['metrics']['val']['recall']['20'] = .1
                    training['result']['metrics']['test']['recall']['20'] = .99
                    state['executions'].append({'round_id': round_['id'], 'experiment_id': item['id'],
                        'execution_id': item['id'], 'fingerprint': item['fingerprint'],
                        'status': 'SUCCESS', 'training_record': training})
                summary = runner.write_round_summary(plan, round_, state)
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['training_planned'], 5)
            self.assertEqual(summary['best']['experiment_id'], 'baseline_k1')
            self.assertIsNone(summary['best']['checkpoint'])
            self.assertFalse(summary['best']['reference_evidence']['artifacts_verified'])
            self.assertEqual(summary['best']['validation']['recall@20'], 0.2693115420017584)


if __name__ == '__main__':
    unittest.main()
