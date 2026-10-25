"""Reviewed K plans, graph semantics, reference migration and ordered recovery."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
import yaml

import test_reproduction as reference
import run_k_search as pipeline

HERE = Path(__file__).resolve().parent
runner = pipeline.runner
port = reference.port


class KSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference.ReproductionTests.setUpClass()

    def test_reviewed_grids_budgets_reference_and_all_six_stages(self):
        plan = runner.load_plan(HERE / '搜参配置.yaml')
        y, n = pipeline.reviewed_rounds(plan)
        self.assertEqual(len(y['experiments']), 6)
        self.assertEqual(len(n['experiments']), 24)
        ref = y['experiments'][0]['reference']
        old = next(e for e in plan['document']['rounds'][1]['experiments'] if e['id'] == 'tau_025')
        self.assertEqual(ref['execution_id'], '20261025T121254_2673b424')
        self.assertEqual(ref['fingerprint'], old['fingerprint'])
        self.assertTrue(ref['allow_data_path_relocation'])
        self.assertEqual([e['template_config']['rebuild_k'] for e in y['experiments']], [1, 3, 5, 7, 9, 11])
        for e in y['experiments']:
            self.assertEqual(e['template_config']['epochs'], 90)
            self.assertEqual(e['template_config']['seed'], 0)
            if not e['reference']:
                self.assertEqual(e['inherit_best']['candidates'], ['baseline_k1'])
                self.assertEqual(set(e['overrides']), {'rebuild_k'})
        groups = [('rebuild_k', [3, 5, 7, 9, 11]), ('lambda_1', [.005, .01, .03, .05]),
                  ('lambda_0', [100, 500, 2000, 10000]), ('omega', [.2, .3, .5, .6]),
                  ('tau', [.1, .15, .25, .3]), ('lambda_2', [1e-6, 1e-7])]
        self.assertEqual(n['experiments'][0]['template_config']['rebuild_k'], 1)
        start = 1
        for stage, (key, values) in zip(n['stages'], groups):
            end = start + len(values)
            self.assertEqual(stage['id'], key)
            self.assertEqual(stage['candidates'], [e['id'] for e in n['experiments'][:end]])
            self.assertEqual([e['overrides'] for e in n['experiments'][start:end]], [{key: v} for v in values])
            self.assertTrue(all(e['inherit_best']['candidates'] == [p['id'] for p in n['experiments'][:start]]
                                for e in n['experiments'][start:end]))
            start = end
        self.assertEqual(start, 24)
        self.assertTrue(all(e['template_config']['epochs'] == 40 and e['template_config']['seed'] == 4096
                            for e in n['experiments']))
        self.assertFalse(any(e['reference'] for e in n['experiments']))
        self.assertEqual(len(runner.select_experiments(plan, y, runner.empty_state(), 'all', [])), 5)

    def test_k_is_carried_to_every_later_stage_and_test_scores_do_not_select(self):
        plan = runner.load_plan(HERE / '搜参配置.yaml')
        round_ = pipeline.reviewed_rounds(plan)[1]
        items = round_['experiments']
        state = runner.empty_state()
        # K=7 wins; all later candidates tie exactly and retain that control.
        scores = [(.1, .1), (.2, .1), (.2, .2), (.3, .1), (.25, .9), (.2, .9)] + [(.3, .1)]*18
        valid = lambda record, *args: record is not None and record['status'] == 'SUCCESS'
        with patch.object(runner, 'check_success', side_effect=valid):
            for index, (item, (recall, ndcg)) in enumerate(zip(items, scores)):
                runner.resolve_dependencies(plan, round_, state)
                if index >= 6:
                    self.assertEqual(item['config']['rebuild_k'], 7)
                    self.assertEqual(item['dependency_selection']['winner_experiment_id'], 'k_07')
                state['executions'].append({'round_id': round_['id'], 'experiment_id': item['id'],
                    'execution_id': str(index), 'fingerprint': item['fingerprint'], 'status': 'SUCCESS',
                    'training_record': {'result': {'metrics': {
                        'val': {'recall': {'20': recall}, 'ndcg': {'20': ndcg}},
                        'test': {'recall': {'20': 1-recall}, 'ndcg': {'20': 1-ndcg}}}}}})
            state['executions'][2]['status'] = 'FAILED'
            runner.resolve_dependencies(plan, round_, state)
            self.assertTrue(all(e['config'] is None and 'k_05' in e['blocked_by'] for e in items[6:]))

    def test_rebuilt_graph_has_k_edges_per_user_including_zero_history_user(self):
        graph = sp.csr_matrix(([1., 1., 1.], ([0, 1, 2], [0, 1, 2])), shape=(4, 12))
        loader = torch.utils.data.DataLoader(port.DiffusionDataset(graph), batch_size=2)

        class RankedDiffusion:
            def p_sample(self, model, interaction, steps, sampling_noise):
                return torch.arange(12, dtype=torch.float32).expand_as(interaction)

        conf = dict(device='cpu', num_users=4, num_bundles=12, sampling_steps=0, sampling_noise=False)
        for k in (1, 3, 5, 7, 9, 11, 12):
            with self.subTest(k=k):
                conf['rebuild_k'] = k
                actual = port.rebuild_graph(None, RankedDiffusion(), loader, conf).to_dense()
                rows, columns = np.repeat(np.arange(4), k), np.tile(np.arange(11, 11-k, -1), 4)
                ub = sp.coo_matrix((np.ones(4*k), (rows, columns)), shape=(4, 12), dtype=np.float32)
                a = sp.bmat([[sp.csr_matrix((4, 4)), ub], [ub.T, sp.csr_matrix((12, 12))]]) + sp.eye(16)
                degree = np.power(np.asarray(a.sum(axis=1)).ravel(), -.5)
                expected = torch.FloatTensor((sp.diags(degree) @ a @ sp.diags(degree)).toarray())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.all(torch.count_nonzero(actual[:4, 4:], dim=1) == k))
                self.assertTrue(torch.all(torch.diag(actual) > 0))

    def preflight_in_process(self, plan, round_, experiments):
        # Validate the same model configuration without repeating Python/torch
        # startup for every tiny trial. Production still uses isolated workers.
        for e in experiments:
            conf = copy.deepcopy(e['config'] if e['config'] is not None else e['template_config'])
            conf['dataset'] = round_['dataset']
            port.validate_config(conf)

    def train_in_process(self, command, attempt, record, *args):
        with Path(attempt['stdout']).open('w', encoding='utf-8') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                return port.main(command[5:])

    def make_plan_with_historical_control(self, root):
        fixture = reference.ReproductionTests()
        fixture.base = reference.ReproductionTests.base
        fixture_path = fixture.make_tiny_data(root)
        tiny = runner.read_yaml(fixture_path)['iFashion']
        (root / 'data/iFashion').rename(root / 'data/Youshu')
        (root / 'data/Youshu/iFashion_data_size.txt').rename(root / 'data/Youshu/Youshu_data_size.txt')
        (root / 'data/Youshu/Youshu_data_size.txt').write_text('6\t100\t8\n')
        shutil.copytree(root / 'data/Youshu', root / 'data/NetEase')
        (root / 'data/NetEase/Youshu_data_size.txt').rename(root / 'data/NetEase/NetEase_data_size.txt')
        production = runner.read_yaml(HERE / '搜参配置.yaml')
        y, n = copy.deepcopy(production['rounds'][-2:])
        bases = runner.read_yaml(HERE / 'config.yaml')
        reduced = {k: tiny[k] for k in ('data_path', 'batch_size_train', 'batch_size_test',
                   'epochs', 'test_interval', 'steps', 'dims', 'embedding_size')}
        for r in (y, n):
            r['overrides'].update(reduced)
            bases[r['dataset']].update(reduced)
        config = root / 'config.yaml'
        config.write_text(yaml.safe_dump(bases), encoding='utf-8')
        historical = {'id': 'historical', 'dataset': 'Youshu', 'note': 'tiny historical K=1',
                      'overrides': copy.deepcopy(y['overrides']),
                      'experiments': [{'id': 'control', 'note': 'tiny K=1 baseline', 'overrides': {}}]}
        document = {'version': 1, 'base_config': str(config), 'work_dir': str(root / 'work'),
                    'result_file': str(root / 'search.txt'), 'defaults': dict(production['defaults']),
                    'rounds': [historical]}
        document['defaults'].update(device='cpu', max_attempts=1, retry_delay_seconds=0)
        path = root / 'plan.yaml'
        path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding='utf-8')
        with patch.object(runner, 'preflight', side_effect=self.preflight_in_process), \
             patch.object(runner, 'run_worker', side_effect=self.train_in_process), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(path, 'historical'), 0)
        original = copy.deepcopy(runner.load_state(root / 'work/history.json')['executions'][0])
        y['experiments'][0]['reference'].update({k: original[k] for k in
                                               ('round_id', 'experiment_id', 'execution_id', 'fingerprint')})
        # The current dataset path moves; the original result/config/artifacts
        # remain immutable and available at their original artifact locations.
        shutil.copytree(root / 'data/Youshu', root / 'relocated/Youshu')
        y['overrides']['data_path'] = str(root / 'relocated')
        document = runner.read_yaml(path)
        document['rounds'].extend([y, n])
        path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding='utf-8')
        return path, original

    def test_reference_relocation_requires_opt_in_matching_data_and_intact_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-k-ref-') as temp:
            root = Path(temp)
            path, original = self.make_plan_with_historical_control(root)
            plan = runner.load_plan(path)
            y = pipeline.reviewed_rounds(plan)[0]
            item = y['experiments'][0]
            self.assertTrue(runner.check_reference(original, item, y, plan))
            strict = copy.deepcopy(item)
            strict['reference']['allow_data_path_relocation'] = False
            self.assertFalse(runner.check_reference(original, strict, y, plan))
            changed = copy.deepcopy(item)
            changed['template_config']['lambda_0'] *= 2
            self.assertFalse(runner.check_reference(original, changed, y, plan))
            wrong_data = copy.deepcopy(y)
            wrong_data['data_files']['user_item.txt'] = 'changed'
            self.assertFalse(runner.check_reference(original, item, wrong_data, plan))
            artifact = Path(original['training_record']['artifacts']['checkpoint_model'])
            content = artifact.read_bytes()
            artifact.write_bytes(content + b'corrupt')
            self.assertFalse(runner.check_reference(original, item, y, plan))
            artifact.write_bytes(content)
            self.assertTrue(runner.check_reference(original, item, y, plan))
            document = runner.read_yaml(path)
            document['rounds'][1]['experiments'][0]['reference']['allow_data_path_relocation'] = 1
            path.write_text(yaml.safe_dump(document), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must be boolean'):
                runner.load_plan(path)

    def test_ordered_29_trial_pipeline_interrupt_failure_resume_and_summaries(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-k-chain-') as temp:
            root = Path(temp)
            path, original = self.make_plan_with_historical_control(root)
            controls = {'interrupt': True, 'fail': True}
            def worker(command, attempt, record, *args):
                if record['round_id'] == pipeline.ROUNDS[0] and record['experiment_id'] == 'k_05' and controls['interrupt']:
                    controls['interrupt'] = False
                    raise KeyboardInterrupt
                if record['round_id'] == pipeline.ROUNDS[1] and record['experiment_id'] == 'k_05' and controls['fail']:
                    controls['fail'] = False
                    return 8
                return self.train_in_process(command, attempt, record, *args)
            with patch.object(runner, 'preflight', side_effect=self.preflight_in_process), \
                 patch.object(runner, 'run_worker', side_effect=worker), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.run(path)
                state = runner.load_state(root / 'work/history.json')
                self.assertFalse(any(r['dataset'] == 'NetEase' for r in state['executions']))
                self.assertEqual(pipeline.run(path), 1)
                state = runner.load_state(root / 'work/history.json')
                net = [r for r in state['executions'] if r['dataset'] == 'NetEase']
                self.assertTrue(all(r['experiment_id'] == 'initial' or r['experiment_id'].startswith('k_') for r in net))
                self.assertEqual(pipeline.run(path), 0)
                state = runner.load_state(root / 'work/history.json')
                self.assertEqual(state['executions'][0], original)
                successes = [r for r in state['executions'][1:] if r['status'] == 'SUCCESS']
                self.assertEqual(len(successes), 29)
                self.assertEqual([r['dataset'] for r in successes], ['Youshu']*5 + ['NetEase']*24)
                self.assertTrue(all(r['config']['rebuild_k'] in (3, 5, 7, 9, 11) for r in successes[:5]))
                plan = runner.load_plan(path)
                for round_ in pipeline.reviewed_rounds(plan):
                    runner.resolve_dependencies(plan, round_, state)
                    for item in round_['experiments']:
                        if item['inherit_best']:
                            record = runner.latest_execution(state, round_['id'], item)
                            parent = next(r for r in state['executions'] if r['execution_id'] ==
                                          record['dependency_selection']['winner_execution_id'])
                            expected = dict(parent['config'], **item['overrides'])
                            if round_['dataset'] == 'Youshu':
                                expected['data_path'] = str(root / 'relocated')
                            self.assertEqual(record['config'], expected)
                    summary = json.loads((root / 'work' / round_['dataset'] / round_['id'] / 'round_summary.json').read_text(encoding='utf-8'))
                    self.assertTrue(summary['complete'])
                    self.assertEqual(summary['configuration_selection_rule'], runner.CROSS_RULE)
                    self.assertEqual(len(summary['stages']), 1 if round_['dataset'] == 'Youshu' else 6)
                    self.assertEqual(summary['best']['execution_id'], summary['stages'][-1]['best']['execution_id'])
                report = (root / 'search.txt').read_bytes()
                count = len(state['executions'])
                self.assertEqual(pipeline.run(path), 0)
                self.assertEqual(len(runner.load_state(root / 'work/history.json')['executions']), count)
                self.assertEqual((root / 'search.txt').read_bytes(), report)

    def test_missing_reference_stops_before_training_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory(prefix='dcbr-k-missing-') as temp:
            root = Path(temp)
            path, original = self.make_plan_with_historical_control(root)
            document = runner.read_yaml(path)
            document['rounds'][1]['experiments'][0]['reference']['execution_id'] = 'missing_execution'
            path.write_text(yaml.safe_dump(document), encoding='utf-8')
            before = path.read_bytes(), (root / 'search.txt').read_bytes()
            with patch.object(runner, 'preflight', side_effect=self.preflight_in_process), \
                 patch.object(runner, 'run_worker') as worker, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.run(path, dry_run=True), 0)
                with self.assertRaisesRegex(ValueError, 'Historical references missing or invalid'):
                    pipeline.run(path)
                worker.assert_not_called()
            self.assertEqual(before, (path.read_bytes(), (root / 'search.txt').read_bytes()))
            self.assertEqual(runner.load_state(root / 'work/history.json')['executions'], [original])
            # Even an erroneous successful return must not bypass the explicit
            # verification gate and launch NetEase with incomplete Youshu data.
            with patch.object(runner, 'run', return_value=0) as run_round, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'incomplete or artifacts'):
                    pipeline.run(path)
                self.assertEqual(run_round.call_count, 1)
                self.assertEqual(run_round.call_args.args[1], pipeline.ROUNDS[0])

    def test_entrypoint_does_not_import_torch(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; sys.path.insert(0, sys.argv[1]); import run_k_search; assert 'torch' not in sys.modules", str(HERE)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
