# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""CPU-only regression tests: real HTTP/process lifecycle, never import vLLM."""
import contextlib
import importlib.util
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET
from types import SimpleNamespace

SMOKE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('iluvatar_model_smoke_runner', SMOKE / 'runner.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.manifest, self.models, self.cases = smoke.load_plan(
            SMOKE / 'suite.json', 'all', SMOKE / 'cases.json')
        self.model = copy.deepcopy(next(m for m in self.models if m['id'] == 'deepseek-v4-flash'))
        self.model['ready_timeout_seconds'] = 5
        self.model['request_timeout_seconds'] = 1
        self.cache = self.root / 'cache'
        self.path = self.cache / self.model['cache_relpath']
        self.path.mkdir(parents=True)
        content = b'{"model_type":"test"}'
        (self.path / 'config.json').write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        self.model.update(config_sha256=digest, cache_total_bytes=len(content),
                          cache_files=[{'path': 'config.json', 'size': len(content), 'sha256': digest}])
        self.marker = {k: self.model[k] for k in ('owner', 'slug', 'platform_version', 'revision')}
        self.marker.update(status='READY', manifest_sha256=self.model['cache_manifest_sha256'],
                           file_count=1, total_bytes=len(content))
        self.save_marker()

    def save_marker(self):
        smoke.write_json(self.path / '.ci-modelhub-ready.json', self.marker)


    def test_owner_rename_accepts_only_the_exact_original_ready_receipt(self):
        source = {key: self.model[key] for key in
                  ('name', 'slug', 'owner', 'platform_version', 'revision', 'storage_relpath')}
        source.update(owner='iluvatar-aac-llmfp', files=self.model['cache_files'],
                      bytes=self.model['cache_total_bytes'])
        self.marker.update(owner=source['owner'], manifest_sha256=hashlib.sha256(
            json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        self.save_marker()
        with patch.object(smoke, 'local_filesystem'):
            self.assertEqual(smoke.preflight(self.model, self.cache), self.path)
            original = dict(self.marker)
            for change in ({'owner': 'another-owner'}, {'manifest_sha256': '0' * 64},
                           {'revision': '0' * 64}, {'file_count': 2}, {'total_bytes': 0}):
                self.marker = dict(original, **change)
                self.save_marker()
                with self.assertRaises(ValueError):
                    smoke.preflight(self.model, self.cache)

    def test_suite_is_valid_and_resource_capped(self):
        self.assertTrue(self.models)
        self.assertEqual(len(self.models), len({m['id'] for m in self.models}))
        self.assertTrue(all(m['logical_gpus'] <= 16 for m in self.models))

    def test_profiles_can_be_added_and_removed_without_runner_changes(self):
        folder = self.root / 'editable-suite'
        (folder / 'models').mkdir(parents=True)
        model = smoke.read_json(SMOKE / 'models' / (self.models[0]['id'] + '.json'))
        (folder / 'weights').mkdir()
        smoke.write_json(folder / model['weight_manifest'], smoke.read_json(SMOKE / model['weight_manifest']))
        added = copy.deepcopy(model)
        added['id'] = 'new-model-profile'
        for item in (model, added):
            smoke.write_json(folder / 'models' / (item['id'] + '.json'), item)
        suite = folder / 'suite.json'
        smoke.write_json(suite, {'models': [model['id'], added['id']]})
        _, selected, _ = smoke.load_plan(suite, 'all', SMOKE / 'cases.json')
        self.assertEqual([item['id'] for item in selected], [model['id'], added['id']])
        smoke.write_json(suite, {'models': [added['id']]})
        _, selected, _ = smoke.load_plan(suite, 'all', SMOKE / 'cases.json')
        self.assertEqual([item['id'] for item in selected], [added['id']])

    def split_suite(self):
        folder = self.root / 'split-suite'
        (folder / 'models').mkdir(parents=True)
        (folder / 'weights').mkdir()
        profile = smoke.read_json(SMOKE / 'models' / (self.models[0]['id'] + '.json'))
        model_path = folder / 'models' / (profile['id'] + '.json')
        weight_path = folder / profile['weight_manifest']
        smoke.write_json(model_path, profile)
        smoke.write_json(weight_path, smoke.read_json(SMOKE / profile['weight_manifest']))
        smoke.write_json(folder / 'suite.json', {'models': [profile['id']]})
        return folder / 'suite.json', model_path, weight_path

    def test_split_manifest_preserves_launch_and_pinned_weight_data(self):
        suite, model_path, _ = self.split_suite()
        self.assertEqual(smoke.load_suite(suite)['models'], [self.models[0]])
        profile = smoke.read_json(model_path)
        self.assertFalse(any(key.startswith('cache_') for key in profile))
        profile['serve_args'].extend(['--seed', '17'])
        smoke.write_json(model_path, profile)
        loaded = smoke.load_suite(suite)['models'][0]
        self.assertEqual(loaded['serve_args'], profile['serve_args'])
        self.assertEqual(loaded['cache_files'], self.models[0]['cache_files'])

    def test_split_manifest_rejects_missing_unsafe_and_linked_paths(self):
        suite, model_path, weight_path = self.split_suite()
        profile = smoke.read_json(model_path)
        for reference in ('../escape.json', '/tmp/outside.json', 'models/other.json',
                          'weights/../outside.json', 'weights/missing.json'):
            smoke.write_json(model_path, dict(profile, weight_manifest=reference))
            with self.assertRaises(ValueError):
                smoke.load_suite(suite)
        smoke.write_json(model_path, profile)
        external = self.root / 'outside.json'
        weight_path.rename(external)
        weight_path.symlink_to(external)
        with self.assertRaises(ValueError):
            smoke.load_suite(suite)

    def test_split_manifest_rejects_wrong_version_and_launch_settings(self):
        suite, _, weight_path = self.split_suite()
        original = smoke.read_json(weight_path)
        for key in ('name', 'owner', 'slug', 'platform_version', 'revision'):
            smoke.write_json(weight_path, dict(original, **{key: 'wrong'}))
            with self.assertRaises(ValueError):
                smoke.load_suite(suite)
        smoke.write_json(weight_path, dict(original, serve_args=['--enforce-eager']))
        with self.assertRaises(ValueError):
            smoke.load_suite(suite)

    def test_bad_selection_never_runs(self):
        for selected in ('', 'unknown', 'deepseek-r1,deepseek-r1'):
            with self.assertRaises(ValueError):
                smoke.load_plan(SMOKE / 'suite.json', selected, SMOKE / 'cases.json')

    def test_manifest_rejects_unsafe_or_inconsistent_config(self):
        mutations = [lambda m: m.update(cache_relpath='../escape'),
                     lambda m: m.update(logical_gpus=32),
                     lambda m: m['parallel'].update(tp=32),
                     lambda m: m.update(check_speculative=not m["check_speculative"]),
                     lambda m: m['environment'].update(CUDA_VISIBLE_DEVICES='0'),
                     lambda m: m['cache_files'][0].update(path='../config.json')]
        for mutate in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest['models'][0])
            path = self.root / 'bad.json'
            (self.root / 'models').mkdir(exist_ok=True)
            for item in manifest['models']:
                smoke.write_json(self.root / 'models' / (item['id'] + '.json'), item)
            smoke.write_json(path, dict(manifest, models=[item['id'] for item in manifest['models']]))
            with self.assertRaises(ValueError):
                smoke.load_plan(path, 'all', SMOKE / 'cases.json')

    def test_exact_answers_and_reasoning(self):
        case = self.cases[0]
        for text in ('FINAL: 42', '<think>wrong guess 43</think>\nFINAL: 42', '42.0', 'FINAL: \\boxed{42}'):
            self.assertTrue(smoke.judge(text, case), text)
        for text in ('142', '42 or 43', 'not 42', '<think>42', '', None,
                     'FINAL: 42\nActually 43', 'The answer includes 42 but is 43'):
            self.assertFalse(smoke.judge(text, case), text)

    def test_json_and_text_do_not_accept_substrings(self):
        self.assertTrue(smoke.judge('FINAL: {"age":27,"name":"Alice"}', self.cases[-1]))
        for text in ('{"age":true,"name":"Alice"}', '{"age":27,"name":"Alice","extra":1}', 'not JSON'):
            self.assertFalse(smoke.judge(text, self.cases[-1]))
        self.assertFalse(smoke.judge('Hangzhou 杭州 Shanghai', self.cases[7]))
        self.assertTrue(smoke.judge('FINAL: 杭州。', self.cases[7]))

    def test_terminal_final_marker_accepts_inline_answer_without_weakening_value_check(self):
        city = self.cases[7]
        self.assertTrue(smoke.judge('The city in the record is 杭州 (Hangzhou). FINAL: 杭州', city))
        self.assertTrue(smoke.judge('Computed result. FINAL: 42', self.cases[0]))
        self.assertTrue(smoke.judge('Extracted fields. FINAL: {"name":"Alice","age":27}', self.cases[-1]))
        for text in ('NOTFINAL: 杭州', 'FINAL: 杭州 or 上海',
                     'FINAL: 杭州\nActually 上海', '<think>FINAL: 杭州',
                     'FINAL: 杭州 FINAL: 上海', 'The answer mentions 杭州 but is 上海'):
            self.assertFalse(smoke.judge(text, city), text)
        self.assertFalse(smoke.judge('Result. FINAL: 142', self.cases[0]))
        self.assertFalse(smoke.judge('Result. FINAL: {"name":"Alice","age":27,"city":"Rome"}', self.cases[-1]))

    def test_reasoning_only_and_truncated_correct_answers_still_fail(self):
        for message, finish in (({'content': None, 'reasoning': 'FINAL: 杭州'}, 'stop'),
                                ({'content': '', 'reasoning_content': 'FINAL: 杭州'}, 'stop'),
                                ({'content': 'FINAL: 杭州'}, 'length')):
            response = {'model': 'test', 'choices': [{'message': message, 'finish_reason': finish}]}
            with patch.object(smoke, 'http', return_value=json.dumps(response)):
                row = smoke.request_case('http://fake', 'test', self.model, self.cases[7], 'serial')
            self.assertFalse(row['request_ok'])
            self.assertFalse(row['correct'])

    def test_request_requires_identity_content_stop_and_correct_answer(self):
        good = {'model': 'test', 'choices': [{'message': {'content': 'FINAL: 42'}, 'finish_reason': 'stop'}]}
        bad = [dict(good, model='other'), dict(good, choices=[]),
               {'model': 'test', 'choices': [{'message': {'content': '42'}, 'finish_reason': 'length'}]},
               {'model': 'test', 'choices': [{'message': {'content': '', 'reasoning_content': '42'}, 'finish_reason': 'stop'}]}]
        for response in [good] + bad:
            with patch.object(smoke, 'http', return_value=json.dumps(response)) as http:
                result = smoke.request_case('http://fake', 'test', self.model, self.cases[0], 'serial')
            self.assertEqual(result['correct'], response is good)
            self.assertEqual(result['request_ok'], response is good)
            self.assertEqual(http.call_count, 1)
        with patch.object(smoke, 'http', side_effect=TimeoutError('timeout')) as http:
            row = smoke.request_case('http://fake', 'test', self.model, self.cases[0], 'serial')
        self.assertFalse(row['request_ok'])
        self.assertEqual(http.call_count, 1)  # no automatic retry

    def test_variant_think_close_tag_requires_exact_terminal_answer(self):
        case = next(item for item in self.cases if item['id'] == 'extract-json')
        prefix = '<think:opensource>Quoted FINAL: {"name":"Bob","age":27}'
        self.assertTrue(smoke.judge(prefix + '</think:opensource>FINAL: {"name":"Alice","age":27}', case))
        self.assertFalse(smoke.judge(prefix + '</think:opensource>FINAL: {"name":"Bob","age":27}', case))
        self.assertFalse(smoke.judge(prefix, case))
        self.assertFalse(smoke.judge(prefix + '</think:opensource>', case))

    def test_model_prompt_suffix_keeps_case_and_exact_judgment(self):
        response = {'model': 'test', 'choices': [{'message': {'content': 'FINAL: 42'}, 'finish_reason': 'stop'}]}
        model = dict(self.model, prompt_suffix='Give one answer and stop.')
        with patch.object(smoke, 'http', return_value=json.dumps(response)) as mocked:
            row = smoke.request_case('http://fake', 'test', model, self.cases[0], 'serial')
        self.assertTrue(row['correct'])
        prompt = mocked.call_args[0][2]['messages'][0]['content']
        self.assertTrue(prompt.startswith(self.cases[0]['prompt'] + '\n'))
        self.assertTrue(prompt.endswith(model['prompt_suffix']))
        self.assertEqual(mocked.call_count, 1)

    def test_profile_sampling_preserves_fixed_smoke_limits_and_identity(self):
        response = {'model': 'test', 'choices': [{'message': {'content': 'FINAL: 42'}, 'finish_reason': 'stop'}]}
        for options, temperature, top_p in (({}, 0, 1), ({'temperature': 0.6, 'top_p': 0.95,
                                                       'seed': 0, 'max_tokens': 1,
                                                       'model': 'wrong', 'n': 2}, 0.6, 0.95)):
            model = dict(self.model, request_options=options)
            with patch.object(smoke, 'http', return_value=json.dumps(response)) as mocked:
                row = smoke.request_case('http://fake', 'test', model, self.cases[0], 'serial')
            self.assertTrue(row['correct'])
            payload = mocked.call_args[0][2]
            self.assertEqual((payload['temperature'], payload['top_p']), (temperature, top_p))
            self.assertEqual((payload['seed'], payload['max_tokens'], payload['model'], payload['n']),
                             (20261110, model.get('max_tokens', 4096), 'test', 1))
            self.assertEqual(mocked.call_count, 1)

    def test_speculative_metrics_require_real_activity(self):
        before = {k: 2 for k in smoke.METRICS}
        after = dict(zip(smoke.METRICS, [4, 12, 6]))
        self.assertEqual(smoke.speculative_delta(before, after)[smoke.METRICS[2]], 4)
        for bad in (before, {k: 0 for k in smoke.METRICS},
                    dict(zip(smoke.METRICS, [4, 5, 9])), {k: float('nan') for k in smoke.METRICS}):
            with self.assertRaises(ValueError):
                smoke.speculative_delta(before, bad)
        raw = '\n'.join('{}{{model_name="test",engine="{}"}} {}'.format(k, i, i + 1)
                        for k in smoke.METRICS for i in range(2))
        raw += '\n' + smoke.METRICS[0] + '{model_name="other"} 9000'
        self.assertEqual(smoke.metrics(raw, 'test'), {k: 3 for k in smoke.METRICS})
        with self.assertRaises(ValueError):
            smoke.metrics('', 'test')

    def test_ready_marker_and_pinned_local_files_required(self):
        with patch.object(smoke, 'local_filesystem'):
            self.assertEqual(smoke.preflight(self.model, self.cache), self.path)
            for key in ('status', 'revision', 'manifest_sha256'):
                marker = dict(self.marker, **{key: 'wrong'})
                smoke.write_json(self.path / '.ci-modelhub-ready.json', marker)
                with self.assertRaises(ValueError):
                    smoke.preflight(self.model, self.cache)
            self.save_marker()
            (self.path / 'config.json').write_bytes(b'x' * self.model['cache_total_bytes'])
            with self.assertRaises(ValueError):
                smoke.preflight(self.model, self.cache)
            (self.path / '.ci-modelhub-ready.json').unlink()
            with self.assertRaises(ValueError):
                smoke.preflight(self.model, self.cache)

    def test_symlink_and_network_mount_rejected(self):
        target = self.root / 'outside.json'
        (self.path / 'config.json').rename(target)
        (self.path / 'config.json').symlink_to(target)
        with patch.object(smoke, 'local_filesystem'), self.assertRaises(ValueError):
            smoke.preflight(self.model, self.cache)
        for filesystem in ('nfs', 'nfs4', 'cifs', 'fuse.sshfs'):
            info = '1 0 0:1 / / rw - ext4 /dev/test rw\n2 1 0:2 / /opt/ci-models ro - {} source ro\n'.format(filesystem)
            with patch.object(Path, 'read_text', return_value=info), self.assertRaises(ValueError):
                smoke.local_filesystem(Path('/opt/ci-models/model/version'))
        with patch.object(Path, 'read_text', return_value='1 0 0:1 / /opt/ci-models ro - ext4 /dev/test ro\n'):
            smoke.local_filesystem(Path('/opt/ci-models/model/version'))

    def test_incomplete_suite_is_red_in_junit(self):
        selected = self.models[:2]
        result = {'id': selected[0]['id'], 'status': 'PASS', 'seconds': 1, 'correct_answers': 20}
        self.assertFalse(smoke.report(self.root, selected, [result], SMOKE / 'suite.json'))
        xml = ET.parse(str(self.root / 'junit.xml')).getroot()
        self.assertEqual(xml.get('failures'), '1')
        self.assertEqual(len(xml.findall('testcase')), 2)
        self.assertEqual(xml.find('.//failure').get('message'), 'Not completed')

    def test_dead_server_fails_startup(self):
        proc = Mock(returncode=1)
        proc.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            smoke.server_ready('http://127.0.0.1:1', 'test', proc, 1)

    def fake_executable(self):
        bindir = self.root / 'bin'
        bindir.mkdir(exist_ok=True)
        executable = bindir / 'vllm'
        executable.write_text('''#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from threading import Lock
args=sys.argv; port=int(args[args.index('--port')+1]); served=args[args.index('--served-model-name')+1]
cases=json.load(open(os.environ['SMOKE_TEST_CASES']))['cases']; answers={c['prompt']:c for c in cases}
count=0; lock=Lock()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def send(self,data):
  self.send_response(200);self.end_headers();self.wfile.write(data.encode())
 def do_GET(self):
  if self.path=='/health': self.send('ok')
  elif self.path=='/v1/models': self.send(json.dumps({'data':[{'id':served}]}))
  elif self.path=='/metrics': self.send('\\n'.join(k+'{model_name="'+served+'"} '+str(count*n) for k,n in [('vllm:spec_decode_num_drafts_total',1),('vllm:spec_decode_num_draft_tokens_total',4),('vllm:spec_decode_num_accepted_tokens_total',2)]))
 def do_POST(self):
  global count
  request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  c=answers[request['messages'][0]['content']]
  answer=json.dumps(c['expected'],ensure_ascii=False) if c['kind']=='json' else str(c['expected'])
  if os.environ.get('SMOKE_TEST_WRONG')=='1' and c['id']=='addition': answer='43'
  with lock: count+=1
  self.send(json.dumps({'model':served,'choices':[{'message':{'content':'FINAL: '+answer},'finish_reason':'stop'}]}))
class Server(ThreadingMixIn,HTTPServer): daemon_threads=True
if os.environ.get('SMOKE_TEST_FATAL')=='1': print('EngineDeadError: injected failure',flush=True)
Server(('127.0.0.1',port),Handler).serve_forever()
''')
        executable.chmod(0o755)
        return bindir

    def run_fake(self, wrong=False, fatal=False):
        bindir = self.fake_executable()
        output = self.root / 'results'
        output.mkdir()
        args = SimpleNamespace(output=str(output), cache_root=str(self.cache), devices=','.join(map(str, range(16))))
        self.model['check_speculative'] = True
        env = dict(PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                   SMOKE_TEST_CASES=str(SMOKE / 'cases.json'),
                   SMOKE_TEST_WRONG='1' if wrong else '0', SMOKE_TEST_FATAL='1' if fatal else '0')
        with patch.dict(os.environ, env), patch.object(smoke, 'local_filesystem'), contextlib.redirect_stdout(io.StringIO()):
            result = smoke.run_model(self.model, self.cases, args)
        launch = smoke.read_json(output / self.model['id'] / 'launch.json')
        self.assertEqual(launch['command'][2], str(self.path))
        self.assertTrue((output / self.model['id'] / 'server.log').is_file())
        self.assertEqual(len((output / self.model['id'] / 'answers.jsonl').read_text().splitlines()), 20)
        self.assertEqual(result['request_successes'], 20)
        return result

    def test_three_real_http_services_keep_results_devices_and_ports_isolated(self):
        bindir = self.fake_executable()
        output = self.root / 'parallel-results'
        output.mkdir()
        first = copy.deepcopy(self.model)
        second = copy.deepcopy(self.model)
        second['id'] = 'second-profile'
        third = copy.deepcopy(self.model)
        third['id'] = 'third-profile'
        for model in (first, second, third):
            model['logical_gpus'] = model['parallel']['tp'] = 4
            model['serve_args'][model['serve_args'].index('--tensor-parallel-size') + 1] = '4'
        args = SimpleNamespace(output=str(output), cache_root=str(self.cache),
                               devices=','.join(map(str, range(16))), max_parallel_models=3)
        env = dict(PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                   SMOKE_TEST_CASES=str(SMOKE / 'cases.json'), SMOKE_TEST_WRONG='0', SMOKE_TEST_FATAL='0')
        with patch.dict(os.environ, env), patch.object(smoke, 'local_filesystem'), contextlib.redirect_stdout(io.StringIO()):
            results = smoke.schedule_models([first, second, third], self.cases, args)
        self.assertTrue(all(r['status'] == 'PASS' and r['correct_answers'] == 20 for r in results))
        self.assertEqual(len(results), 3)
        groups = [set(r['allocated_devices']) for r in results]
        self.assertEqual([len(g) for g in groups], [4, 4, 4])
        self.assertEqual(len(set.union(*groups)), 12)
        commands = [r['command'] for r in results]
        self.assertEqual(len({cmd[cmd.index('--port') + 1] for cmd in commands}), 3)
        for result in results:
            self.assertIn('startup_seconds', result['timings'])
            self.assertIn('cleanup_seconds', result['timings'])
            self.assertTrue((output / result['id'] / 'result.json').exists())

    def test_real_http_process_suite_passes(self):
        result = self.run_fake()
        self.assertEqual(result['status'], 'PASS', result['failures'])
        self.assertEqual(result['correct_answers'], 20)
        self.assertEqual(result['speculative_delta'][smoke.METRICS[1]], 80)

    def test_wrong_answer_fails_even_with_http_success(self):
        result = self.run_fake(wrong=True)
        self.assertEqual(result['correct_answers'], 18)
        self.assertEqual(result['status'], 'FAIL')

    def test_fatal_server_log_fails_even_with_correct_answers(self):
        result = self.run_fake(fatal=True)
        self.assertEqual(result['correct_answers'], 20)
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('Fatal server log', ' '.join(result['failures']))


if __name__ == '__main__':
    unittest.main()
