#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Pinned ModelHub serving smoke. Stdlib only; --validate never starts a model."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import threading
import unicodedata
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation

HERE = Path(__file__).resolve().parent
METRICS = (
    'vllm:spec_decode_num_drafts_total',
    'vllm:spec_decode_num_draft_tokens_total',
    'vllm:spec_decode_num_accepted_tokens_total',
)
FATAL = re.compile(r'EngineDeadError|CUDA out of memory|OutOfMemoryError|'
                   r'(?i:worker[^\n]*(?:died unexpectedly|exited unexpectedly))')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load_weight_manifest(model, root):
    """Resolve pinned weight metadata separately from editable launch/test settings."""
    if 'weight_manifest' not in model:
        return model  # Older PR revisions keep their inline cache manifest.
    reference = model['weight_manifest']
    if not isinstance(reference, str) or not re.fullmatch(r'weights/[A-Za-z0-9_-][A-Za-z0-9_.-]*\.json', reference):
        raise ValueError('Weight manifest must name a JSON file under weights/')
    path = root / reference
    if path.resolve() != path or not path.is_file():
        raise ValueError('Missing weight manifest or symlink: ' + reference)
    weights = json.loads(path.read_text())
    identity = {'name', 'owner', 'slug', 'platform_version', 'revision'}
    metadata = {'url', 'storage_relpath', 'config_sha256', 'configuration_source', 'notes',
                'cache_relpath', 'cache_manifest_sha256', 'cache_total_bytes', 'cache_files'}
    if not isinstance(weights, dict) or not identity <= weights.keys() or not set(weights) <= identity | metadata:
        raise ValueError('Weight manifest may contain only identity and weight metadata')
    if any(model.get(key) != weights[key] for key in identity):
        raise ValueError('Weight manifest identity/version disagrees with model profile')
    if metadata.intersection(model):
        raise ValueError('Split model profiles must keep weight metadata in the manifest')
    return dict({key: value for key, value in model.items() if key != 'weight_manifest'},
                **{key: value for key, value in weights.items() if key not in identity})


def load_suite(path):
    """Expand the product-owned list into the corresponding per-model JSON files."""
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file():
        raise ValueError('Suite must be a regular file without symlink components')
    suite = json.loads(path.read_text())
    ids = suite['models']
    if not isinstance(ids, list) or not ids or not all(
            isinstance(name, str) and re.fullmatch(r'[a-z0-9-]+', name) for name in ids):
        raise ValueError('Suite models must be an explicit list of model IDs')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate model in suite')
    models = []
    for name in ids:
        profile = path.parent / 'models' / (name + '.json')
        if profile.resolve() != profile or not profile.is_file():
            raise ValueError('Missing model profile or symlink: ' + name)
        model = json.loads(profile.read_text())
        if model['id'] != name:
            raise ValueError('Model file ID disagrees with suite: ' + name)
        models.append(load_weight_manifest(model, path.parent))
    return dict(suite, models=models)


def load_plan(manifest_path, selected, cases_path):
    manifest, cases = load_suite(manifest_path), read_json(cases_path)
    models = manifest['models']
    ids = [m['id'] for m in models]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Manifest must contain unique nonempty model IDs')
    if len(cases['cases']) != 10 or len({c['id'] for c in cases['cases']}) != 10:
        raise ValueError('Smoke requires exactly 10 unique cases, each run twice')
    for case in cases['cases']:
        if case['kind'] not in ('number', 'text', 'json') or not case['prompt']:
            raise ValueError('Invalid answer checker: ' + case['id'])
        if not judge('FINAL: ' + (json.dumps(case['expected']) if case['kind'] == 'json'
                                 else str(case['expected'])), case):
            raise ValueError('Invalid expected answer: ' + case['id'])
    for model in models:
        if model['owner'] != 'iluvatar-aiac-llmfp-inference':
            raise ValueError('Unexpected ModelHub owner')
        expected_path = '.snapshots/model/{}/{}'.format(model['slug'], model['platform_version'])
        if model['storage_relpath'] != expected_path or '..' in Path(expected_path).parts:
            raise ValueError('Unpinned or invalid snapshot path')
        if not re.fullmatch(r'[a-f0-9]{64}', model['revision']):
            raise ValueError('Missing revision')
        if not re.fullmatch(r'[a-z0-9-]+', model['id']):
            raise ValueError('Invalid model ID')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', model['slug']) or not re.fullmatch(r'v[0-9]+', model['platform_version']):
            raise ValueError('Invalid pinned model identity')
        if model['cache_relpath'] != model['slug'] + '/' + model['platform_version'] + '-' + model['revision'][:12]:
            raise ValueError('Invalid local cache path')
        if not re.fullmatch(r'[a-f0-9]{64}', model['cache_manifest_sha256']):
            raise ValueError('Missing cache manifest fingerprint')
        paths = []
        for file in model['cache_files']:
            path = Path(file['path'])
            if path.is_absolute() or '..' in path.parts or str(path) != file['path'] or not path.parts:
                raise ValueError('Unsafe published file path')
            if file['size'] < 0 or not re.fullmatch(r'[a-f0-9]{64}', file['sha256']):
                raise ValueError('Invalid published file checksum/size')
            paths.append(file['path'])
        if not paths or len(paths) != len(set(paths)) or 'config.json' not in paths:
            raise ValueError('Empty, duplicate or incomplete published file manifest')
        if sum(f['size'] for f in model['cache_files']) != model['cache_total_bytes']:
            raise ValueError('Cache manifest byte count mismatch')
        parallel = model['parallel']
        n = parallel['tp'] * parallel['pp'] * parallel['dp']
        if not 1 <= n <= 16 or n != model['logical_gpus']:
            raise ValueError('GPU allocation must match TP x PP x DP and be <= 16')
        argv = model['serve_args']
        for key, flag in [('tp', '--tensor-parallel-size'), ('pp', '--pipeline-parallel-size'),
                          ('dp', '--data-parallel-size')]:
            actual = int(argv[argv.index(flag) + 1]) if flag in argv else 1
            if actual != parallel[key]:
                raise ValueError('Parallel arguments disagree with allocation')
        if '--enforce-eager' not in argv:
            config = json.loads(argv[argv.index('--compilation-config') + 1])
            if config.get('mode') != 0 or config.get('cudagraph_mode') != 'FULL_DECODE_ONLY':
                raise ValueError('Unsupported execution mode')
        if bool(model['check_speculative']) != ('--speculative-config' in argv):
            raise ValueError('Speculative arguments must have a runtime metric gate')
        if any(k in model['environment'] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')):
            raise ValueError('The runner owns device allocation')
        if any(flag in argv for flag in ('--host', '--port', '--served-model-name')):
            raise ValueError('The runner owns HTTP address and served identity')
    names = ids if selected == 'all' else [s.strip() for s in selected.split(',')]
    if not names or any(not n or n not in ids for n in names) or len(names) != len(set(names)):
        raise ValueError('Unknown, empty or duplicate model selection: ' + selected)
    return manifest, [m for m in models if m['id'] in names], cases['cases']


def final_answer(content):
    if not isinstance(content, str):
        return ''
    # Some content-only templates use </think:opensource> without an opening tag.
    closing = list(re.finditer(r'</think(?::[a-zA-Z0-9_-]+)?>', content, flags=re.I))
    if closing:
        content = content[closing[-1].end():]
    elif re.search(r'<think(?::[a-zA-Z0-9_-]+)?>', content, flags=re.I):
        return ''  # reasoning alone is not a final answer
    last_line = content.strip().splitlines()[-1] if content.strip() else ''
    # Accept an explicit terminal marker after prose as well as on its own line.
    # Still compare the entire suffix; never search reasoning or answer substrings.
    match = re.search(r'(?i)(?<!\w)FINAL\s*[:：]\s*(.+?)\s*$', last_line)
    if match:
        return match.group(1).strip()
    return content.strip()


def judge(content, case):
    answer = final_answer(content)
    if not answer:
        return False
    if case['kind'] == 'number':
        boxed = re.fullmatch(r'\\boxed\{([^{}]+)\}', answer)
        if boxed:
            answer = boxed.group(1)
        answer = unicodedata.normalize('NFKC', answer).strip()
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', answer):
            return False
        try:
            return Decimal(answer) == Decimal(str(case['expected']))
        except InvalidOperation:
            return False
    if case['kind'] == 'json':
        try:
            actual = json.loads(answer)
            # JSON booleans must not compare equal to numeric answers in Python.
            return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(
                case['expected'], sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            return False
    normalize = lambda value: unicodedata.normalize('NFKC', value).strip().strip('。.!！').casefold()
    return normalize(answer) in {normalize(v) for v in [case['expected']] + case.get('aliases', [])}


def http(base, path, payload=None, timeout=10):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError('HTTP {}'.format(response.status))
        data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError('Response exceeds smoke limit')
        return data.decode('utf-8')


def request_case(base, served, model, case, phase):
    result = {'case': case['id'], 'phase': phase, 'expected': case['expected'],
              'request_ok': False, 'correct': False}
    payload = dict(model.get('request_options', {}))
    prompt = case['prompt']
    if model.get('prompt_suffix'):
        prompt += '\n' + model['prompt_suffix']
    payload.update(model=served, messages=[{'role': 'user', 'content': prompt}],
                   temperature=payload.get('temperature', 0),
                   top_p=payload.get('top_p', 1), n=1, seed=20261110,
                   max_tokens=model.get('max_tokens', 4096), stream=False)
    result['request'] = payload
    start = time.monotonic()
    try:
        response = json.loads(http(base, '/v1/chat/completions', payload,
                                   timeout=model.get('request_timeout_seconds', 300)))
        result['response'] = response
        if response.get('model') != served or len(response.get('choices', [])) != 1:
            raise ValueError('Wrong served model or choice count')
        choice = response['choices'][0]
        content = choice['message'].get('content')
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Generation did not finish normally: ' + str(choice.get('finish_reason')))
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Empty final answer (reasoning_content alone does not pass)')
        result['request_ok'] = True
        result['correct'] = judge(content, case)
        if not result['correct']:
            result['error'] = 'Final answer does not match the fixed expected value'
    except Exception as exc:
        result['error'] = '{}: {}'.format(type(exc).__name__, exc)
    result['seconds'] = round(time.monotonic() - start, 3)
    return result


def metrics(text, served):
    values = {}
    for line in text.splitlines():
        match = re.match(r'([^ {]+)(\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$', line)
        if not match or match.group(1) not in METRICS:
            continue
        labels = match.group(2) or ''
        if 'model_name=' in labels and ('model_name="' + served + '"') not in labels:
            continue
        number = float(match.group(3))
        if not math.isfinite(number) or number < 0:
            raise ValueError('Invalid speculative counter')
        name = match.group(1)
        values[name] = values.get(name, 0) + number
    if any(name not in values for name in METRICS):
        raise ValueError('Speculative metrics missing')
    return values


def speculative_delta(before, after):
    delta = {key: after[key] - before[key] for key in METRICS}
    if any(not math.isfinite(value) or value <= 0 for value in delta.values()):
        raise ValueError('No actual speculative activity or counters reset: ' + str(delta))
    if delta[METRICS[2]] > delta[METRICS[1]]:
        raise ValueError('Accepted tokens exceed drafted tokens')
    return delta


def local_filesystem(path):
    """Check the actual bind mount in the runtime container, not its parent."""
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        mount = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), left.split()[4])
        if str(path) == mount or str(path).startswith(mount.rstrip('/') + '/'):
            mounts.append((len(mount), right.split()[0]))
    filesystem = max(mounts)[1] if mounts else None
    if filesystem not in ('ext4', 'xfs', 'btrfs', 'zfs'):
        raise ValueError('Model cache must be on a local filesystem, found: ' + str(filesystem))


def checked_file(path, relative):
    target = path / relative
    if target.resolve() != target or not target.is_file():
        raise ValueError('Missing file or symlink in pinned local cache: ' + str(target))
    return target


def preflight(model, cache_root):
    root = Path(cache_root).absolute()
    path = root / model['cache_relpath']
    if path.resolve() != path or not path.is_dir():
        raise ValueError('Local cache missing or contains symlinks: ' + str(path))
    local_filesystem(path)
    marker = read_json(checked_file(path, '.ci-modelhub-ready.json'))
    required = {key: model[key] for key in ('owner', 'slug', 'platform_version', 'revision')}
    required.update(status='READY', manifest_sha256=model['cache_manifest_sha256'],
                    file_count=len(model['cache_files']), total_bytes=model['cache_total_bytes'])

    # Organization rename changes identity metadata, not the verified weight files.
    # Keep the original READY receipt valid only for the exact pinned old manifest.
    if (model['owner'] == 'iluvatar-aiac-llmfp-inference'
            and marker.get('owner') == 'iluvatar-aac-llmfp'):
        original = {key: model[key] for key in
                    ('name', 'slug', 'owner', 'platform_version', 'revision', 'storage_relpath')}
        original.update(owner='iluvatar-aac-llmfp', files=model['cache_files'],
                        bytes=model['cache_total_bytes'])
        original_sha = hashlib.sha256(json.dumps(
            original, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        required.update(owner='iluvatar-aac-llmfp', manifest_sha256=original_sha)
    if any(marker.get(key) != value for key, value in required.items()):
        raise ValueError('Local cache is not READY for the pinned manifest/version')
    for file in model['cache_files']:
        target = checked_file(path, file['path'])
        if target.stat().st_size != file['size']:
            raise ValueError('Missing or incomplete published file: ' + str(target))
        # Full shard checksums were verified before the READY atomic rename.
        # Recheck small configuration/tokenizer/code files, without rereading TB
        # of immutable tensors on every PR. The container mounts this tree :ro.
        if file['size'] <= 16 * 1024 * 1024:
            if hashlib.sha256(target.read_bytes()).hexdigest() != file['sha256']:
                raise ValueError('Local metadata checksum mismatch: ' + str(target))
    if hashlib.sha256((path / 'config.json').read_bytes()).hexdigest() != model['config_sha256']:
        raise ValueError('Pinned config.json checksum mismatch')
    return path


PORT_LOCK = threading.Lock()
USED_PORTS = set()


def choose_ports():
    # Never hand the same HTTP/master port to two workers in this suite.
    with PORT_LOCK:
        sockets = []
        ports = []
        try:
            while len(ports) < 2:
                sock = socket.socket()
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
                if port in USED_PORTS:
                    sock.close()
                    continue
                sockets.append(sock)
                ports.append(port)
                USED_PORTS.add(port)
            return ports
        finally:
            for sock in sockets:
                sock.close()


class RunControl:
    """Cancel only model process groups created by this runner."""
    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.processes = {}

    def check(self):
        if self.cancelled.is_set():
            raise InterruptedError('Model suite cancelled')

    @staticmethod
    def terminate(proc):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def register(self, proc):
        with self.lock:
            self.processes[proc.pid] = proc
            if self.cancelled.is_set():
                self.terminate(proc)

    def unregister(self, proc):
        if proc is not None:
            with self.lock:
                self.processes.pop(proc.pid, None)

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            for proc in self.processes.values():
                self.terminate(proc)


def server_ready(base, served, proc, timeout, control=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if control is not None:
            control.check()
        if proc.poll() is not None:
            raise RuntimeError('Server exited during startup: {}'.format(proc.returncode))
        try:
            data = json.loads(http(base, '/v1/models'))
            if served in {m['id'] for m in data['data']}:
                http(base, '/health')
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError('Model startup timeout')


def stop_process(proc):
    if proc is None:
        return
    # The session/group belongs only to this server. Always clean descendants,
    # even if the group leader already died; never pkill another model/server.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)
    deadline = time.monotonic() + 10
    while True:
        live = []
        for stat in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = stat.read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == proc.pid and fields[0] not in ('Z', 'X'):
                    live.append(stat.parent.name)
            except (OSError, ValueError, IndexError):
                continue
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Server processes remain after SIGKILL: ' + ','.join(live))
        time.sleep(0.1)


def run_model(model, cases, args, control=None):
    control = control or RunControl()
    directory = Path(args.output) / model['id']
    directory.mkdir()
    result = {'id': model['id'], 'name': model['name'], 'status': 'FAIL',
              'expected_requests': 20, 'request_successes': 0, 'correct_answers': 0,
              'weight': {k: model[k] for k in ('slug', 'platform_version', 'revision', 'config_sha256')},
              'rows': [], 'failures': [], 'coverage': 'text smoke; not benchmark accuracy or long-run stability'}
    proc = None
    start = time.monotonic()
    result['started_at'] = time.time()
    stage_start = start
    timings = {}
    log = None
    try:
        control.check()
        model_path = preflight(model, args.cache_root)
        devices = args.devices.split(',')
        if len(devices) < model['logical_gpus']:
            raise ValueError('Not enough allocated GPUs for ' + model['id'])
        port, master_port = choose_ports()
        served = 'ci-smoke-' + model['id'] + '-' + uuid.uuid4().hex[:12]
        base = 'http://127.0.0.1:' + str(port)
        command = ['vllm', 'serve', str(model_path), '--served-model-name', served,
                   '--host', '127.0.0.1', '--port', str(port)]
        command += [str(a).replace('{master_port}', str(master_port)) for a in model['serve_args']]
        # Prevent per-model flags from leaking between cases or from a caller.
        env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('VLLM_', 'IXFORMER_', 'NCCL_', 'TORCH_NCCL_', 'CUDA_VISIBLE_', 'TORCH_USE_ILU_'))}
        env.update(model['environment'])
        env.update(CUDA_VISIBLE_DEVICES=','.join(devices[:model['logical_gpus']]),
                   VLLM_HOST_IP='127.0.0.1', VLLM_USAGE_STATS_DISABLED='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                   HF_HOME=str(directory / 'cache' / 'huggingface'),
                   XDG_CACHE_HOME=str(directory / 'cache'),
                   TRITON_CACHE_DIR=str(directory / 'cache' / 'triton'))
        timings['preflight_seconds'] = round(time.monotonic() - stage_start, 3)
        stage_start = time.monotonic()
        result['allocated_devices'] = devices[:model['logical_gpus']]
        result['command'] = command
        result['environment'] = {k: env[k] for k in set(model['environment']) | {
            'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'VLLM_HOST_IP'}}
        write_json(directory / 'launch.json', {k: result[k] for k in ('command', 'environment', 'weight')})
        log = (directory / 'server.log').open('wb')
        print('[model-smoke] starting ' + model['id'], flush=True)
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=str(directory), start_new_session=True)
        control.register(proc)
        server_ready(base, served, proc, model.get('ready_timeout_seconds', 1800), control)
        timings['startup_seconds'] = round(time.monotonic() - stage_start, 3)
        stage_start = time.monotonic()
        before = None
        if model['check_speculative']:
            raw = http(base, '/metrics')
            (directory / 'metrics-before.txt').write_text(raw)
            before = metrics(raw, served)
        with (directory / 'answers.jsonl').open('w') as answers:
            def record(row):
                result['rows'].append(row)
                answers.write(json.dumps(row, ensure_ascii=False) + '\n')
                answers.flush()
                print('[model-smoke] {} {} {} {}'.format(model['id'], row['phase'], row['case'],
                                                       'PASS' if row['correct'] else 'FAIL'), flush=True)
            for case in cases:
                control.check()
                record(request_case(base, served, model, case, 'serial'))
                if proc.poll() is not None:
                    raise RuntimeError('Server died during serial requests')
            control.check()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(request_case, base, served, model, case, 'concurrent') for case in cases]
                for future in concurrent.futures.as_completed(futures):
                    control.check()
                    record(future.result())
        timings['requests_seconds'] = round(time.monotonic() - stage_start, 3)
        stage_start = time.monotonic()
        if proc.poll() is not None:
            raise RuntimeError('Server died during concurrent requests')
        http(base, '/health')
        if before is not None:
            # Workers export counters asynchronously. Poll the metric endpoint,
            # never rerun a question to turn a failed result into a pass.
            deadline = time.monotonic() + 20
            while True:
                control.check()
                raw = http(base, '/metrics')
                (directory / 'metrics-after.txt').write_text(raw)
                try:
                    result['speculative_delta'] = speculative_delta(before, metrics(raw, served))
                    break
                except ValueError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(2)
        log.flush()
        with (directory / 'server.log').open(errors='replace') as stream:
            for line in stream:
                if FATAL.search(line):
                    raise RuntimeError('Fatal server log: ' + line[:500].strip())
        if proc.poll() is not None:
            raise RuntimeError('Server exited before final health check')
        http(base, '/health')
        timings['validation_seconds'] = round(time.monotonic() - stage_start, 3)
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, (KeyboardInterrupt, InterruptedError)):
            result['interrupted'] = True
        result['failures'].append('{}: {}'.format(type(exc).__name__, exc))
    finally:
        control.unregister(proc)
        cleanup_start = time.monotonic()
        try:
            stop_process(proc)
        except Exception as exc:
            result['failures'].append('Server cleanup failed: ' + str(exc))
            result['cleanup_failed'] = True
        if log:
            log.close()
        result['request_successes'] = sum(row['request_ok'] for row in result['rows'])
        result['correct_answers'] = sum(row['correct'] for row in result['rows'])
        if len(result['rows']) != 20:
            result['failures'].append('Incomplete smoke: {}/20 attempted'.format(len(result['rows'])))
        if result['request_successes'] != 20 or result['correct_answers'] != 20:
            result['failures'].append('Requires 20/20 successful requests and 20/20 correct answers')
        result['status'] = 'PASS' if not result['failures'] else 'FAIL'
        timings['cleanup_seconds'] = round(time.monotonic() - cleanup_start, 3)
        result['timings'] = timings
        result['finished_at'] = time.time()
        result['seconds'] = round(time.monotonic() - start, 3)
        write_json(directory / 'result.json', result)
    return result


def schedule_models(selected, cases, args, on_result=None, run=run_model):
    """Use disjoint GPU sets; a model releases its allocation only after cleanup."""
    devices = args.devices.split(',')
    limit = args.max_parallel_models
    if not 1 <= limit <= 3:
        raise ValueError('At most three models may run concurrently')
    if any(m['logical_gpus'] > len(devices) for m in selected):
        raise ValueError('Selected model exceeds the allocated GPU count')
    pending = list(selected)
    active = {}
    free = set(devices)
    results = []
    control = RunControl()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=limit)

    def collect(future):
        model, assigned = active.pop(future)
        result = future.result()
        results.append(result)
        if result.get('cleanup_failed') or result.get('interrupted'):
            control.cancel()
        else:
            free.update(assigned)
        if on_result:
            on_result(results)

    try:
        while pending or active:
            while not control.cancelled.is_set() and len(active) < limit:
                index = next((i for i, m in enumerate(pending)
                              if m['logical_gpus'] <= len(free)), None)
                if index is None:
                    break
                model = pending.pop(index)
                assigned = [d for d in devices if d in free][:model['logical_gpus']]
                free.difference_update(assigned)
                worker_args = argparse.Namespace(**vars(args))
                worker_args.devices = ','.join(assigned)
                print('[model-smoke] allocate {} GPUs={}'.format(model['id'], worker_args.devices), flush=True)
                future = pool.submit(run, model, cases, worker_args, control)
                active[future] = (model, assigned)
            if not active:
                break
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                collect(future)
    except BaseException:
        control.cancel()
        raise
    finally:
        pool.shutdown(wait=True)
        for future in list(active):
            try:
                collect(future)
            except Exception:
                pass  # A missing model remains a failure in the suite report.
    return results


def report(output, selected, results, manifest):
    by_id = {r['id']: r for r in results}
    summary = {'status': 'PASS' if len(results) == len(selected) and all(
        r['status'] == 'PASS' for r in results) else 'FAIL',
        'selected_models': [m['id'] for m in selected], 'results': results,
        'manifest_sha256': hashlib.sha256(json.dumps(load_suite(manifest), sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    timed = [r for r in results if 'started_at' in r and 'finished_at' in r]
    if timed:
        wall = max(r['finished_at'] for r in timed) - min(r['started_at'] for r in timed)
        work = sum(r['seconds'] for r in timed)
        summary['timing'] = {'completed_models_wall_seconds': round(wall, 3),
                             'total_model_seconds': round(work, 3),
                             'overlap_seconds': round(max(0, work - wall), 3)}
    write_json(Path(output) / 'summary.json', summary)
    suite = ET.Element('testsuite', name='modelhub-smoke', tests=str(len(selected)))
    failures = 0
    for model in selected:
        result = by_id.get(model['id'])
        case = ET.SubElement(suite, 'testcase', classname='model_smoke', name=model['name'],
                             time=str(result.get('seconds', 0) if result else 0))
        if not result or result['status'] != 'PASS':
            failures += 1
            failure = ET.SubElement(case, 'failure', message='Smoke failed' if result else 'Not completed')
            failure.text = json.dumps(result['failures'] if result else ['Model did not run'], ensure_ascii=False)
        ET.SubElement(case, 'system-out').text = json.dumps({
            'correct': result['correct_answers'] if result else 0, 'required': 20,
            'version': model['platform_version'], 'revision': model['revision']}, ensure_ascii=False)
    suite.set('failures', str(failures))
    ET.ElementTree(suite).write(str(Path(output) / 'junit.xml'), encoding='utf-8', xml_declaration=True)
    return summary['status'] == 'PASS'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', '--manifest', dest='manifest', default=str(HERE / 'suite.json'))
    parser.add_argument('--cases', default=str(HERE / 'cases.json'))
    parser.add_argument('--models', default='all', help='all or comma-separated manifest IDs')
    parser.add_argument('--cache-root', default='/opt/ci-models')
    parser.add_argument('--devices', default=','.join(str(i) for i in range(16)))
    parser.add_argument('--output', default='model-smoke-results')
    parser.add_argument('--max-parallel-models', type=int, choices=(1, 2, 3), default=3,
                        help='Up to three disjoint GPU allocations by default; 1 for serial comparison')
    parser.add_argument('--validate', action='store_true', help='Validate configuration without GPU/network access')
    parser.add_argument('--check-weights', action='store_true', help='Check pinned local snapshot files without GPU access')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    manifest, selected, cases = load_plan(args.manifest, args.models, args.cases)
    devices = args.devices.split(',')
    if not 1 <= len(devices) <= 16 or len(set(devices)) != len(devices) or any(
            not re.fullmatch(r'\d+', d) for d in devices):
        parser.error('--devices must list 1 to 16 distinct logical GPU indices')
    if args.validate or args.check_weights:
        for model in selected:
            if args.check_weights:
                preflight(model, args.cache_root)
            print('{} | {} | {} GPUs | speculative={}'.format(
                model['id'], model['platform_version'], model['logical_gpus'], model['check_speculative']))
        return 0
    args.output = str(Path(args.output).resolve())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Output must be empty; preserve previous failed results')
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'cases.json', {'cases': cases})
    results = []
    report(output, selected, results, args.manifest)
    try:
        def record(completed):
            results[:] = completed
            report(output, selected, results, args.manifest)
        schedule_models(selected, cases, args, on_result=record)
    except KeyboardInterrupt:
        pass  # Active model groups have been stopped; unfinished models stay red.
    finally:
        passed = report(output, selected, results, args.manifest)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
