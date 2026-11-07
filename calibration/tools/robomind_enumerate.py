"""Enumerate every episode of a RoboMIND split by reading tar headers only (no frame data).

The episode ids (MMDD_HHMMSS) give the recording session of every episode, i.e. which calibrated pose applies.
The archives are streamed (multi-part archives chained); member data is skipped, only names are kept. Results are
written per task to <out>/<task>.json, so an interrupted run resumes.

    python tools/robomind_enumerate.py --out <dir> [--embodiment h5_ur_1rgb] [--split benchmark1_1_compressed] [--workers 8]
"""
import argparse
import io
import json
import os
import re
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

EP = re.compile(r'/(\d{4}_\d{6})/')
_print_lock = threading.Lock()


class ChainedParts(io.RawIOBase):
    """Read several HTTP objects end-to-end as one stream (multi-part tar.gz)."""

    def __init__(self, urls, token, chunk=4 << 20):
        self.urls, self.token, self.chunk = list(urls), token, chunk
        self.i, self.resp, self.nbytes = 0, None, 0

    def _open_next(self):
        if self.resp is not None:
            self.resp.close()
            self.resp = None
        if self.i >= len(self.urls):
            return False
        req = urllib.request.Request(self.urls[self.i], headers={'Authorization': f'Bearer {self.token}'})
        self.resp = urllib.request.urlopen(req, timeout=120)
        self.i += 1
        return True

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.resp is None and not self._open_next():
                return 0
            n = self.resp.readinto(b)
            if n:
                self.nbytes += n
                return n
            if not self._open_next():
                return 0


def enumerate_task(task, urls, token, out_dir, total, done_counter):
    dest = os.path.join(out_dir, f'{task}.json')
    if os.path.exists(dest):
        with done_counter['lock']:
            done_counter['n'] += 1
        return task, None, 0, True
    eps, t0 = set(), time.time()
    stream = ChainedParts(urls, token)
    try:
        with tarfile.open(fileobj=io.BufferedReader(stream, buffer_size=8 << 20), mode='r|gz') as tf:
            for m in tf:
                hit = EP.search('/' + m.name)
                if hit:
                    eps.add(hit.group(1))
    except Exception as e:
        with _print_lock:
            print(f'  !! {task}: {type(e).__name__} {str(e)[:80]}  (kept {len(eps)} so far)', flush=True)
    json.dump(sorted(eps), open(dest, 'w'))
    dt = time.time() - t0
    with done_counter['lock']:
        done_counter['n'] += 1
        k = done_counter['n']
    with _print_lock:
        print(f'[{k}/{total}] {task[:46]:46s} {len(eps):5d} episodes  '
              f'{stream.nbytes/1e9:5.2f} GB  {dt/60:5.1f} min  {stream.nbytes/1e6/max(dt,1):4.0f} MB/s', flush=True)
    return task, sorted(eps), stream.nbytes, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='directory for the per-task episode lists')
    ap.add_argument('--split', default='benchmark1_1_compressed')
    ap.add_argument('--embodiment', default='h5_ur_1rgb')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from huggingface_hub import HfApi, get_token
    token = os.environ.get('HF_TOKEN') or get_token()
    api = HfApi(token=token)
    ents = list(api.list_repo_tree('x-humanoid-robomind/RoboMIND',
                                   path_in_repo=f'{args.split}/{args.embodiment}', repo_type='dataset'))
    parts = {}
    for e in ents:
        base = os.path.basename(e.path).split('.tar.gz')[0]
        parts.setdefault(base, []).append(e.path)
    for k in parts:
        parts[k].sort()  # part-aa, part-ab, ...
    tasks = sorted(parts)
    total_bytes = sum(getattr(e, 'size', 0) or 0 for e in ents)
    print(f'{len(tasks)} tasks, {total_bytes/1e12:.2f} TB compressed, {args.workers} workers', flush=True)

    root = 'https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND/resolve/main/'
    counter = {'n': 0, 'lock': threading.Lock()}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(enumerate_task, t, [root + p for p in parts[t]], token, args.out, len(tasks), counter)
                for t in tasks]
        for f in futs:
            f.result()
    print(f'\nall done in {(time.time()-t0)/3600:.2f} h -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
