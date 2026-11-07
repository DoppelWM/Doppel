"""Download RoboMIND episodes from the gated Hugging Face repo by streaming the task archives.

Accept the terms at https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND and run ``hf auth login`` (or set
``HF_TOKEN``). Each task is a multi-part ``<task>.tar.gz.part-aa, -ab, ...`` archive; it is streamed and only the
requested episodes are written.

    # a few UR5e episodes per task, e.g. for tools/robomind_sessions.py
    python tools/fetch_robomind.py --out <dir> --embodiment h5_ur_1rgb --all_tasks --n_episodes 20
    # specific episode ids (JSON list of "MMDD_HHMMSS" or "<task>__MMDD_HHMMSS")
    python tools/fetch_robomind.py --out <dir> --embodiment h5_ur_1rgb --all_tasks --keep_episodes ids.json

Output: ``<out>/<task>/<episode_id>/data/trajectory.hdf5``. A finished archive leaves ``<out>/_done/<task>.json``,
so a re-run resumes.
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi, get_token

REPO = 'x-humanoid-robomind/RoboMIND'
_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


class _HttpParts(io.RawIOBase):
    """Sequential reader over the concatenated multi-part archive fetched with HTTP range-free streaming."""

    def __init__(self, urls, token, retries=3):
        self.urls = list(urls)
        self.token = token
        self.retries = retries
        self.resp = None
        self.total = 0

    def _open_next(self):
        if not self.urls:
            return False
        url = self.urls.pop(0)
        req = urllib.request.Request(url, headers={'Authorization': f'Bearer {self.token}'} if self.token else {})
        for attempt in range(self.retries):
            try:
                self.resp = urllib.request.urlopen(req, timeout=60)
                return True
            except (urllib.error.URLError, OSError) as e:                 # a multi-hour stream will meet a few
                if attempt == self.retries - 1:
                    raise
                log(f'[fetch]   retry {attempt + 1}/{self.retries - 1} opening {os.path.basename(url)}: {e}')
                time.sleep(5 * (attempt + 1))
        return False

    def readinto(self, b):
        while True:
            if self.resp is None and not self._open_next():
                return 0
            n = self.resp.readinto(b)
            if n:
                self.total += n
                return n
            self.resp = None

    def readable(self):
        return True


_TASK_EP = re.compile(r'^(?:(.+)__)?(\d{4}_\d{6})$')       # "<task>__<MMDD_HHMMSS>" or a bare episode id


def _task_ep(e):
    """(task | None, episode id | None) from "<task>__<id>", a bare id, or {"task", "ep"}."""
    if isinstance(e, dict):
        return e.get('task'), e.get('ep')
    m = _TASK_EP.match(str(e))
    return (m.group(1), m.group(2)) if m else (None, None)


def load_keep_episodes(path):
    """(set of episode ids, {task: set of ids} or None).

    Accepted: ``["1023_133146", ...]``, ``{"episodes": [...]}``, or ``{session: {"session_episodes_by_task":
    {task: [ids]}, "episodes": [...]}}``; with a per-task map, archives without wanted episodes are not opened.
    """
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, list):
        return set(blob), None
    if isinstance(blob, dict) and isinstance(blob.get('episodes'), list) and blob['episodes'] and \
            isinstance(blob['episodes'][0], str):
        return set(blob['episodes']), None
    records = blob.get('records', blob) if isinstance(blob, dict) else {}
    keep, by_task = set(), {}
    for rec in records.values():
        if not isinstance(rec, dict):
            continue
        for task, ids in (rec.get('session_episodes_by_task') or {}).items():
            by_task.setdefault(task, set()).update(ids)
            keep.update(ids)
        for e in rec.get('fit_episodes') or rec.get('episodes') or []:    # the fitted scenes, always wanted too
            task, ep = _task_ep(e)
            if ep:
                keep.add(ep)
                if task:
                    by_task.setdefault(task, set()).add(ep)
    if not keep:
        raise SystemExit(f'{path}: no episode ids found')
    return keep, (by_task or None)


def episode_of(member_name):
    """.../success_episodes/train/<episode>/data/trajectory.hdf5 -> <episode>, or None for anything else."""
    rel = member_name.split('/')
    if 'data' in rel and rel.index('data') >= 1:
        return rel[rel.index('data') - 1], rel[rel.index('data'):]
    return None, None


def extract(tar, member, dst):
    """Write one member, unless an identical-sized file is already there (that is what makes a re-run cheap)."""
    if os.path.exists(dst) and os.path.getsize(dst) == member.size:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with tar.extractfile(member) as src, open(dst + '.part', 'wb') as f:
        while True:
            chunk = src.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    os.replace(dst + '.part', dst)
    return True


def fetch_task(task, entries, args, keep, want):
    """Stream one task archive. ``keep``: ids to extract (None = the --n_episodes budget). ``want``: the
    ids this archive is known to hold, or None when that is unknown."""
    out_task = os.path.join(args.out, task)
    marker = os.path.join(args.out, '_done', f'{task}.json')
    prev = None if args.force else _marker(marker)
    # a marker written for a smaller --n_episodes does not count as done
    if prev is not None and (keep is not None or (prev.get('n_episodes_requested') or 0) >= args.n_episodes):
        log(f'[fetch] {task}: already done (marker {marker})')
        return int(prev.get('episodes') or 0)
    if keep is not None and want is not None:
        if not want:
            log(f'[fetch] {task}: no wanted episodes in this archive, not opened')
            _mark(marker, task, 0, 0, args, keep, want)
            return 0
        have = {e for e in want if os.path.exists(os.path.join(out_task, e, 'data', 'trajectory.hdf5'))}
        if have == want:
            log(f'[fetch] {task}: all {len(want)} wanted episodes already on disk')
            _mark(marker, task, len(want), 0, args, keep, want)
            return len(want)

    parts = sorted(p for p in entries if os.path.basename(p).startswith(task + '.tar.gz.part-'))
    if not parts:                       # RuntimeError, not SystemExit: one bad name must not kill the other 166
        raise RuntimeError(f'no archive parts for task {task!r}; available: '
                           f'{sorted({os.path.basename(p).split(".tar.gz")[0] for p in entries})}')
    urls = [f'https://huggingface.co/datasets/{REPO}/resolve/main/{p}' for p in parts]
    log(f'[fetch] {task}: streaming {len(parts)} part(s)')
    stream = io.BufferedReader(_HttpParts(urls, args.token, retries=args.retries), buffer_size=1 << 22)
    done, seen, finished = set(), set(), None
    t0 = time.time()
    with tarfile.open(fileobj=stream, mode='r|gz') as tar:
        for m in tar:
            if not m.isfile():
                continue
            ep, tail = episode_of(m.name)
            if ep is None:
                continue
            # An archive is written episode by episode, so the first member of ANOTHER episode is the earliest
            # safe place to stop: breaking right after a trajectory.hdf5 would drop that episode's sidecar files.
            if finished is not None and ep != finished:
                break
            if keep is not None:                                          # keep-list mode: no episode budget
                if ep not in keep:
                    continue
            elif ep not in seen and len(seen) >= args.n_episodes:         # --n_episodes budget spent
                continue
            seen.add(ep)
            dst = os.path.join(out_task, ep, *tail)
            extract(tar, m, dst)
            if not m.name.endswith('trajectory.hdf5'):
                continue
            done.add(ep)
            if keep is None:
                log(f'[fetch]   {ep}: {dst} ({m.size / 1e6:.1f} MB; {stream.raw.total / 1e6:.0f} MB streamed)')
                if len(done) >= args.n_episodes:
                    finished = ep
            else:
                if len(done) % 50 == 0 or want is None:
                    log(f'[fetch]   {task}: {len(done)} episodes ({stream.raw.total / 1e9:.1f} GB streamed)')
                if want is not None and done >= want:                     # every wanted id seen: stop early
                    finished = ep
    gb, mins = stream.raw.total / 1e9, (time.time() - t0) / 60
    if want is not None and not done >= want:
        log(f'[fetch] {task}: WARNING {len(want - done)} wanted episode(s) were not in the archive, e.g. '
            f'{sorted(want - done)[:3]}')
    log(f'[fetch] {task}: {len(done)} episodes -> {out_task} ({gb:.1f} GB streamed, {mins:.0f} min)')
    _mark(marker, task, len(done), stream.raw.total, args, keep, want)
    return len(done)


def _marker(path):
    """The task's completion marker, or None when it is absent or unreadable - a half-written marker from a
    killed run must read as 'not done', never as 'done'."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        if os.path.exists(path):
            log(f'[fetch] WARNING unreadable marker {path} ({e}); treating the task as unfinished')
        return None


def _mark(marker, task, n, nbytes, args, keep=None, want=None):
    if args.dry_run:
        return
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    blob = {'task': task, 'episodes': n, 'bytes_streamed': nbytes, 'finished': time.time(),
            'mode': 'keep_list' if keep is not None else 'n_episodes',
            'n_episodes_requested': None if keep is not None else args.n_episodes,
            'wanted': None if want is None else len(want)}
    with open(marker + '.part', 'w') as f:                # atomic: a killed run must not leave a partial marker
        json.dump(blob, f)
    os.replace(marker + '.part', marker)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='benchmark1_1_compressed')
    ap.add_argument('--embodiment', default='h5_franka_3rgb')
    ap.add_argument('--tasks', nargs='+', default=None, help='task archive names (without .tar.gz.part-xx)')
    ap.add_argument('--all_tasks', action='store_true', help='every task archive of the split/embodiment')
    ap.add_argument('--keep_episodes', default=None,
                    help='JSON with the episode ids to keep (see load_keep_episodes); overrides --n_episodes')
    ap.add_argument('--n_episodes', type=int, default=3, help='episodes to extract per task (no --keep_episodes)')
    ap.add_argument('--workers', type=int, default=None, help='task archives streamed at once (default 8 with '
                                                              '--all_tasks, else 1)')
    ap.add_argument('--retries', type=int, default=3, help='attempts per archive part before giving up on the task')
    ap.add_argument('--force', action='store_true', help='re-stream tasks that already have a _done marker')
    ap.add_argument('--dry_run', action='store_true', help='list the archives and the wanted episode counts only')
    args = ap.parse_args()
    if not args.tasks and not args.all_tasks:
        raise SystemExit('give --tasks or --all_tasks')
    args.token = os.environ.get('HF_TOKEN') or get_token()

    keep, by_task = (load_keep_episodes(args.keep_episodes) if args.keep_episodes else (None, None))
    api = HfApi(token=args.token)
    prefix = f'{args.split}/{args.embodiment}/'
    entries = [e.path for e in api.list_repo_tree(REPO, path_in_repo=prefix.rstrip('/'), repo_type='dataset')]
    all_tasks = sorted({os.path.basename(p).split('.tar.gz')[0] for p in entries
                        if '.tar.gz.part-' in os.path.basename(p)})
    tasks = args.tasks if args.tasks else all_tasks
    if keep is not None:
        log(f'[fetch] keep list: {len(keep)} episode ids'
            + (f' across {len(by_task)} task archives' if by_task else ' (no per-task map: every archive is streamed)'))
    log(f'[fetch] {len(tasks)} task archive(s) of {prefix} ({len(all_tasks)} available)')

    if args.dry_run:
        for t in tasks:
            n = len(by_task.get(t, ())) if by_task else None
            log(f'[fetch]   {t}: {"?" if n is None else n} wanted episode(s)')
        return

    workers = args.workers or (8 if args.all_tasks else 1)
    failed, total = [], 0

    def run(task):
        want = set(by_task.get(task, ())) if by_task else None
        return fetch_task(task, entries, args, keep, want)

    if workers <= 1:
        for t in tasks:
            try:
                total += run(t)
            except Exception as e:  # noqa: BLE001
                failed.append(t)
                log(f'[fetch] ERROR {t}: {e}')
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run, t): t for t in tasks}
            for fu in as_completed(futs):
                try:
                    total += fu.result()
                except Exception as e:  # noqa: BLE001
                    failed.append(futs[fu])
                    log(f'[fetch] ERROR {futs[fu]}: {e}')
    log(f'[fetch] {total} episodes under {args.out}; {len(failed)} task(s) failed'
        + (f': {failed[:10]}' if failed else '') + ' (re-run to resume)')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
