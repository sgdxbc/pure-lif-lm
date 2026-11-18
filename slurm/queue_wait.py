"""How much time this account's GPU jobs spent waiting for the scheduler, from SLURM's accounting records. On the
cluster (login node is fine, no torch needed):

  sacct -u $USER -S 2026-11-15T00:00 -E now -X -P -n \
        -o JobID,JobName%60,ReqTRES%120,Submit,Eligible,Start,End,State > /tmp/$USER-sacct.txt
  python3 slurm/queue_wait.py /tmp/$USER-sacct.txt [--since 2026-11-17T00:00]

A job's wait is from when it could start (Eligible: for a relay part queued with --dependency, the end of the part
before it; otherwise its submission) to when it started; a job still pending has waited until now. Each wait is split
by what our own jobs held at the time: "own limit full" while our running jobs on that GPU type were already at the
per-user limit (QOS normal MaxTRESPerUser, LIMITS below), so the job queued behind our own work; "slot free" while
we had a free slot of that type, so it waited for the cluster (busy cards, or other users ahead by fair-share).
Jobs cancelled before they started are counted apart. Run time is Start to End (or now).
"""
import argparse, re, sys
from collections import defaultdict
from datetime import datetime

LIMITS = {"h100-96": 1, "a100-80": 1, "h100-47": 2, "a100-40": 2, "h200-141": 1}


def ts(s):
    return None if s in ("", "Unknown", "None") else datetime.fromisoformat(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sacct")
    p.add_argument("--since", default="", help="count only waiting and running time after this time")
    p.add_argument("--now", default="", help="the time to count pending and running jobs up to (default: now)")
    a = p.parse_args()
    now = datetime.fromisoformat(a.now) if a.now else datetime.now()
    since = datetime.fromisoformat(a.since) if a.since else None
    jobs = []
    for line in open(a.sacct):
        f = line.rstrip("\n").split("|")
        if len(f) < 8 or "." in f[0]:
            continue
        jid, name, tres, sub, elig, start, end, state = f[:8]
        m = re.search(r"gres/gpu:([\w-]+)=(\d+)", tres)
        if not m:
            continue   # CPU-only jobs
        gpu = m.group(1)
        sub, elig, start, end = ts(sub), ts(elig), ts(start), ts(end)
        elig = elig or sub
        if start and start > now:
            start = None   # an estimated start of a pending job
        jobs.append(dict(id=jid, name=name, gpu=gpu, elig=elig, start=start, end=end, state=state.split()[0]))

    def clip(t0, t1):
        if since:
            t0 = max(t0, since)
        return (t0, t1) if t1 > t0 else None

    # our running intervals per GPU type
    runs = defaultdict(list)
    for j in jobs:
        if j["start"]:
            runs[j["gpu"]].append((j["start"], j["end"] or now))

    def busy(gpu, t0, t1):
        """Hours of [t0, t1) during which our running jobs of this type were at the per-user limit."""
        lim = LIMITS.get(gpu, 10 ** 9)
        ev = [(t0, 0)]
        for s, e in runs[gpu]:
            s, e = max(s, t0), min(e, t1)
            if e > s:
                ev += [(s, 1), (e, -1)]
        ev.sort()
        n, last, full = 0, t0, 0.0
        for t, d in ev:
            if n >= lim:
                full += (t - last).total_seconds()
            n += d; last = t
        if n >= lim:
            full += (t1 - last).total_seconds()
        return full / 3600

    tot = defaultdict(lambda: defaultdict(float))
    waits = defaultdict(list)
    per_run = defaultdict(lambda: [0.0, 0.0, 0])
    for j in jobs:
        g = tot[j["gpu"]]
        if j["start"] is None and j["state"].startswith("CANCELLED"):
            iv = clip(j["elig"], j["end"] or now)
            if iv:
                g["cancelled_waiting_h"] += (iv[1] - iv[0]).total_seconds() / 3600
                g["cancelled_n"] += 1
            continue
        w_end = j["start"] or now
        iv = clip(j["elig"], w_end)
        if iv:
            w = (iv[1] - iv[0]).total_seconds() / 3600
            full = busy(j["gpu"], *iv)
            g["wait_h"] += w; g["wait_own_h"] += full
            waits[j["gpu"]].append(w)
            per_run[j["name"]][1] += w
        if j["start"] is None:
            g["pending_n"] += 1
            continue
        iv = clip(j["start"], j["end"] or now)
        if iv:
            r = (iv[1] - iv[0]).total_seconds() / 3600
            g["run_h"] += r; per_run[j["name"]][0] += r; per_run[j["name"]][2] += 1
            if j["state"] in ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT") and r < 0.5:
                g["failed_short_n"] += 1
        g["started_n"] += 1

    span = f" since {since}" if since else ""
    print(f"GPU jobs{span}, up to {now:%Y-%m-%d %H:%M} (hours)")
    print(f"{'GPU':9} {'jobs':>5} {'running':>8} {'waiting':>8} {'own limit full':>15} {'slot free':>10} "
          f"{'median wait':>12} {'max wait':>9} {'pending':>8} {'cancelled (waited)':>19}")
    S = defaultdict(float)
    for gpu in sorted(tot):
        g, w = tot[gpu], sorted(waits[gpu])
        med = w[len(w) // 2] if w else 0.0
        print(f"{gpu:9} {int(g['started_n']):5d} {g['run_h']:8.1f} {g['wait_h']:8.1f} {g['wait_own_h']:15.1f} "
              f"{g['wait_h'] - g['wait_own_h']:10.1f} {med:12.2f} {max(w, default=0):9.1f} {int(g['pending_n']):8d} "
              f"{int(g['cancelled_n']):6d} ({g['cancelled_waiting_h']:.1f} h)")
        for k in ("run_h", "wait_h", "wait_own_h"):
            S[k] += g[k]
    print(f"{'all':9} {'':5} {S['run_h']:8.1f} {S['wait_h']:8.1f} {S['wait_own_h']:15.1f} "
          f"{S['wait_h'] - S['wait_own_h']:10.1f}")
    print("\nper run (job name): parts started, hours running, hours waiting (sorted by waiting)")
    for name, (r, w, n) in sorted(per_run.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"  {name:40} {n:3d} {r:7.1f} {w:7.1f}")


if __name__ == "__main__":
    main()
