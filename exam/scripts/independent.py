"""Second implementation shared by the verification scripts.

Imports none of sweettv.timeutil, schedule, matching, split, baseline, metrics,
evaluation, mf, ppr or pipeline. It has its own offset rule, bisect matching,
dict aggregation, ranking, MAP loop, ALS solver (Cholesky) and an exact linear
solve for PageRank, so agreement is evidence rather than repetition.
"""
import bisect
import csv
import hashlib
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime

import numpy as np

from sweettv.config import EXAM_DIR, REPO_ROOT, data_dir

EPOCH = datetime(1970, 1, 1)
JUMP = datetime(2020, 3, 29, 3, 0, 0)
LOCAL_PATH = re.compile("/" + "Users/|/" + "home/|/" + "private/|/" + "tmp/|[A-Z]:" + r"\\")
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt"}

checks = []


def check(name, passed, detail=None):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    print(("PASS " if passed else "FAIL ") + name + (f" :: {detail}" if detail is not None else ""), flush=True)


def to_utc(wall):
    offset = 7200 if wall < JUMP else 10800
    return int((wall - EPOCH).total_seconds()) - offset


def view_time(text):
    return to_utc(datetime.fromisoformat(text))


def schedule_time(text):
    return to_utc(datetime.fromisoformat(f"{text[6:10]}-{text[3:5]}-{text[:2]}{text[10:]}"))


def check_sources(cfg):
    for name, expected in cfg["expected_sha256"].items():
        digest = hashlib.sha256()
        with open(data_dir(cfg) / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                digest.update(block)
        check(f"source unchanged: {name}", digest.hexdigest() == expected)


def read_timeline(cfg, upper):
    """Own overlap resolution; returns labelled airings by channel."""
    by_channel = defaultdict(list)
    for role in ("schedule_train", "schedule_validation"):
        with open(data_dir(cfg) / cfg["files"][role], newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                start, length = schedule_time(row[2]), int(row[3])
                if length > 0 and start < upper:
                    by_channel[int(row[0])].append((start, -length, -int(row[5])))
    labelled = {}
    for channel, rows in by_channel.items():
        rows.sort()
        unique = [r for i, r in enumerate(rows) if i == 0 or r[0] != rows[i - 1][0]]
        starts, ends, ids = [], [], []
        for i, (start, negative_length, negative_id) in enumerate(unique):
            end = start - negative_length
            if i + 1 < len(unique):
                end = min(end, unique[i + 1][0])
            if negative_id:
                starts.append(start), ends.append(end), ids.append(-negative_id)
        labelled[channel] = (starts, ends, ids)
    return labelled


def rederive_interactions(cfg, lower, middle, upper):
    """Per-(user, airing, window) overlap from the raw CSVs."""
    labelled = read_timeline(cfg, upper)
    base, offset = {}, 0
    for channel in sorted(labelled):
        base[channel] = offset
        offset += len(labelled[channel][0])
    airing_id, airing_length = [0] * offset, [0] * offset
    for channel, (starts, ends, ids) in labelled.items():
        for i in range(len(starts)):
            airing_id[base[channel] + i] = ids[i]
            airing_length[base[channel] + i] = ends[i] - starts[i]

    users, totals = {}, defaultdict(int)
    train_channels = set()
    rows, latest_stop, latest_start = 0, 0, 0
    with open(data_dir(cfg) / cfg["files"]["view_log"], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for user_id, channel, start_text, _, seconds in reader:
            rows += 1
            channel = int(channel)
            user = users.setdefault(user_id, len(users))
            begin = view_time(start_text)
            finish = begin + int(seconds)
            latest_stop, latest_start = max(latest_stop, finish), max(latest_start, begin)
            if begin < middle:
                train_channels.add(channel)
            if channel not in labelled:
                continue
            starts, ends, _ = labelled[channel]
            for window, (low, high) in enumerate(((lower, middle), (middle, upper))):
                a, b = max(begin, low), min(finish, high)
                if a >= b:
                    continue
                i = bisect.bisect_right(ends, a)
                while i < len(starts) and starts[i] < b:
                    totals[(user << 1 | window, base[channel] + i)] += min(b, ends[i]) - max(a, starts[i])
                    i += 1
    return {"labelled": labelled, "airing_id": airing_id, "airing_length": airing_length, "users": users,
            "totals": totals, "train_channels": train_channels, "rows": rows,
            "latest_stop": latest_stop, "latest_start": latest_start, "middle": middle, "upper": upper}


def watched_pairs(core, seconds, fraction):
    """[train, validation] maps (user, item) -> summed capped seconds, plus airing rows."""
    sides, airing_rows = [defaultdict(int), defaultdict(int)], [0, 0]
    for (user_window, airing), overlap in core["totals"].items():
        length = core["airing_length"][airing]
        capped = min(overlap, length)
        if capped >= min(seconds, fraction * length):
            window = user_window & 1
            sides[window][(user_window >> 1, core["airing_id"][airing])] += capped
            airing_rows[window] += 1
    return sides, airing_rows


def rederive_pool(core):
    """Ids airing in the validation window on a channel viewed in training."""
    return sorted({i for channel, (starts, ends, ids) in core["labelled"].items() if channel in core["train_channels"]
                   for s, e, i in zip(starts, ends, ids) if s < core["upper"] and e > core["middle"]})


def read_cohort_users(cfg, users):
    with open(data_dir(cfg) / cfg["files"]["submission_template"], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [users[row[0]] for row in reader]


def top5_targets(validation, cohort, pool):
    allowed = set(pool)
    seconds_of = defaultdict(list)
    for (user, item), seconds in validation.items():
        if item in allowed:
            seconds_of[user].append((-seconds, item))
    return {user: {item for _, item in sorted(seconds_of[user])[:5]} for user in cohort}


def popularity_order(train, pool):
    watchers = defaultdict(int)
    for _, item in train:
        watchers[item] += 1
    return sorted(pool, key=lambda item: (-watchers[item], item)), {item: watchers[item] for item in pool}


def map_at_5(lists, targets, users):
    total, nonempty = 0.0, 0
    for user in users:
        wanted = targets[user]
        if not wanted:
            continue
        nonempty += 1
        hits, score = 0, 0.0
        for rank, item in enumerate(lists[user], 1):
            if item in wanted:
                hits += 1
                score += hits / rank
        total += score / min(5, len(wanted))
    return total / len(users), total / nonempty


def _as_groups(train):
    """Sorted users and items, plus (row, column, seconds) triples."""
    users = sorted({user for user, _ in train})
    items = sorted({item for _, item in train})
    user_row = {user: row for row, user in enumerate(users)}
    item_row = {item: row for row, item in enumerate(items)}
    triples = [(user_row[user], item_row[item], seconds) for (user, item), seconds in train.items()]
    return users, items, triples


def als(train, params, seed, init_std, scale):
    """Weighted ALS with Cholesky solves, written without reference to sweettv.mf."""
    users, items, triples = _as_groups(train)
    factors, alpha, reg = params["factors"], params["confidence_alpha"], params["regularization"]
    by_user, by_item = defaultdict(list), defaultdict(list)
    for row, column, seconds in triples:
        weight = 1.0 + alpha * math.log1p(seconds / scale)
        by_user[row].append((column, weight))
        by_item[column].append((row, weight))

    def sweep(fixed, groups, count):
        base = fixed.T @ fixed + reg * np.eye(factors)
        result = np.zeros((count, factors))
        for row in range(count):
            pairs = sorted(groups[row])
            index = [p[0] for p in pairs]
            weight = np.array([p[1] for p in pairs])
            chosen = fixed[index]
            matrix = base + (chosen * (weight - 1.0)[:, None]).T @ chosen
            lower = np.linalg.cholesky(matrix)
            result[row] = np.linalg.solve(lower.T, np.linalg.solve(lower, chosen.T @ weight))
        return result

    item_factors = np.random.default_rng(seed).normal(0.0, init_std, (len(items), factors))
    for _ in range(params["iterations"]):
        user_factors = sweep(item_factors, by_user, len(users))
        item_factors = sweep(user_factors, by_item, len(items))
    return users, items, user_factors, item_factors


def exact_pagerank(train, cohort, params, scale):
    """Solve the PageRank system directly, with no power iteration.

    With dangling mass returned to s, p is proportional to the solution x of
    (I - d Q^T) x = (1 - d) s, where Q is the sub-stochastic transition matrix.
    """
    users, items, triples = _as_groups(train)
    weighting, d = params["interaction_weighting"], params["link_probability"]
    strength = {"binary": lambda s: 1.0, "log_seconds": lambda s: math.log1p(s / scale), "seconds": lambda s: s / scale}[weighting]
    rows_of = defaultdict(list)
    for row, column, seconds in triples:
        rows_of[row].append((column, strength(seconds)))
    weights = np.zeros((len(items), len(items)))
    for pairs in rows_of.values():
        index = np.array([p[0] for p in pairs])
        value = np.array([p[1] for p in pairs])
        weights[np.ix_(index, index)] += np.outer(value, value) / value.sum()
    np.fill_diagonal(weights, 0.0)
    totals = weights.sum(axis=1)
    substochastic = np.zeros_like(weights)
    active = totals > 0
    substochastic[active] = weights[active] / totals[active, None]
    position = {user: row for row, user in enumerate(users)}
    known = [user for user in cohort if user in position]
    seeds = np.zeros((len(items), len(known)))
    for column, user in enumerate(known):
        for row, value in rows_of[position[user]]:
            seeds[row, column] = value
    seeds /= seeds.sum(axis=0, keepdims=True)
    solved = np.linalg.solve(np.eye(len(items)) - d * substochastic.T, (1.0 - d) * seeds)
    return items, known, solved / solved.sum(axis=0, keepdims=True)


def lists_from(items, known, scores, pool, fallback, cohort):
    """Top-5 by score then id, restricted to the pool, topped up from fallback."""
    members = set(pool)
    allowed = [(position, item) for position, item in enumerate(items) if item in members]
    columns = np.array([p for p, _ in allowed])
    ids = [item for _, item in allowed]
    lists = {}
    for row, user in enumerate(known):
        values = scores[row, columns]
        ranked = sorted(range(len(ids)), key=lambda c: (-values[c], ids[c]))[:5]
        lists[user] = [ids[c] for c in ranked if values[c] > 1e-15 or values[c] < 0]
    finished = {}
    for user in cohort:
        chosen = list(lists.get(user, []))
        for item in fallback:
            if len(chosen) == 5:
                break
            if item not in chosen:
                chosen.append(item)
        finished[user] = chosen
    return finished


def fused_lists(items, known, scores, pool, order, weight, constant, cohort):
    """Reciprocal-rank fusion in plain Python, ranking the whole pool per user."""
    position = {item: column for column, item in enumerate(items)}
    popular = {item: rank for rank, item in enumerate(order, 1)}
    finished = {user: order[:5] for user in cohort}
    for row, user in enumerate(known):
        values = {item: (scores[row, position[item]] if item in position else 0.0) for item in pool}
        values = {item: (value if value > 1e-15 else -1.0) for item, value in values.items()}
        ranked = sorted(pool, key=lambda item: (-values[item], item))
        rank_of = {item: rank for rank, item in enumerate(ranked, 1)}
        fused = {item: weight / (constant + rank_of[item]) + (1 - weight) / (constant + popular[item]) for item in pool}
        finished[user] = sorted(pool, key=lambda item: (-fused[item], item))[:5]
    return finished


def reproduce(cfg, lower, middle, upper, mf_params, ppr_params):
    """Rebuild every list and primary score for one train/validation split."""
    core = rederive_interactions(cfg, lower, middle, upper)
    (train, validation), _ = watched_pairs(core, cfg["watched"]["primary_seconds"], cfg["watched"]["airing_fraction"])
    pool = rederive_pool(core)
    cohort = read_cohort_users(cfg, core["users"])
    targets = top5_targets(validation, cohort, pool)
    order, _ = popularity_order(train, pool)
    scale = cfg["models"]["strength_scale_seconds"]

    users, items, user_factors, item_factors = als(train, mf_params, cfg["seed"], cfg["models"]["mf"]["init_std"], scale)
    row = {user: position for position, user in enumerate(users)}
    known = [user for user in cohort if user in row]
    mf_scores = user_factors[[row[user] for user in known]] @ item_factors.T
    ppr_items, ppr_known, probabilities = exact_pagerank(train, cohort, ppr_params, scale)
    lists = {
        "popularity": {user: order[:5] for user in cohort},
        "mf": lists_from(items, known, mf_scores, pool, order, cohort),
        "ppr": lists_from(ppr_items, ppr_known, probabilities.T, pool, order, cohort),
    }
    scores = {method: map_at_5(lists[method], targets, cohort) for method in lists}
    return {"core": core, "train": train, "validation": validation, "pool": pool, "cohort": cohort, "order": order,
            "targets": targets, "lists": lists, "scores": scores, "ppr_items": ppr_items, "ppr_known": ppr_known,
            "probabilities": probabilities}


def rederive_test_pool(cfg, eligible):
    """Own availability read: labelled airings overlapping the test window."""
    start = view_time(cfg["final"]["test_window"]["start"])
    end = view_time(cfg["final"]["test_window"]["end"])
    pool = set()
    for role in ("schedule" + "_test", "schedule_validation"):
        with open(data_dir(cfg) / cfg["files"][role], newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                identifier, length = int(row[5]), int(row[3])
                if identifier and length > 0 and int(row[0]) in eligible:
                    begins = schedule_time(row[2])
                    if begins < end and begins + length > start:
                        pool.add(identifier)
    return sorted(pool)


def airing_starts(core):
    """Airing start per global airing index, in the same order as the totals keys."""
    starts = []
    for channel in sorted(core["labelled"]):
        starts.extend(core["labelled"][channel][0])
    return starts


def repeat_score_map(core, seconds, fraction, half_life, prediction_start):
    """Training-side (user, item) repeat scores, computed row by row."""
    starts = airing_starts(core)
    scores = defaultdict(float)
    latest = 0
    for (user_window, airing), overlap in core["totals"].items():
        if user_window & 1:
            continue
        length = core["airing_length"][airing]
        capped = min(overlap, length)
        if capped >= min(seconds, fraction * length):
            latest = max(latest, starts[airing])
            age_days = (prediction_start - starts[airing]) / 86400
            weight = capped if half_life is None else capped * math.exp(-math.log(2) * age_days / half_life)
            scores[(user_window >> 1, core["airing_id"][airing])] += weight
    return scores, latest


def repeat_lists(scores, cohort, pool, order):
    """Top-5 watched pool ids by score then id, topped up from popularity."""
    allowed = set(pool)
    seen = defaultdict(list)
    for (user, item), value in scores.items():
        if item in allowed:
            seen[user].append((-value, item))
    finished = {}
    for user in cohort:
        chosen = [item for _, item in sorted(seen[user])[:5]]
        for item in order:
            if len(chosen) == 5:
                break
            if item not in chosen:
                chosen.append(item)
        finished[user] = chosen
    return finished


def repeat_fused_lists(scores, cohort, pool, order, weight, constant):
    """Fusion of repeat scores with popularity through the plain-Python ranker."""
    column = {item: index for index, item in enumerate(pool)}
    wanted = set(cohort)
    known = sorted({user for (user, item) in scores if user in wanted and item in column})
    row = {user: index for index, user in enumerate(known)}
    matrix = np.zeros((len(known), len(pool)))
    for (user, item), value in scores.items():
        if user in row and item in column:
            matrix[row[user], column[item]] = value
    return fused_lists(pool, known, matrix, pool, order, weight, constant, cohort)


def check_candidate_file(path, template_users, pool, lists):
    """Parse one candidate file by hand against the template and expected lists."""
    raw = path.read_bytes()
    text = raw.decode("ascii")
    name = path.name
    check(f"{name}: header and exactly one final newline", text.startswith("user_id,tv_show_id\n") and text.endswith("\n") and not text.endswith("\n\n"))
    check(f"{name}: no CR, NaN, exponent or decimal point", not re.search(r"[\reE.]|nan|NaN", text.split("\n", 1)[1]))
    rows = text[:-1].split("\n")[1:]
    check(f"{name}: 1,260 rows", len(rows) == 1260, len(rows))
    parts = [row.split(",") for row in rows]
    check(f"{name}: exactly two cells per row", all(len(p) == 2 for p in parts))
    check(f"{name}: exact template user order", [p[0] for p in parts] == template_users)
    ids = [p[1].split(" ") for p in parts]
    check(f"{name}: five positive integer ids per row", all(len(r) == 5 and all(i.isdigit() and i[0] != "0" for i in r) for r in ids))
    numbers = [[int(i) for i in r] for r in ids]
    check(f"{name}: five distinct ids per row", all(len(set(r)) == 5 for r in numbers))
    check(f"{name}: only test-pool ids", all(set(r) <= pool for r in numbers))
    same = sum(row == lists[user] for row, user in zip(numbers, template_users))
    check(f"{name}: all rows equal the independent recompute", same == 1260, f"{same}/1260")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def check_test_boundary_sources(cfg):
    """Only the guarded accessor in csvio.py may name the test schedule."""
    role, name = "schedule" + "_test", cfg["files"]["schedule" + "_test"]
    scanned = [*EXAM_DIR.glob("sweettv/*.py"), *EXAM_DIR.glob("scripts/0[1-35-8]_*.py"), *EXAM_DIR.glob("scripts/1[0-4]_*.py")]
    offenders = [p.name for p in scanned if name in p.read_text(encoding="utf-8")
                 or (role in p.read_text(encoding="utf-8") and p.name != "csvio.py")]
    check("only csvio.py names the test schedule", not offenders, offenders)


def check_repository(cfg):
    offenders = []
    for path in EXAM_DIR.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts:
            if LOCAL_PATH.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    check("no absolute local paths under exam/", not offenders, offenders)
    git = lambda *args: subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    check("final submission path is not ignored", git("check-ignore", "exam/submission/final_submission.csv").returncode == 1)
    check("generated arrays are ignored", git("check-ignore", "exam/results/x.npz").returncode == 0)
    check("raw data is ignored inside exam/", git("check-ignore", "exam/dataset11-30.csv").returncode == 0)
    check("nothing is staged", git("diff", "--cached", "--name-only").stdout.strip() == "")
    check("git diff --check", git("diff", "--check").stdout.strip() == "")
    new_files = git("ls-files", "--others", "--exclude-standard", "exam").stdout.split()
    whitespace = [p for p in new_files if git("diff", "--no-index", "--check", "/dev/null", p).stdout.strip()]
    check("untracked exam files pass whitespace check", not whitespace, whitespace)
    biggest = max((REPO_ROOT / p).stat().st_size for p in new_files)
    check("no large file under exam/", biggest < 1 << 20, biggest)
