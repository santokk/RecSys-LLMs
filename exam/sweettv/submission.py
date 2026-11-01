"""Submission row rules and candidate-file checks."""
import re

HEADER = "user_id,tv_show_id"


def validate_row(ids, pool=None, k=5):
    ids = list(ids)
    if len(ids) != k:
        raise ValueError(f"expected exactly {k} ids, got {len(ids)}")
    if len(set(ids)) != k:
        raise ValueError(f"duplicate ids in {ids}")
    if any(not isinstance(i, int) or i <= 0 for i in ids):
        raise ValueError(f"ids must be positive integers: {ids}")
    if pool is not None and not set(ids) <= set(pool):
        raise ValueError(f"ids outside the candidate pool: {ids}")
    return ids


def format_row(user_id, ids, pool=None, k=5):
    if not isinstance(user_id, str):
        raise TypeError("user ids must stay strings")
    return f"{user_id},{' '.join(str(i) for i in validate_row(ids, pool, k))}"


def render_csv(user_ids, lists, pool, k=5):
    """The full file: template user order, one row each, trailing newline."""
    rows = [format_row(user_id, ids, pool, k) for user_id, ids in zip(user_ids, lists, strict=True)]
    return "\n".join([HEADER, *rows]) + "\n"


def validate_csv(text, user_ids, pool, k=5):
    """Check a finished file against the template byte by byte of structure."""
    if not text.endswith("\n") or text.endswith("\n\n") or "\r" in text:
        raise ValueError("file must end with exactly one newline and use LF")
    lines = text[:-1].split("\n")
    if lines[0] != HEADER:
        raise ValueError(f"bad header: {lines[0]!r}")
    rows = lines[1:]
    if len(rows) != len(user_ids):
        raise ValueError(f"expected {len(user_ids)} rows, got {len(rows)}")
    pattern = re.compile(r"^(\d+),([1-9]\d*(?: [1-9]\d*){%d})$" % (k - 1))
    for expected, row in zip(user_ids, rows):
        match = pattern.match(row)
        if match is None:
            raise ValueError(f"malformed row: {row!r}")
        if match.group(1) != expected:
            raise ValueError(f"row order differs from the template at {row!r}")
        validate_row([int(i) for i in match.group(2).split(" ")], pool, k)
