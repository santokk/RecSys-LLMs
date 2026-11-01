"""Streaming CSV access. Sources are only ever opened for reading."""
import csv

from .config import source_path

VIEW_HEADER = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]
SCHEDULE_HEADER = [
    "channel_id", "channel_title", "start_time", "duration", "tv_show_title",
    "tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
    "tv_show_genre_3", "year_of_production", "director", "actors",
]
SUBMISSION_HEADER = ["user_id", "tv_show_id"]
FIT_SCHEDULE_ROLES = ("schedule_train", "schedule_validation")

csv.field_size_limit(1 << 30)

# Parsed sources for this process. Stored in every artifact manifest so a
# reader of the artifact can see that the test schedule was never parsed.
parsed_sources = set()


def iter_rows(path, header):
    parsed_sources.add(path.name)
    with open(path, "r", newline="", encoding="utf-8", errors="strict") as f:
        reader = csv.reader(f)
        found = next(reader)
        if found != header:
            raise ValueError(f"{path.name}: unexpected header {found}")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{path.name}: bad row {row[:4]}")
            yield row


def iter_view_chunks(cfg):
    """Yield view rows in bounded lists, renaming `duraton` to `duration_sec`."""
    chunk = []
    for user_id, channel, start, stop, duration_sec in iter_rows(source_path(cfg, "view_log"), VIEW_HEADER):
        chunk.append((user_id, int(channel), start, stop, int(duration_sec)))
        if len(chunk) == cfg["chunk_rows"]:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def iter_fit_schedule(cfg, role):
    """Yield schedule rows for fitting. The test schedule is refused here."""
    if role not in FIT_SCHEDULE_ROLES:
        raise PermissionError(f"{role} is not available for fitting")
    yield from iter_rows(source_path(cfg, role), SCHEDULE_HEADER)


def test_schedule_name(cfg):
    return cfg["files"]["schedule_test"]


def require_test_schedule_unread(cfg):
    """Stop any fit in a process that has already parsed the test schedule."""
    if test_schedule_name(cfg) in parsed_sources:
        raise PermissionError("the test schedule was parsed in this process, so fitting is refused")


def iter_test_availability(cfg, refit_manifest):
    """Yield labelled test-period airings, availability and lookup fields only.

    Requires the finished refit manifest, so models exist before this file is
    opened. Rows carry channel, time, id, title and category. There is no user
    field in this file, and nothing returned here can reach a fit.
    """
    if refit_manifest.get("stage") != "refit" or test_schedule_name(cfg) in refit_manifest.get("parsed_sources", ()):
        raise PermissionError("the test schedule opens only after a clean full refit")
    for row in iter_rows(source_path(cfg, "schedule_test"), SCHEDULE_HEADER):
        show_id = int(row[5])
        if show_id:
            yield {"channel": int(row[0]), "start_text": row[2], "duration": int(row[3]), "show_id": show_id,
                   "title": row[4], "category": row[6]}


def read_cohort(cfg):
    """Evaluation user ids as strings, in template order."""
    return [row[0] for row in iter_rows(source_path(cfg, "submission_template"), SUBMISSION_HEADER)]
