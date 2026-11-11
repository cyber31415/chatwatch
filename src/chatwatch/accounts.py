"""Account age from a Twitch user-id, with no API call.

Twitch assigns user ids in creation order (research/01 §6: 125/125 sampled
ids monotonic, 2007-2026). Interpolating between dated anchors gives creation
time to within days for post-2019 accounts. The counter jumped from ~280.8M
to ~400.0M in early Dec 2018; nothing lives in the gap, so it is clamped.

Anchors were looked up 2026-10-20. Past the newest anchor we extrapolate at
~0.45M ids/day, which is what research/01 measured for mid/late 2026. The
head goes stale, so refresh the last rows occasionally (or use Helix
GET /users with an app token for exact dates).
"""
import bisect
import datetime as dt

_ANCHORS = [
    (12_826, "2007-05-22"), (42_000, "2007-07-11"), (17_466_000, "2010-11-06"),
    (27_357_000, "2012-01-10"), (38_957_000, "2013-01-02"), (56_638_000, "2014-02-12"),
    (80_496_000, "2015-01-21"), (95_108_000, "2015-07-03"), (115_443_000, "2016-02-10"),
    (129_196_000, "2016-07-10"), (146_423_000, "2017-01-29"), (166_176_000, "2017-07-22"),
    (192_523_000, "2018-01-24"), (238_676_000, "2018-07-14"), (280_000_000, "2018-12-01"),
    (280_800_000, "2018-12-02"), (400_000_000, "2018-12-04"), (405_000_000, "2018-12-28"),
    (433_971_000, "2019-05-04"), (472_077_000, "2019-11-10"), (510_355_000, "2020-04-07"),
    (551_515_000, "2020-07-06"), (597_364_000, "2020-10-18"), (638_552_000, "2021-01-19"),
    (682_088_000, "2021-04-30"), (715_223_000, "2021-08-07"), (762_720_000, "2022-01-16"),
    (793_566_000, "2022-05-07"), (835_883_000, "2022-10-08"), (886_327_000, "2023-02-28"),
    (929_878_000, "2023-07-04"), (995_140_000, "2023-11-23"), (1_019_993_000, "2024-01-18"),
    (1_097_812_000, "2024-06-11"), (1_169_154_000, "2024-10-24"), (1_239_503_000, "2025-01-26"),
    (1_310_276_000, "2025-05-13"), (1_382_686_000, "2025-10-20"), (1_433_402_000, "2026-01-26"),
    (1_493_335_000, "2026-05-05"), (1_528_123_000, "2026-08-12"), (1_566_873_000, "2026-10-20"),
]
_IDS = [a for a, _ in _ANCHORS]
_TS = [dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp() for _, d in _ANCHORS]
IDS_PER_DAY_NOW = 450_000


def created_ts(uid):
    """Estimated creation time (unix seconds) for a numeric user id, or None."""
    try:
        x = int(uid)
    except (TypeError, ValueError):
        return None
    if x <= _IDS[0]:
        return _TS[0]
    if x >= _IDS[-1]:
        return _TS[-1] + (x - _IDS[-1]) / IDS_PER_DAY_NOW * 86400
    i = bisect.bisect_right(_IDS, x)
    a, b = _IDS[i - 1], _IDS[i]
    ta, tb = _TS[i - 1], _TS[i]
    return ta + (tb - ta) * (x - a) / (b - a)


def age_days(uid, now_ms):
    """Account age in days at time now_ms; None if the id isn't numeric.
    Never negative (extrapolation past the newest anchor can overshoot)."""
    c = created_ts(uid)
    if c is None:
        return None
    return max(0.0, (now_ms / 1000 - c) / 86400)
