"""Recursive numeric comparison of JSON-like structures."""

import json
import math


def cmp(a, b, path="", bad=None):
    bad = [] if bad is None else bad
    a = json.loads(json.dumps(a))
    if isinstance(b, dict):
        if not isinstance(a, dict):
            bad.append((path, "type"))
            return bad
        for k in b:
            if k not in a:
                bad.append((path + "/" + k, "missing"))
                continue
            cmp(a[k], b[k], path + "/" + k, bad)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            bad.append((path, "len"))
            return bad
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, f"{path}[{i}]", bad)
    elif isinstance(b, float):
        if not (
            isinstance(a, (int, float))
            and (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b)))
        ):
            bad.append((path, a, b))
    elif a != b:
        bad.append((path, a, b))
    return bad
