"""Usage ledger: one JSON line per model call, frontier and SLM alike, and the frontier cost computed from it."""
import json
import threading
import time
from pathlib import Path

# $ per million tokens: uncached input, cached input, output. Override with Ledger(prices=...) or --prices.
DEFAULT_PRICES = {'gpt-6.1-sol': (2.0, 0.10, 10.0)}


class Ledger:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.rows = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, role, model, usage, seconds, **extra):
        row = {'time': round(time.time(), 3), 'role': role, 'model': model, 'seconds': round(seconds, 3),
               'input_tokens': usage.get('input_tokens'), 'cached_input_tokens': usage.get('cached_input_tokens'),
               'output_tokens': usage.get('output_tokens'), **extra}
        with self.lock:
            self.rows.append(row)
            if self.path:
                with self.path.open('a') as f:
                    f.write(json.dumps(row) + '\n')
        return row


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cost(rows, prices=None, role='frontier'):
    """Dollar cost of the calls with the given role: uncached input, cached input and output (incl. reasoning)."""
    prices = {**DEFAULT_PRICES, **(prices or {})}
    total = 0.0
    for r in rows:
        if r.get('role') != role:
            continue
        p = prices.get(r.get('model'))
        if p is None:
            raise KeyError(f'no price for model {r.get("model")!r}; pass prices')
        cached = r.get('cached_input_tokens') or 0
        uncached = (r.get('input_tokens') or 0) - cached
        total += (uncached * p[0] + cached * p[1] + (r.get('output_tokens') or 0) * p[2]) / 1e6
    return total
