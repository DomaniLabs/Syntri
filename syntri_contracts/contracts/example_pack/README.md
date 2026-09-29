# Example pack: `check_balance`

The smallest useful capability pack. A user asks for their balance; the pack
asks the backend to fetch it, then replies with the figure. It moves no money
and needs no confirmation, so it is the ideal template to copy.

```
example_pack/
├── __init__.py          # exports CheckBalancePack
├── pack.py              # the pack itself
├── manifest.yaml        # human-readable copy of the manifest
├── fixtures/
│   └── 01-check-balance-basic.json
└── README.md            # this file
```

## What it does

1. **`tools()`** exposes one tool, `balance.fetch`, executed by the **backend**
   (never the agent) with `moves_money=False`.
2. **`advance()`** has two branches:
   - No tool result yet → return a `request_tool` `Decision` asking the backend
     to run `balance.fetch`.
   - A successful result → return a `reply` `Decision` with the formatted
     balance (e.g. `Your balance is ₦5,000.`).
3. **`fixtures()`** loads every JSON file in `fixtures/` so
   `syntri pack validate` can replay the pack.

## Copy it

```bash
cp -r contracts/example_pack my_pack
```

Then edit `manifest.yaml` / the `manifest` in `pack.py`, add your tools, and
write your own `advance()` logic. See the top-level
[`contracts/README.md`](../README.md) for the full guide.
