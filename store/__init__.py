"""Store backends: the append-only experience store, as the engine sees it.

`Store` (in `base.py`) is the contract. Everything above the storage layer —
capture, the policy audit logger, the corpus builder, the training pipeline —
is written against it and against nothing else, so choosing a backend is one
line at startup and is invisible afterwards.

This public package ships one backend:

    JsonlStore  — store/jsonl.py
    ------------------------------
    One append-only file per record type, written atomically. No database,
    no schema, no migration, nothing to operate. The right answer for a
    lightweight single-tenant deployment, local development, a corpus, an
    offline replay, or an audit sink on a box where Postgres is not worth
    standing up.

    The cost is honest: reads scan the whole file and filter in memory,
    writes copy the file, and nothing coordinates two processes pointed at
    one directory. Fine for one WhatsApp bot; wrong for a busy tenant.

The production backend, `ExperienceStore` (SQLite/Postgres), lives in
syntri-core and is not part of the public distribution. Both satisfy the same
`Store` interface, so moving between them is a matter of replaying the log —
nothing in the engine has to be told.
"""

from store.base import Store
from store.jsonl import JsonlStore

__all__ = ["JsonlStore", "Store"]
