"""
Floor resolution: what confidence an intent has to clear to be acted on.

The numbers themselves live in the instance's taxonomy — either written
on the intent as `confidence_floor`, or inherited from its risk tier's
default. Nothing here hardcodes a value for any particular intent, and
nothing here knows what a `send_money` is. This module only answers "what
does this taxonomy require for this intent, and what do we do when it has
nothing to say".

That last case is the whole reason this is a module and not a one-line
call to `Taxonomy.floor_for()`. An intent the taxonomy has never heard of
is the dangerous one: it is a typo in a pack manifest, a classifier
emitting a label from a taxonomy version that has since been edited, or a
caller passing a name straight through from user input. `floor_for()`
answers 0.75 for all of those — the elevated-tier default — which treats
an unknown intent as merely somewhat risky. It is not. It is unknown, and
the only floor an unknown intent can honestly be given is one nothing can
clear.

Hence FAIL_CLOSED: 1.0, not 0.0 and not 0.75. A real classifier does not
produce 1.0, so an unconfigured intent is refused rather than waved
through on a default that happened to be low enough.
"""
from __future__ import annotations

from typing import Any

#: The floor an intent gets when the taxonomy does not describe it.
#:
#: 1.0 rather than 0.0: an absent configuration must not read as "no
#: restriction". Deliberately unreachable — softmax output from a trained
#: classifier does not reach 1.0, and the token-overlap fallback caps at
#: 1.0 only on an exact example match, which is not a licence to move
#: money under an intent nobody declared.
FAIL_CLOSED = 1.0


def resolve_floor(
    intent_name: str,
    taxonomy: Any,
    *,
    default: float = FAIL_CLOSED,
) -> float:
    """
    The confidence floor for `intent_name` under `taxonomy`.

    Delegates to the taxonomy for intents it declares. For anything else —
    an unknown intent, an empty name, or no taxonomy at all — returns
    `default`, which fails closed.
    """
    if not intent_name or taxonomy is None:
        return default

    lookup = getattr(taxonomy, "intent", None)
    if lookup is None:
        # Not a Taxonomy. Refuse rather than guess at its shape.
        return default
    if lookup(intent_name) is None:
        return default

    try:
        return float(taxonomy.floor_for(intent_name))
    except Exception:  # noqa: BLE001 - a malformed taxonomy must not permit
        return default


def is_risky(intent_name: str, taxonomy: Any) -> bool:
    """
    Whether acting on this intent could move money or cannot be undone.

    An intent the taxonomy does not declare counts as risky. The question
    this answers is "may an untrusted classification source dispatch
    this", and for an intent we cannot describe, the answer is no.
    """
    if not intent_name or taxonomy is None:
        return True

    lookup = getattr(taxonomy, "intent", None)
    if lookup is None:
        return True

    spec = lookup(intent_name)
    if spec is None:
        return True
    return bool(getattr(spec, "moves_money", False))
