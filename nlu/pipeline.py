"""
The understanding step: text in, intent and entities out.

This is a baseline, not a finished NLU layer. It exists because
/understand and /turn need an understanding step and it is the smallest
honest one that works end to end. The fallback tier in particular is
weak by construction; `syntri train` is what replaces it.

Intent comes from the instance's live trained classifier when one exists.
Two places are consulted, in order: the platform store's promoted model,
and then the local file registry that `syntri train --local` writes. The
store wins where both have one. Either way the result reports
`source="model"`.

When neither has a model — every instance before its first successful
training run — it falls back to token overlap against the example
utterances written in the taxonomy. That fallback is genuinely weak: it
cannot generalise past the words the developer happened to type, and it
will confidently pick the least-wrong intent from a small set. It
reports itself as `source="taxonomy"` so a caller can tell the two
apart, and `syntri train` is what replaces it.

Entities come from regular expressions: the ones declared on each
EntityDef in the taxonomy first, then a small set of built-in patterns for
the shapes that recur across every customer (amounts, phone numbers,
account numbers, dates). This is a pattern matcher, not a named-entity
recogniser. It has no model behind it and it does not learn.

`source` IS A SAFETY SIGNAL, NOT A DIAGNOSTIC
---------------------------------------------

Every `Understanding` reports where its intent came from, and the
policy engine reads that field to decide what may be acted on. The
three values and what each one means downstream:

    "model"     a trained classifier decided. Its confidence is a
                calibrated probability, so `policy/engine.py` measures
                it against the intent's floor from the taxonomy
                (`floor_for`) — 0.85 for money, 0.90 for irreversible.
                Clear the floor and the dispatch proceeds.

    "taxonomy"  the token-overlap fallback decided. The number it
                returns is a similarity score, not a probability of
                anything, so it is *not* measured against a floor
                calibrated for one. Instead the risk tier governs:
                intents the taxonomy marks `money` or `irreversible`
                are refused outright with `source_insufficient`, at any
                reported score. A 0.99 overlap means the sentence
                resembled an example, which is not evidence that the
                user wants money moved. Non-risky intents pass, so
                cold start still works.

    "none"      nothing matched. Refused as `no_intent`.

The practical consequence, and the reason this is documented here as
well as in `policy/engine.py`: **promoting a trained model changes what
an instance is allowed to do.** Before promotion, an untrained instance
can search flights and read balances but cannot transfer money, however
confident the fallback sounds. After `syntri train` and `syntri eval`
promote a model, `source` becomes "model" and money intents become
reachable — gated by the floor rather than refused outright. Nothing in
the policy configuration changes; the source does.

See `policy/engine.py` for the rules themselves and `TRUSTED_SOURCES`
for the exact set of strings treated as calibrated.


ONE THING THIS DELIBERATELY DOES NOT DO

It does not enforce those floors itself, and still should not: refusing
to act on a weak classification is a policy decision, not an
understanding one. This module reports; `syntri/policy/` decides.

# TODO(syntri): resolve relative dates. "next friday" is extracted as
# the literal string and handed to the pack as the DEPART_DATE slot.
# Unblocked by the user's timezone and the conversation's reference
# date reaching this layer — neither does today. See OPEN_ISSUES.md.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Words carrying no intent signal. Kept short on purpose — an aggressive
#: stop list strips the verbs ("send", "buy", "book") that are most of the
#: signal in a short command.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "to", "of", "in", "on", "at", "for", "with", "from", "by",
    "and", "or", "but", "if", "so", "please", "pls", "abeg",
    "do", "does", "did", "can", "could", "would", "will", "shall",
    "want", "like", "need", "get", "got", "have", "has", "had",
    "this", "that", "these", "those", "there", "here", "what", "how",
})

_TOKEN = re.compile(r"[a-z0-9']+")


#: Suffixes stripped before comparing tokens, longest first. Crude on
#: purpose: it exists so "flying"/"flights" and "fly"/"flight" meet in the
#: middle, which is most of what the fallback matcher gets wrong. A real
#: stemmer belongs with a real model, and both arrive with `syntri train`.
_SUFFIXES = ("ing", "ies", "es", "ed", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> list[str]:
    return [
        _stem(t)
        for t in _TOKEN.findall((text or "").lower())
        if t not in _STOPWORDS
    ]


# --------------------------------------------------------------------------
# built-in entity patterns
# --------------------------------------------------------------------------
#
# Ordered most-specific first, and each match consumes its span so a later
# pattern cannot re-read the same digits. Without that, an 11-digit phone
# number is also a perfectly good AMOUNT.

_BUILTIN_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 0803 123 4567 / +234 803 123 4567 — Nigerian mobile shapes.
    # 0 + a 3-digit network prefix + 7 digits = 11, or +234 and the same
    # ten after the leading zero is dropped.
    ("PHONE", re.compile(
        r"(?<!\d)(?:\+?234[\s-]?|0)[789]\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"
    )),
    # NUBAN account numbers are exactly ten digits.
    ("ACCOUNT_NUMBER", re.compile(r"(?<!\d)\d{10}(?!\d)")),
    # 5,000 / 5000 / 5k / ₦5,000 / NGN 5000
    ("AMOUNT", re.compile(
        r"(?:[₦$]|\b(?:ngn|usd|ghs|kes|zar)\s*)?"
        r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d+(?:\.\d{1,2})?)\s*([km])?\b",
        re.IGNORECASE,
    )),
    # "from lagos" / "leaving abuja" — the origin of a journey. Letters
    # only, so "to 0123456789" in a transfer is never read as a place.
    ("ORIGIN", re.compile(
        r"\b(?:from|leaving|departing)\s+"
        r"([a-z]+(?:[\s-][a-z]+){0,2})\b",
        re.IGNORECASE,
    )),
    # "to accra" / "flying to port harcourt" — the destination.
    ("DESTINATION", re.compile(
        r"\b(?:to|for|into)\s+"
        r"([a-z]+(?:[\s-][a-z]+){0,2})\b",
        re.IGNORECASE,
    )),
    # 2026-10-02 / 02/10/2026 / next friday / on the 2nd / 12 october
    ("DEPART_DATE", re.compile(
        r"\b(\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
        r"|(?:next|this|coming)\s+"
        r"(?:mon|tues|wednes|thurs|fri|satur|sun)day"
        r"|(?:today|tomorrow)"
        r"|(?:the\s+)?\d{1,2}(?:st|nd|rd|th)"
        r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
        re.IGNORECASE,
    )),
]

#: Multipliers for the "5k" / "2m" shorthand.
_MAGNITUDE = {"k": 1_000, "m": 1_000_000}

#: Words that end a place name rather than belonging to it. The
#: prepositional patterns greedily take up to three words — "port
#: harcourt" is two and "dar es salaam" is three — so the tail has to be
#: trimmed back off: "to accra next friday" is Accra, not "accra next
#: friday". This is the crude half of a crude extractor; a gazetteer of
#: the customer's actual route network replaces it and should.
_PLACE_STOPWORDS = frozenset({
    "next", "this", "coming", "today", "tomorrow", "tonight", "on", "at",
    "in", "by", "before", "after", "around", "please", "pls", "abeg",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "morning", "afternoon", "evening", "night", "week", "month",
    "and", "or", "with", "for", "from", "to", "the", "a", "an",
})


def _trim_place(value: str) -> str:
    """Cut a captured place name at the first word that cannot be part of one."""
    words: list[str] = []
    for word in value.split():
        if word.lower() in _PLACE_STOPWORDS:
            break
        words.append(word)
    return " ".join(words).strip()


def _normalise_amount(match: re.Match) -> str:
    digits = match.group(1).replace(",", "")
    suffix = (match.group(2) or "").lower()
    if suffix in _MAGNITUDE:
        try:
            return str(int(float(digits) * _MAGNITUDE[suffix]))
        except ValueError:
            return digits
    return digits


@dataclass
class Understanding:
    """What the pipeline made of one utterance."""

    text: str
    intent: str
    confidence: float
    entities: dict[str, str] = field(default_factory=dict)
    alternatives: list[dict] = field(default_factory=list)
    #: "model" when a trained classifier decided, "taxonomy" when the
    #: cold-start fallback did, "none" when neither could.
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "entities": dict(self.entities),
            "alternatives": list(self.alternatives),
            "source": self.source,
            "text": self.text,
        }


class NLUPipeline:
    """
    One instance's understanding step.

    Built per request from the instance's taxonomy and, when one has been
    trained and promoted, its live intent classifier.
    """

    def __init__(
        self,
        taxonomy,
        classifier=None,
        *,
        gazetteers: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.classifier = classifier
        self.gazetteers = gazetteers or {}

    # -- public -----------------------------------------------------------

    def understand(self, text: str) -> Understanding:
        intent, confidence, alternatives, source = self._classify(text)
        return Understanding(
            text=text,
            intent=intent,
            confidence=confidence,
            entities=self.extract_entities(text),
            alternatives=alternatives,
            source=source,
        )

    # -- intent -----------------------------------------------------------

    def _classify(self, text: str) -> tuple[str, float, list[dict], str]:
        if self.classifier is not None:
            try:
                intent, confidence = self.classifier.predict_one(text)
                return intent, float(confidence), self._alternatives(text, intent), "model"
            except Exception as exc:  # noqa: BLE001 - a model artefact
                log.warning("live classifier failed, falling back: %s", exc)

        return self._classify_from_taxonomy(text)

    def _alternatives(self, text: str, chosen: str) -> list[dict]:
        """Runner-up intents, when the classifier can produce them."""
        scores = getattr(self.classifier, "predict_proba_one", None)
        if scores is None:
            return []
        try:
            ranked = scores(text)
        except Exception:  # noqa: BLE001
            return []
        return [
            {"intent": name, "confidence": round(float(score), 4)}
            for name, score in ranked
            if name != chosen
        ][:3]

    def _classify_from_taxonomy(self, text: str) -> tuple[str, float, list[dict], str]:
        """
        Cold-start fallback: nearest taxonomy example by token overlap.

        Scored as the proportion of the *utterance's* tokens that appear in
        an example, so a long example does not win by containing more
        words. It is a weak signal and it is reported as such.
        """
        tokens = set(_tokens(text))
        if not tokens or self.taxonomy is None:
            return "unknown", 0.0, [], "none"

        scored: list[tuple[float, str]] = []
        for intent_def in getattr(self.taxonomy, "intents", []) or []:
            best = 0.0
            for example in intent_def.examples or []:
                example_tokens = set(_tokens(example))
                if not example_tokens:
                    continue
                overlap = len(tokens & example_tokens)
                if not overlap:
                    continue
                # Harmonic-ish: reward covering the utterance and the
                # example, so "send" alone does not match every money intent.
                score = (overlap / len(tokens)) * (
                    overlap / len(example_tokens)
                ) ** 0.5
                best = max(best, score)
            if best > 0:
                scored.append((best, intent_def.name))

        if not scored:
            return "unknown", 0.0, [], "none"

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        top_score, top_intent = scored[0]
        return (
            top_intent,
            min(1.0, round(top_score, 4)),
            [
                {"intent": name, "confidence": min(1.0, round(score, 4))}
                for score, name in scored[1:4]
            ],
            "taxonomy",
        )

    # -- entities ---------------------------------------------------------

    def extract_entities(self, text: str) -> dict[str, str]:
        """
        Pull entity values out of an utterance.

        Declared patterns win over built-ins: a customer who wrote a regex
        for POLICY_NUMBER means that one, and a customer who declared
        AMOUNT with their own pattern has overridden ours on purpose.
        """
        if not text:
            return {}

        found: dict[str, str] = {}
        # Spans already claimed, so two entities cannot read the same digits.
        claimed: list[tuple[int, int]] = []

        declared = list(getattr(self.taxonomy, "entities", []) or [])
        declared_names = {e.name for e in declared}

        def overlaps(start: int, end: int) -> bool:
            return any(s < end and start < e for s, e in claimed)

        # 1. Patterns the taxonomy declares.
        for entity in declared:
            for raw in entity.patterns or []:
                try:
                    pattern = re.compile(raw, re.IGNORECASE)
                except re.error as exc:
                    log.warning(
                        "entity %s has an invalid pattern %r: %s",
                        entity.name, raw, exc,
                    )
                    continue
                match = pattern.search(text)
                if match is None or overlaps(*match.span()):
                    continue
                value = match.group(1) if match.groups() else match.group(0)
                found[entity.name] = value.strip()
                claimed.append(match.span())
                break

        # 2. Gazetteer lookups for entities that name one.
        for entity in declared:
            if entity.name in found or not entity.gazetteer:
                continue
            table = self.gazetteers.get(entity.gazetteer) or {}
            hit = self._gazetteer_hit(text, table)
            if hit is not None:
                value, span = hit
                if not overlaps(*span):
                    found[entity.name] = value
                    claimed.append(span)

        # 3. Built-in shapes, only for entities this taxonomy declares and
        #    that nothing has filled yet.
        for name, pattern in _BUILTIN_PATTERNS:
            if name in found or name not in declared_names:
                continue
            for match in pattern.finditer(text):
                if overlaps(*match.span()):
                    continue
                span = match.span()
                if name == "AMOUNT":
                    value = _normalise_amount(match)
                elif name in {"ORIGIN", "DESTINATION"}:
                    value = _trim_place(match.group(1))
                    # Claim only the words actually kept. Claiming the whole
                    # match would swallow the trailing "next friday" that
                    # DEPART_DATE still needs to read.
                    start = match.start(1)
                    span = (start, start + len(value))
                else:
                    value = match.group(0).strip()
                if not value:
                    continue
                found[name] = value
                claimed.append(span)
                break

        return found

    def _gazetteer_hit(
        self, text: str, table: dict[str, str]
    ) -> tuple[str, tuple[int, int]] | None:
        """
        Longest gazetteer surface form present in the text.

        Longest-first so "first bank" is not matched as "bank", which would
        leave the qualifier behind and resolve to the wrong institution.
        """
        lowered = text.lower()
        for surface in sorted(table, key=len, reverse=True):
            if not surface:
                continue
            match = re.search(
                rf"(?<!\w){re.escape(surface.lower())}(?!\w)", lowered
            )
            if match is not None:
                return table[surface] or surface, match.span()
        return None


# NOTE: syntri-core's pipeline.py also defines the construction helpers
# `load_live_classifier`, `_from_file_registry` and `pipeline_for`. They
# wire a promoted intent classifier into the pipeline and depend on the
# private `syntri.learning.*` training pipeline and on store internals
# (`store.get_live_model`, `record.taxonomy()`), so they are deliberately
# left in syntri-core. Construct `NLUPipeline(taxonomy, classifier=...)`
# directly here: pass any object exposing `predict_one` (and optionally
# `predict_proba_one`) as the classifier, or None for the taxonomy fallback.
