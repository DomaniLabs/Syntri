"""The understanding step: text in, intent and entities out.

This is the standalone base pipeline. It classifies intent (a trained
classifier when one is passed in, else a token-overlap fallback against the
taxonomy's examples) and extracts entities by regex. It reports where its
intent came from in `Understanding.source`, which the policy engine reads as a
safety signal.

The private model-loading/construction helpers stay in syntri-core; build an
`NLUPipeline` directly here. See `nlu/pipeline.py`.
"""

from syntri_contracts.nlu.pipeline import NLUPipeline, Understanding

__all__ = ["NLUPipeline", "Understanding"]
