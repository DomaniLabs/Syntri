"""CLI commands available in the public distribution.

Only `pack` is here. syntri-core's `train` and `eval` commands are operator
commands that drive the private `ExperienceStore` and the `syntri.learning`
training pipeline; they cannot run without private core and are not extracted.
"""

from cli.commands.pack import pack

__all__ = ["pack"]
