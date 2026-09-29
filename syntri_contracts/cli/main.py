"""Entry point for the public pack-tooling CLI.

Exposed as the `syntri-pack` console script (see pyproject.toml). Deliberately
NOT named `syntri`: that command belongs to syntri-core, and a second package
claiming it would shadow the real one depending on install order.

    syntri-pack validate ./my-pack/
    syntri-pack list
"""

from __future__ import annotations

import click

from syntri_contracts.cli.commands.pack import pack


@click.group()
@click.version_option(package_name="syntri-contracts", prog_name="syntri-pack")
def main() -> None:
    """Build and validate Syntri capability packs."""


main.add_command(pack)


if __name__ == "__main__":
    main()
