"""
CLI output formatting.
Consistent colours and symbols across all commands.
"""
from __future__ import annotations

import click

OK    = click.style("✓", fg="green")
FAIL  = click.style("✗", fg="red")
WARN  = click.style("!", fg="yellow")
INFO  = click.style("·", fg="cyan")
ARROW = click.style("→", fg="cyan")

def ok(msg: str)   -> None: click.echo(f"  {OK}  {msg}")
def fail(msg: str) -> None: click.echo(f"  {FAIL}  {msg}", err=True)
def warn(msg: str) -> None: click.echo(f"  {WARN}  {msg}")
def info(msg: str) -> None: click.echo(f"  {INFO}  {msg}")
def arrow(msg: str)-> None: click.echo(f"  {ARROW}  {msg}")

def header(title: str) -> None:
    click.echo()
    click.echo(click.style(f"  {title}", bold=True))
    click.echo(click.style("  " + "─" * len(title), fg="bright_black"))

def kv(key: str, value: str, width: int = 22) -> None:
    k = click.style(f"{key:<{width}}", fg="bright_black")
    click.echo(f"  {k} {value}")
