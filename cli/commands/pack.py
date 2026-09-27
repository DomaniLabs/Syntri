"""
syntri pack — list, validate and install capability packs.

`validate` is the one that matters. It is what a third-party developer
runs before submitting a pack, and it deliberately needs no server, no
account and no network: it imports the pack from a directory, registers it
into a throwaway registry to run the same startup checks the platform
would, and replays every fixture. A pack that passes here will not be
rejected at registration for anything this could have caught.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import click

from cli.fmt import arrow, fail, header, info, kv, ok, warn

#: Where a pack directory keeps its pack module, in preference order.
_PACK_MODULES = ("pack.py", "__init__.py")


@click.group()
def pack() -> None:
    """Capability packs — what your agent can do."""


# --------------------------------------------------------------------------
# syntri pack list
# --------------------------------------------------------------------------


@pack.command("list")
def list_packs() -> None:
    """List the capability packs installed on this machine and their intents.

    Reads packs from installed entry points. (syntri-core additionally offers
    a `--remote` listing against the platform you are logged in to; that path
    depends on the private SaaS client and is not part of the public CLI.)
    """
    _list_local()


def _list_local() -> None:
    from capabilities.registry import default_registry

    header("Capability packs")
    try:
        registry = default_registry()
    except Exception as exc:  # noqa: BLE001 - a broken installed pack
        fail(f"a registered pack is invalid: {exc}")
        raise SystemExit(1) from exc

    manifests = registry.installed()
    if not manifests:
        info("No capability packs registered.")
        click.echo()
        return

    for manifest in manifests:
        promotable, reason = registry.can_promote(manifest.name)
        status = (
            click.style("live", fg="green") if promotable
            else click.style("shadow", fg="yellow")
        )
        kv(manifest.name, f"v{manifest.version}  {status}")
        kv("", ", ".join(manifest.intents), width=22)
        if not promotable:
            kv("", click.style(reason, fg="yellow"), width=22)
    click.echo()
    info(f"{len(manifests)} pack(s), {len(registry.intents())} intents, "
         f"{len(registry.tools())} tools.")
    click.echo()


# --------------------------------------------------------------------------
# syntri pack validate
# --------------------------------------------------------------------------


@pack.command("validate")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--verbose", is_flag=True, help="Print every fixture turn.")
def validate(path: Path, verbose: bool) -> None:
    """
    Load a pack from a directory, run its fixtures, and report.

    Needs no running server — this is the pre-submission check.
    """
    from capabilities.registry import CapabilityRegistry
    from capabilities.replay import replay_pack

    header(f"Validating {path.name}")

    try:
        pack_obj = _load_pack_from(path)
    except Exception as exc:  # noqa: BLE001 - arbitrary third-party import
        fail(f"could not load a pack from {path}: {exc}")
        raise SystemExit(1) from exc

    manifest = pack_obj.manifest

    # The same checks the platform runs at startup, so a pack that passes
    # here cannot be rejected at registration for a reason this knew about.
    registry = CapabilityRegistry()
    try:
        registry.register(pack_obj)
    except ValueError as exc:
        fail(str(exc))
        click.echo()
        raise SystemExit(1) from exc

    for intent in manifest.intents:
        ok(f"{intent} pack validated")

    tools = pack_obj.tools()
    info(f"{len(tools)} tool(s): {', '.join(t.name for t in tools)}")

    money = [t.name for t in tools if t.moves_money]
    if money:
        info(f"moves money: {', '.join(money)}")

    report = replay_pack(pack_obj)

    if report.total == 0:
        warn("no fixtures — this pack cannot be promoted to live")
        click.echo()
        arrow("Add at least one fixture under fixtures/ and run this again.")
        click.echo()
        raise SystemExit(1)

    for outcome in report.outcomes:
        # A pack that loads its fixtures through `load_fixtures` records the
        # filename; one that reads the JSON itself has only the name. Do not
        # print the name twice when they are the same thing.
        label = (
            f"{outcome.source}: {outcome.name}"
            if outcome.source and outcome.source != outcome.name
            else outcome.name
        )
        if outcome.ok:
            if verbose:
                ok(f"{label} ({len(outcome.turns)} turns)")
        else:
            fail(label)
            for failure in outcome.failures:
                click.echo(f"        {failure}")

    click.echo()
    if report.ok:
        ok(f"{report.passed} fixture{'s' if report.passed != 1 else ''} passed")
        click.echo()
        arrow("Ready to register.")
        click.echo()
        return

    fail(f"{report.failed} of {report.total} fixtures failed")
    click.echo()
    raise SystemExit(1)


def _load_pack_from(path: Path):
    """
    Import a pack directory and instantiate the CapabilityPack in it.

    The directory is put on sys.path first so a pack laid out as a package
    — which is how the example pack and every published one are laid out —
    can import its own submodules.
    """
    directory = path.resolve()
    module_file = next(
        (directory / name for name in _PACK_MODULES if (directory / name).is_file()),
        None,
    )
    if module_file is None:
        raise FileNotFoundError(
            f"no {' or '.join(_PACK_MODULES)} in {directory}"
        )

    roots = [str(root) for root in _import_roots(directory)]
    added = [root for root in roots if root not in sys.path]
    for root in reversed(added):
        sys.path.insert(0, root)

    try:
        spec = importlib.util.spec_from_file_location(
            f"_syntri_pack_{directory.name}", module_file
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build an import spec for {module_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for root in added:
            if root in sys.path:
                sys.path.remove(root)

    candidates = [
        value for value in vars(module).values()
        if _is_pack_class(value, module)
    ]
    if not candidates:
        raise LookupError(
            f"{module_file} defines no concrete CapabilityPack subclass"
        )
    if len(candidates) > 1:
        names = ", ".join(sorted(c.__name__ for c in candidates))
        raise LookupError(
            f"{module_file} defines more than one pack ({names}); a pack "
            f"directory must contain exactly one"
        )
    return candidates[0]()


def _import_roots(directory: Path) -> list[Path]:
    """
    Directories to put on sys.path so a pack's own imports resolve.

    The pack's parent comes first. Then, if that parent is itself a Python
    package, the walk continues upward to the first ancestor that is not —
    that ancestor is the repo root, and it is what an absolute import like
    `from contracts.capability import ...` needs on the path. The public
    example pack is laid out exactly that way.
    """
    roots = [directory.parent]
    ancestor = directory.parent
    while (ancestor / "__init__.py").is_file() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
        roots.append(ancestor)
    return roots


def _is_pack_class(value: object, module) -> bool:
    """
    Whether a module-level name is the pack class this directory defines.

    Detected structurally, not with issubclass. A published pack brings
    its own installation of `syntri-contracts`, so its CapabilityPack can
    be a different class object from this one — a different version, or
    the same module resolved twice. issubclass is then False for a pack
    that is perfectly real; see _is_manifest in capabilities/registry.py
    for the same problem.

    `__module__` is checked so a pack that imports another pack, or the
    base class itself, does not get picked up as a second candidate.
    """
    if not isinstance(value, type):
        return False
    if getattr(value, "__abstractmethods__", None):
        return False
    if getattr(value, "__module__", None) != module.__name__:
        return False
    return (
        hasattr(value, "manifest")
        and callable(getattr(value, "tools", None))
        and callable(getattr(value, "advance", None))
    )


# --------------------------------------------------------------------------
# syntri pack install
# --------------------------------------------------------------------------


@pack.command("install")
@click.argument("name")
def install(name: str) -> None:
    """Install a capability pack (marketplace is not open yet)."""
    header(f"Install {name}")
    info("The Syntri pack marketplace is not open yet.")
    click.echo()
    arrow("For now, install the distribution directly:")
    click.echo()
    click.echo(f"    pip install syntri-capability-{name}")
    click.echo()
    info(
        "Any distribution advertising a 'syntri.capabilities' entry point is "
        "discovered automatically at server startup."
    )
    click.echo()
    arrow("Check what is installed with:")
    click.echo()
    click.echo("    syntri pack list")
    click.echo()
