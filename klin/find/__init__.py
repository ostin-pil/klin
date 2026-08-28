"""Discovery adapters: browsing a vendor's public catalogue without a browser.

`fetch` makes bytes appear and `find` only looks. The verbs are separate
because their obligations are: a fetcher owes the ledger a record and the
cache a verified file, while a finder owes nothing downstream at all — it
prints what the vendor publishes so a person can decide whether a download
is worth their clicks. Nothing here writes anything anywhere.

Adapters are discovered the way `klin.fetch` discovers vendors: a module in
this package declaring `NAME` and `configure` becomes a subcommand, so a
second catalogue is a new file and no edit anywhere else.

What every finder owes, inherited from the fetch adapters' first rule:
**never guess a licence.** A listing's price is a price and a storefront is
not a licence; where a vendor page carries a real licence field the finder
shows it verbatim, and where it does not the finder says so instead of
paraphrasing the description into one.
"""

import importlib
import pkgutil


class FindError(Exception):
    pass


def adapters():
    """Every adapter module in this package, by name. Discovery, not a list."""
    found = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module("%s.%s" % (__name__, info.name))
        if hasattr(module, "NAME") and hasattr(module, "configure"):
            found[module.NAME] = module
    return found


def _run_adapter(args, stream):
    return args.module.run(args, stream)


def configure(parser):
    """Wire every discovered adapter in as a subcommand."""
    vendors = parser.add_subparsers(dest="vendor")
    for name in sorted(adapters()):
        module = adapters()[name]
        sub = vendors.add_parser(name, help=getattr(module, "HELP", None))
        module.configure(sub)
        sub.set_defaults(func=_run_adapter, module=module)
