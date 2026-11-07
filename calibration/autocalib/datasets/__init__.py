"""Dataset drivers. Each module exposes ``run(args)`` for ``calibrate.py``."""
from . import bridge, robomind  # noqa: F401

DRIVERS = {'bridge': bridge.run, 'robomind': robomind.run}
