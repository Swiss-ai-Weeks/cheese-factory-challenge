"""Kit --exec entry point used by the streamed GUI container."""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
print("FACTORY_ENTRY loaded", flush=True)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def _start() -> None:
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.robot_motion.examples", True)
    from sim.factory.config import load_config
    from sim.factory.run_factory import run

    classifier = os.environ.get("CHEESE_CLASSIFIER", "model")
    max_objects_text = os.environ.get("CHEESE_MAX_OBJECTS")
    max_objects = int(max_objects_text) if max_objects_text else None
    try:
        await run(load_config(), classifier, max_objects, cleanup=False)
    except Exception as exc:
        print(f"FACTORY_FATAL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        if os.environ.get("CHEESE_EXIT_ON_COMPLETE") == "1":
            # Isaac 6.1 full-streaming teardown hangs/crashes in debug_draw
            # after post_quit. This is a disposable evaluation container and
            # all artifacts are synchronously flushed, so exit at the process
            # boundary instead of clearing the live USD stage.
            os._exit(1)
    else:
        if os.environ.get("CHEESE_EXIT_ON_COMPLETE") == "1":
            os._exit(0)


asyncio.ensure_future(_start())
