"""Dev server for the splatfield wasm app: serves the trunk dist/ output AND
the app's same-origin data fetches (ZIP_URL, DEMO_SCENE_URL in src/fetch.rs)
from one origin, so they find the pinned bytes without a second server.
Trunk rebuilds wipe dist/, so these are served straight from durable master
copies under data/ (git-ignored, survives rebuilds) instead of being copied
into dist/.

stdlib ThreadingHTTPServer: parallel requests. application/wasm is
registered explicitly — browsers refuse streaming compile for
octet-stream, and some python mimetypes DBs lack the entry. No COOP/COEP:
the worker runs numThreads=1, so no cross-origin isolation is needed.
"""
import argparse
import functools
import http.server
import mimetypes
import os

mimetypes.add_type("application/wasm", ".wasm")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# URL path -> data/ master copy. Keep in sync with the relative URLs the
# app fetches: /models.zip (ZIP_URL) and the help panel's demo scene
# (DEMO_SCENE_URL).
DATA_FILES = {
    "/models.zip": "splatfield-models.zip",
    "/bear.3d71a266_sh1.sog": "bear.3d71a266_sh1.sog",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        override = DATA_FILES.get(path.split("?", 1)[0])
        if override is not None:
            return os.path.join(REPO_ROOT, "data", override)
        return super().translate_path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Serve dist/ + the app's same-origin data files on one origin."
    )
    parser.add_argument(
        "root", nargs="?", default="dist", help="document root, relative to the repo (default: dist)"
    )
    parser.add_argument("--port", type=int, default=8931)
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=os.path.join(REPO_ROOT, args.root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(
        f"serving {args.root}/ + {', '.join(DATA_FILES)} from data/ "
        f"on http://127.0.0.1:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
