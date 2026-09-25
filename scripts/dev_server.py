"""Dev server for the splatfield wasm app: serves the trunk dist/ output AND
the release zip at /models.zip from one origin, so the app's same-origin
fetch (ZIP_URL in src/fetch.rs) finds the pinned bytes without a second
server. Trunk rebuilds wipe dist/, so the zip is served straight from the
durable master copy data/splatfield-models.zip (git-ignored, survives
rebuilds) instead of being copied into dist/.

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
MODELS_ZIP = os.path.join(REPO_ROOT, "data", "splatfield-models.zip")


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path.split("?", 1)[0] == "/models.zip":
            return MODELS_ZIP
        return super().translate_path(path)


def main():
    parser = argparse.ArgumentParser(
        description="Serve dist/ + the pinned models zip on one origin."
    )
    parser.add_argument(
        "root", nargs="?", default="dist", help="document root, relative to the repo (default: dist)"
    )
    parser.add_argument("--port", type=int, default=8931)
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=os.path.join(REPO_ROOT, args.root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(
        f"serving {args.root}/ + data/splatfield-models.zip at /models.zip "
        f"on http://127.0.0.1:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
