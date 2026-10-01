#!/usr/bin/env bash
# The one deployment shape, shared by both deploy legs (Pages and the
# EdgeOne mirror — each calls this with its dist dir so the two hosts
# cannot drift): the pinned models zip ships as `models.zip.p0…p<N-1>`
# parts behind a `models.zip.manifest.json` of `{"parts": N}`, plus the
# demo scene, into the dist dir given as $1. The naming is the exact
# convention src/fetch.rs's download_zip probes — pinned there by the
# url-contract test, the one cross-artifact anchor the compiler can't
# check against this script.
#
# The parts exist because EdgeOne Makers caps deploy files at 25 MB and
# the 158 MB archive cannot exist there whole; Pages ships the same
# shape so dist/ is one shape and download_zip's single-file branch is
# the dev server's alone.
set -euo pipefail

dest="${1:?usage: ship_models.sh <dist-dir>}"

curl -fsSL --retry 3 -o "$dest/models.zip" "$MODELS_URL"
echo "$MODELS_SHA256  $dest/models.zip" | sha256sum --check -
test "$(stat -c%s "$dest/models.zip")" = "$MODELS_BYTES"

split -n 7 -d -a 1 "$dest/models.zip" "$dest/models.zip.p"
rm "$dest/models.zip"
n=$(find "$dest" -name 'models.zip.p?' | wc -l)
# A part over 25,000,000 bytes would die in EdgeOne's deployment
# processing (verified empirically: a 30 MB file uploads then fails the
# deploy) — fail here where the cause is visible. The byte count covers
# both readings of EdgeOne's "25 MB".
big=$(find "$dest" -name 'models.zip.p?' -size +25000000c)
test -z "$big" || { echo "part exceeds EdgeOne's 25 MB cap: $big"; exit 1; }
printf '{"parts": %s}\n' "$n" > "$dest/models.zip.manifest.json"

# The demo scene cannot be fetched from the release URL client-side: the
# github.com download redirect chain sends no CORS headers. No hash pin
# for the bear — the content-hashed name is the identity, and a bad
# download just fails the demo load in the status pill.
curl -fsSL --retry 3 -o "$dest/bear.3d71a266_sh1.sog" "$DEMO_SCENE_URL"
