#!/usr/bin/env bash
set -euo pipefail

# remove old generated stubs
sudo rm -rf generated src/openai/
mkdir -p src/openai
touch src/openai/.gitkeep

sudo openapi-generator-cli generate \
    -i ./configs/openai_openapi.yaml \
    -g rust-axum \
    --skip-validate-spec \
    -o ./generated
sudo chown -R user:users generated

# Copy over generated stubs into it's own subpackage
cp -r generated/src/* src/openai/
rm -rf generated

# rename "lib.rs" to "mod.rs" to make it a module
mv src/openai/lib.rs src/openai/mod.rs

cargo fmt --all