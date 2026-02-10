#!/bin/bash
# Resumes download of missing DNS4 components: dev_testset and impulse_responses

AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
OUTPUT_PATH="./datasets/dns4"

mkdir -p "$OUTPUT_PATH"

BLOB_NAMES=(
    "datasets_fullband.dev_testset_000.tar.bz2"
    "datasets_fullband.impulse_responses_000.tar.bz2"
)

for BLOB in "${BLOB_NAMES[@]}"
do
    URL="$AZURE_URL/$BLOB"
    echo "Downloading and extracting: $BLOB"
    # Pipe curl output to tar for on-the-fly extraction
    # -x: extract
    # -j: bzip2
    # -f -: read from stdin
    # -C: change directory
    curl "$URL" | tar -C "$OUTPUT_PATH" -f - -x -j
done
