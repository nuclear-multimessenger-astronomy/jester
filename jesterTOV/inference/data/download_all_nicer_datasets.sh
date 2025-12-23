#!/bin/bash
# Download all NICER datasets from Zenodo (except J0030 Amsterdam recent)
# This script downloads datasets one at a time with delays to avoid rate limiting

set -e  # Exit on error

# Change to the data directory
cd "$(dirname "$0")"

# Function to download a dataset
download_dataset() {
    local psr=$1
    local group=$2
    local version=$3
    local delay=$4

    echo ""
    echo "=========================================="
    echo "Downloading: $psr / $group / $version"
    echo "=========================================="

    uv run python explore_nicer_datasets.py \
        --psr "$psr" \
        --group "$group" \
        --version "$version"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded $psr / $group / $version"
    else
        echo "✗ Failed to download $psr / $group / $version"
        echo "  You can download manually from the Zenodo page"
    fi

    if [ -n "$delay" ]; then
        echo ""
        echo "⏳ Waiting $delay seconds before next download..."
        sleep "$delay"
    fi
}

# Download all datasets with delays
download_dataset "J0030" "amsterdam" "intermediate" 2
download_dataset "J0030" "amsterdam" "original" 2
download_dataset "J0030" "maryland" "original" 2
download_dataset "J0740" "amsterdam" "intermediate" 2
download_dataset "J0740" "amsterdam" "original" 2
download_dataset "J0740" "maryland" "original" 2
download_dataset "J0740" "amsterdam" "recent" 2

echo ""
echo "=========================================="
echo "Download script completed!"
echo "=========================================="
echo ""
echo "Downloaded files are in: ./zenodo_data/"
echo ""
echo "Next steps:"
echo "  - Check the downloaded files in zenodo_data/"
echo "  - Run exploration on individual datasets if needed"
echo "  - Update README with dataset information"
