#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <json_file_path>"
    exit 1
fi

model_name="$1"
json_file="$2"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it to parse JSON files."
    echo "Install with: sudo apt-get install jq (Ubuntu/Debian)"
    echo "Or: brew install jq (MacOS)"
    exit 1
fi

# Check if JSON file exists
if [ ! -f datasets/"$json_file".json ]; then
    echo "Error: JSON file not found: $json_file"
    exit 1
fi

# Get the length of the JSON array
length=$(jq 'length' datasets/"$json_file".json)

if [ $? -ne 0 ]; then
    echo "Error: Failed to parse JSON file. Ensure it contains a valid array."
    exit 1
fi

echo "Found $length items in JSON file"

# Run the command for each index in the JSON array
for idx in $(seq 0 $((length - 1))); do
    echo "Running iteration $idx..."
    python prompt-minimization-main.py dataset=$json_file data_idx=$idx model_name="$model_name"
done