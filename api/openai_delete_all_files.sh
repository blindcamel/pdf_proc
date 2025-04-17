#!/bin/bash

# Replace with your actual API key or use environment variable
API_KEY=${OPENAI_API_KEY}

if [ -z "$API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

echo "Retrieving list of files..."

# Get list of files and save IDs to a temporary file
curl -s https://api.openai.com/v1/files \
  -H "Authorization: Bearer $API_KEY" \
  | jq -r '.data[].id' > files_to_delete.txt

if [ ! -s files_to_delete.txt ]; then
    echo "No files found to delete."
    rm files_to_delete.txt
    exit 0
fi

# Count the number of files
file_count=$(wc -l < files_to_delete.txt)
echo "Found $file_count files to delete."

# Counter for progress
counter=0

# Delete each file
while read file_id; do
    counter=$((counter + 1))
    echo "($counter/$file_count) Deleting file: $file_id"
    
    delete_response=$(curl -s -X DELETE https://api.openai.com/v1/files/$file_id \
      -H "Authorization: Bearer $API_KEY")
    
    success=$(echo $delete_response | jq -r '.deleted')
    
    if [ "$success" = "true" ]; then
        echo "Successfully deleted: $file_id"
    else
        echo "Failed to delete: $file_id"
        echo "Response: $delete_response"
    fi
    
    # Small pause to prevent rate limiting
    sleep 0.5
done < files_to_delete.txt

# Clean up
rm files_to_delete.txt
echo "File deletion process completed."