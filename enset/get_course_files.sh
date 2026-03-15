#!/bin/bash

export GOOGLE_WORKSPACE_CLI_KEYRING_BACKEND=file

# Course labels
declare -A COURSES
COURSES["609532492550"]="Machine Learning -bdcc1 -2022-2023"
COURSES["631285532352"]="DeepLearning-BDCC2-FC"
COURSES["631374053769"]="bddc2-deepLearning"

echo "=========================================="
echo " Downloading Classroom Files"
echo "=========================================="

for ID in "${!COURSES[@]}"; do

    COURSE_NAME="${COURSES[$ID]}"
    echo "📚 Course: $COURSE_NAME"

    mkdir -p "courses/$COURSE_NAME"

    # Function to process links
    download_files () {
        jq -r '.. | .driveFile? | select(.id != null) | "\(.title)|\(.id)"' |
        while IFS="|" read -r title id; do

            echo "⬇️ Downloading: $title"

            gws drive files get --params "{\"fileId\": \"$id\", \"alt\": \"media\"}" -o "courses/$COURSE_NAME/$title"

        done
    }

    echo "  -> Course Materials"
    gws classroom courses courseWorkMaterials list --params "{\"courseId\": \"$ID\"}" | download_files

    echo "  -> Assignments"
    gws classroom courses courseWork list --params "{\"courseId\": \"$ID\"}" | download_files

    echo "------------------------------------------"

done

echo "✅ All files downloaded"
