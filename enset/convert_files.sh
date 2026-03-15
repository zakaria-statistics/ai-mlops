
#!/bin/bash

# Configuration
COURSES_DIR="courses"
CONVERTED_DIR="converted"

echo "=========================================="
echo " Converting Course Files to PDF & Extracting RARs"
echo "=========================================="

# 1. Ensure converted directory exists
mkdir -p "$CONVERTED_DIR"

# Function to mirror directory structure in converted/
mirror_dir() {
    local rel_dir=$(dirname "${1#$COURSES_DIR/}")
    mkdir -p "$CONVERTED_DIR/$rel_dir"
    echo "$CONVERTED_DIR/$rel_dir"
}

# 2. Convert PPTX to PDF
echo "📊 Converting PPTX files..."
find "$COURSES_DIR" -type f -name "*.pptx" | while read -r file; do
    dest_dir=$(mirror_dir "$file")
    echo "  -> Converting: $(basename "$file")"
    soffice --headless --convert-to pdf --outdir "$dest_dir" "$file" > /dev/null 2>&1
done

# 3. Convert DOCX to PDF
echo "📝 Converting DOCX files..."
find "$COURSES_DIR" -type f -name "*.docx" | while read -r file; do
    dest_dir=$(mirror_dir "$file")
    echo "  -> Converting: $(basename "$file")"
    soffice --headless --convert-to pdf --outdir "$dest_dir" "$file" > /dev/null 2>&1
done

# 4. Copy existing PDF, IPYNB, CSV, TXT, and MP4 files
echo "📂 Copying essential lab files (.pdf, .ipynb, .csv, .txt, .mp4)..."
find "$COURSES_DIR" -type f \( -name "*.pdf" -o -name "*.ipynb" -o -name "*.csv" -o -name "*.txt" -o -name "*.mp4" \) | while read -r file; do
    dest_dir=$(mirror_dir "$file")
    echo "  -> Copying: $(basename "$file")"
    cp "$file" "$dest_dir/"
done

# 5. Copy image resources (for notebooks)
echo "🖼️ Copying image resources..."
find "$COURSES_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.gif" \) | while read -r file; do
    dest_dir=$(mirror_dir "$file")
    echo "  -> Copying image: $(basename "$file")"
    cp "$file" "$dest_dir/"
done

# 6. Extract RAR files
echo "📦 Extracting RAR files..."
find "$COURSES_DIR" -type f -name "*.rar" | while read -r file; do
    dest_dir=$(mirror_dir "$file")
    echo "  -> Extracting: $(basename "$file")"
    unrar x -o+ "$file" "$dest_dir" > /dev/null 2>&1
done

echo "✅ All conversions and extractions complete."
