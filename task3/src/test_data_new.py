import os
import sys

# Target directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DATA_NEW'))

html_dir = os.path.join(base_dir, 'html')
pdf_dir = os.path.join(base_dir, 'pdf')
img_dir = os.path.join(base_dir, 'img')

def process_folder(folder_path, output_file_name):
    from main import extract_and_prepare

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return 1

    failures = 0
    supported = (".html", ".htm", ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif")
    output_path = os.path.join(base_dir, output_file_name)
    print(f"\nProcessing {folder_path} -> Output to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(supported):
                try:
                    result = extract_and_prepare(file_path)
                    
                    # Write the full details
                    out_file.write(f"=== File: {filename} ===\n")
                    out_file.write(f"Language: {result['dominant_language']} | Script: {result['dominant_script']}\n")
                    out_file.write(f"Extraction Method: {result['extraction_method']}\n")
                    text = "\n\n".join(passage["text"] for passage in result["passages"])
                    out_file.write(f"Text Extracted:\n{text}\n")
                    out_file.write("="*50 + "\n\n")
                    
                except Exception as e:
                    failures += 1
                    print(f"Error processing {filename}: {e}", file=sys.stderr)
    return failures

if __name__ == '__main__':
    failures = process_folder(html_dir, 'html_results.txt')
    failures += process_folder(pdf_dir, 'pdf_results.txt')
    failures += process_folder(img_dir, 'img_results.txt')
    if failures:
        sys.exit(1)
    print("\nAll done! Output files are saved in the DATA_NEW folder.")
