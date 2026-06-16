#!/usr/bin/env python3
"""
3GPP Specification Downloader - ASCII Safe
=========================================
Downloads curated key 5G specifications directly from the official 3GPP archive,
unzips them, and saves the extracted files into the target document folder.
"""

import os
import zipfile
import urllib.request
import urllib.error
import argparse
import sys
import time

# Curated list of critical 5G specifications
# Format: (Spec ID, Title, Series folder, Filename prefix)
CURATED_SPECS = [
    ("TS 38.331", "NR RRC Protocol", "38_series", "38331"),
    ("TS 23.501", "5G System Architecture", "23_series", "23501"),
    ("TS 24.501", "5G Non-Access-Stratum NAS Protocol", "24_series", "24501"),
    ("TS 38.300", "5G NR Overall Description", "38_series", "38300"),
    ("TS 38.401", "5G NG-RAN Architecture", "38_series", "38401"),
    ("TS 24.229", "IP Multimedia Call Control / SIP", "24_series", "24229"),
    ("TS 38.213", "Physical Layer Control Procedures", "38_series", "38213"),
    ("TS 38.214", "Physical Layer Data Procedures", "38_series", "38214"),
    ("TS 38.321", "Medium Access Control MAC Protocol", "38_series", "38321"),
    ("TS 38.304", "UE Procedures in Idle Mode", "38_series", "38304"),
    ("TS 23.502", "5G System Core Signaling Procedures", "23_series", "23502"),
    ("TS 38.413", "NG-RAN NG Application Protocol NGAP", "38_series", "38413"),
    ("TS 38.473", "NG-RAN F1 Application Protocol F1AP", "38_series", "38473"),
    ("TS 38.423", "NG-RAN Xn Application Protocol XnAP", "38_series", "38423"),
    ("TS 38.323", "NR Packet Data Convergence Protocol PDCP", "38_series", "38323"),
    ("TS 38.322", "NR Radio Link Control RLC Protocol", "38_series", "38322")
]

# 3GPP Release suffix character mapping (Release 17 = h, Release 16 = g, Release 15 = f)
RELEASES = [("Release 17", "h"), ("Release 16", "g"), ("Release 15", "f")]

def parse_args():
    parser = argparse.ArgumentParser(description="Downloader for 3GPP Telecom Specifications")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3gpp_docs"),
        help="Directory to save downloaded and extracted documents."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only download the first 2 specifications (TS 38.331 and TS 23.501) for fast smoke testing."
    )
    return parser.parse_args()

def download_and_extract(spec_id, series, prefix, target_dir):
    """
    Downloads a spec zip file trying Release 17, 16, and 15 in order,
    extracts the Word (.docx) file, and cleans up the zip archive.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Try different releases in order
    for rel_name, rel_char in RELEASES:
        # standard 3GPP zip file name formatting (e.g., 38331-h00.zip for v17.0.0)
        zip_filename = f"{prefix}-{rel_char}00.zip"
        url = f"https://www.3gpp.org/ftp/Specs/archive/{series}/{spec_id.replace('TS ', '')}/{zip_filename}"
        
        print(f"  Trying {rel_name}: {zip_filename}...")
        
        # Temporary zip save path
        temp_zip_path = os.path.join(target_dir, zip_filename)
        
        try:
            # Add user agent to prevent blocks
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_zip_path, 'wb') as out_file:
                out_file.write(response.read())
            
            print(f"  [SUCCESS] Downloaded {zip_filename} successfully.")
            
            # Extract zip contents
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Extract only docx or pdf files
                    if file_info.filename.endswith(('.docx', '.pdf')):
                        # Rename extracted file to preserve Spec ID for clean metadata tagging
                        ext = os.path.splitext(file_info.filename)[1]
                        clean_filename = f"{spec_id.replace(' ', '_')}_{rel_name.replace(' ', '_')}{ext}"
                        extracted_path = os.path.join(target_dir, clean_filename)
                        
                        with open(extracted_path, 'wb') as f_out:
                            f_out.write(zip_ref.read(file_info.filename))
                            
                        print(f"  [SUCCESS] Extracted: {clean_filename} ({file_info.file_size / (1024*1024):.2f} MB)")
            
            # Clean up temporary zip file
            os.remove(temp_zip_path)
            return True
            
        except urllib.error.HTTPError as e:
            # 404 is normal if that specific release index doesn't have a 15.0.0 or 17.0.0 version
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            continue
        except Exception as e:
            print(f"  [WARNING] Error downloading {spec_id} from {url}: {e}")
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            continue
            
    print(f"  [ERROR] Failed to download {spec_id} across all releases (15/16/17).")
    return False

def main():
    args = parse_args()
    
    print("=" * 70)
    print("[INFO] 3GPP Specification Downloader")
    print("=" * 70)
    print(f"Output Directory: {os.path.abspath(args.output_dir)}")
    print("=" * 70)
    
    specs_to_download = CURATED_SPECS[:2] if args.quick else CURATED_SPECS
    print(f"Starting download of {len(specs_to_download)} specifications...")
    
    success_count = 0
    start_time = time.time()
    
    for idx, (spec_id, title, series, prefix) in enumerate(specs_to_download):
        print(f"\n[{idx+1}/{len(specs_to_download)}] Downloading {spec_id} - {title}...")
        success = download_and_extract(spec_id, series, prefix, args.output_dir)
        if success:
            success_count += 1
            
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[INFO] Download Process Completed!")
    print(f"Successfully processed {success_count}/{len(specs_to_download)} specs.")
    print(f"Total time elapsed: {total_time/60:.2f} minutes")
    print(f"Documents are saved in: {os.path.abspath(args.output_dir)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
