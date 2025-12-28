import os
import requests
import zipfile
import io
from tqdm import tqdm
import shutil

def download_sample_data(dest_dir):
    # Zenodo Link for RPLHR-CT (Test set is usually smaller and sufficient for samples)
    url = "https://zenodo.org/records/17239183/files/test.zip?download=1"
    
    print(f"Downloading RPLHR-CT (Test Set) from Zenodo: {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Zenodo might not send content-length for dynamic downloads
        total_size = int(response.headers.get('content-length', 0))
        
        buffer = io.BytesIO()
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as t:
            for chunk in response.iter_content(1024*1024): # 1MB chunks
                t.update(len(chunk))
                buffer.write(chunk)
                
        print("Download complete. Analyzing zip content...")
        
        with zipfile.ZipFile(buffer) as z:
            # We are looking for High Resolution images to populate our 'data' folder
            # The dataset usually has structure like 'test/HR' and 'test/LR'
            # We will prioritize 'HR' or 'GT' folders.
            
            hr_candidates = []
            for file_info in z.infolist():
                if file_info.is_dir():
                    continue
                
                fname = file_info.filename
                # Check for HR indicators in path
                if 'HR' in fname or 'high' in fname.lower() or 'GT' in fname:
                     if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                         hr_candidates.append(file_info)
            
            # If no explicit HR folder found, just take all valid images (fallback)
            if not hr_candidates:
                for file_info in z.infolist():
                    if not file_info.is_dir() and file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                        hr_candidates.append(file_info)

            if len(hr_candidates) > 0:
                print(f"Found {len(hr_candidates)} High-Resolution images. Extracting to {dest_dir}...")
                
                count = 0
                for file_info in tqdm(hr_candidates, desc="Extracting"):
                    # Flatten: just use the filename, ignore directory structure
                    filename = os.path.basename(file_info.filename)
                    if not filename: continue
                    
                    target_path = os.path.join(dest_dir, filename)
                    
                    with z.open(file_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    count += 1
                    
                print(f"Successfully prepared {count} images in '{dest_dir}'.")
            else:
                print("No valid images found in the downloaded archive.")

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please manually download the dataset from https://zenodo.org/records/17239183")

if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    download_sample_data(data_dir)
