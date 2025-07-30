#!/usr/bin/env python3
"""
Model Downloader from Google Drive

This script helps download trained models from Google Drive to your local project.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import argparse

class GoogleDriveDownloader:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Expected model files
        self.expected_files = [
            "ppo_portfolio_agent_final.pth",
            "training_config.yaml",
            "training_metrics.npz"
        ]
    
    def download_file_from_google_drive(self, file_id, destination):
        """Download a file from Google Drive using file ID."""
        print(f"📥 Downloading to {destination}...")
        
        URL = "https://docs.google.com/uc?export=download"
        
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        # Handle large files with confirmation token
        token = self._get_confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        
        self._save_response_content(response, destination)
        print(f"✅ Downloaded: {destination}")
    
    def _get_confirm_token(self, response):
        """Extract confirmation token for large files."""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def _save_response_content(self, response, destination):
        """Save response content to file."""
        CHUNK_SIZE = 32768
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    
    def download_from_shared_link(self, shared_link, filename):
        """Download file from Google Drive shared link."""
        try:
            # Extract file ID from shared link
            if "drive.google.com" in shared_link:
                if "/file/d/" in shared_link:
                    file_id = shared_link.split("/file/d/")[1].split("/")[0]
                elif "id=" in shared_link:
                    file_id = shared_link.split("id=")[1].split("&")[0]
                else:
                    print("❌ Could not extract file ID from link")
                    return False
                
                destination = self.models_dir / filename
                self.download_file_from_google_drive(file_id, destination)
                return True
            else:
                print("❌ Not a valid Google Drive link")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    def download_from_folder_link(self, folder_link):
        """Download all model files from a shared Google Drive folder."""
        print("📁 Downloading from Google Drive folder...")
        print("⚠️ Automatic folder download not implemented yet.")
        print("\nPlease:")
        print("1. Open the folder link in your browser")
        print("2. Download each file individually")
        print("3. Place them in the models/ directory")
        print("\nExpected files:")
        for filename in self.expected_files:
            print(f"  📄 {filename}")
        return False
    
    def interactive_download(self):
        """Interactive download process."""
        print("🔗 Google Drive Model Download")
        print("=" * 40)
        print("\nOptions:")
        print("1. Download individual files using shared links")
        print("2. Download from shared folder")
        print("3. Manual download instructions")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            self._download_individual_files()
        elif choice == "2":
            folder_link = input("Enter Google Drive folder link: ").strip()
            self.download_from_folder_link(folder_link)
        elif choice == "3":
            self._show_manual_instructions()
        else:
            print("❌ Invalid choice")
    
    def _download_individual_files(self):
        """Download files one by one using shared links."""
        print("\n📄 Individual File Download")
        print("Please provide Google Drive shared links for each file:")
        
        for filename in self.expected_files:
            print(f"\n📄 {filename}:")
            link = input("Enter shared link (or 'skip'): ").strip()
            
            if link.lower() == 'skip':
                print(f"⏭️ Skipped {filename}")
                continue
            
            if link:
                success = self.download_from_shared_link(link, filename)
                if not success:
                    print(f"❌ Failed to download {filename}")
            else:
                print(f"⏭️ No link provided for {filename}")
    
    def _show_manual_instructions(self):
        """Show manual download instructions."""
        print("\n📋 Manual Download Instructions")
        print("=" * 40)
        print("\n1. Open your Google Drive")
        print("2. Navigate to 'AI_Portfolio_Models' folder")
        print("3. Download these files:")
        
        for filename in self.expected_files:
            print(f"   📄 {filename}")
        
        print(f"\n4. Place downloaded files in: {self.models_dir.absolute()}")
        print("\n5. Run the integration script:")
        print("   python colab_model_integration.py")
    
    def verify_downloads(self):
        """Verify that all required files have been downloaded."""
        print("\n🔍 Verifying downloaded files...")
        
        missing_files = []
        for filename in self.expected_files:
            file_path = self.models_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ {filename} ({file_size:.1f} MB)")
            else:
                print(f"  ❌ {filename} (missing)")
                missing_files.append(filename)
        
        if missing_files:
            print(f"\n⚠️ Missing files: {missing_files}")
            print("Please download the missing files and try again.")
            return False
        else:
            print("\n✅ All model files downloaded successfully!")
            print("\nNext step: Run integration script")
            print("python colab_model_integration.py")
            return True

def main():
    """Main downloader interface."""
    parser = argparse.ArgumentParser(description="Download trained models from Google Drive")
    parser.add_argument("--folder-link", help="Google Drive folder shared link")
    parser.add_argument("--file-links", nargs=3, help="Individual file shared links")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files")
    
    args = parser.parse_args()
    
    downloader = GoogleDriveDownloader()
    
    if args.verify_only:
        downloader.verify_downloads()
    elif args.folder_link:
        downloader.download_from_folder_link(args.folder_link)
    elif args.file_links:
        filenames = downloader.expected_files
        for link, filename in zip(args.file_links, filenames):
            downloader.download_from_shared_link(link, filename)
        downloader.verify_downloads()
    else:
        downloader.interactive_download()
        downloader.verify_downloads()

if __name__ == "__main__":
    main()
