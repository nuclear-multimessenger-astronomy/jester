"""Download NICER datasets from Zenodo for reproducibility

This module uses the `zenodo_get` package to download datasets from Zenodo.
Install with: uv pip install zenodo-get
"""

import subprocess
from pathlib import Path
from typing import Optional
import shutil
import time


# Zenodo dataset information for NICER pulsars
# TODO: change the names in the dirs to last_name_year format
ZENODO_DATASETS = {
    "J0030": {
        "amsterdam": {
            "intermediate": {
                "name": "Vinciguerra et al. 2023",
                "zenodo_id": "8239000",
                "url": "https://zenodo.org/records/8239000",
                "description": "Amsterdam analysis of PSR J0030+0451 up to 2018 NICER data",
                "last_author": "Vinciguerra",
                "hotspot_models": [],  # To be determined from exploration
            },
            "original": {
                "name": "Riley et al. 2019",
                "zenodo_id": "3473466",
                "url": "https://zenodo.org/records/3473466",
                "description": "Original Amsterdam analysis of PSR J0030+0451",
                "last_author": "Riley",
                "hotspot_models": [],  # To be determined from exploration
            },
        },
        "maryland": {
            "original": {
                "name": "Miller et al. 2019",
                "zenodo_id": "3473464",  # TODO: change to the correct one
                "url": "https://zenodo.org/records/3473464",
                "description": "Original Maryland analysis of PSR J0030+0451",
                "last_author": "Miller",
                "hotspot_models": [],  # To be determined from exploration
            }
        },
    },
    "J0740": {
        "amsterdam": {
            "recent": {
                "name": "Salmi et al. 2024",
                "zenodo_id": "10519473",
                "url": "https://zenodo.org/records/10519473",
                "description": "Most recent Amsterdam analysis of PSR J0740+6620",
                "last_author": "Salmi",
                "hotspot_models": [],  # To be determined from exploration
            },
            "intermediate": {
                "name": "Salmi et al. 2022",
                "zenodo_id": "6827537",
                "url": "https://zenodo.org/records/6827537",
                "description": "Intermediate Amsterdam analysis of PSR J0740+6620",
                "last_author": "Salmi",
                "hotspot_models": [],  # To be determined from exploration
            },
            "original": {
                "name": "Riley et al. 2022",
                "zenodo_id": "7096886",
                "url": "https://zenodo.org/records/7096886",
                "description": "Original Amsterdam analysis of PSR J0740+6620",
                "last_author": "Riley",
                "hotspot_models": [],  # To be determined from exploration
            },
        },
        "maryland": {
            "original": {
                "name": "Miller et al. 2021",
                "zenodo_id": "4670689",
                "url": "https://zenodo.org/records/4670689",
                "description": "Original Maryland analysis of PSR J0740+6620",
                "last_author": "Miller",
                "hotspot_models": [],  # To be determined from exploration
            }
        },
    },
}


class ZenodoDownloader:
    """Download and manage NICER datasets from Zenodo using zenodo_get"""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize downloader

        Parameters
        ----------
        base_dir : Path, optional
            Base directory for downloads. If not provided, uses ./zenodo_data/
        """
        if base_dir is None:
            # Default to a subdirectory in the data folder
            base_dir = Path(__file__).parent / "zenodo_data"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def check_zenodo_get_installed(self) -> bool:
        """
        Check if zenodo_get is installed

        Returns
        -------
        bool
            True if zenodo_get is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["zenodo_get", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def download_zenodo_record(
        self,
        zenodo_id: str,
        output_dir: Path,
        timeout_seconds: int = 10*3600,
        max_retries: int = 50,
        retry_delay: int = 5,
    ) -> bool:
        """
        Download a Zenodo record using zenodo_get with retry logic

        Parameters
        ----------
        zenodo_id : str
            Zenodo record ID (e.g., "8239000")
        output_dir : Path
            Directory to save files
        timeout_seconds : int, optional
            Timeout for download in seconds (default: 1 hour)
        max_retries : int, optional
            Maximum number of retry attempts (default: 3)
        retry_delay : int, optional
            Delay between retries in seconds (default: 60)

        Returns
        -------
        bool
            True if download succeeded, False otherwise
        """
        # Check if zenodo_get is installed
        if not self.check_zenodo_get_installed():
            print("ERROR: zenodo_get is not installed!")
            print("Install with: uv pip install zenodo-get")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        # Run zenodo_get command with verbose output and retry logic
        # Note: zenodo_get downloads to current directory, so we change to output_dir
        cmd = ["zenodo_get", zenodo_id]

        for attempt in range(max_retries):
            if attempt > 0:
                print(f"\n⏳ Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)

            print(f"\n{'='*60}")
            print(f"Running: {' '.join(cmd)} in directory {output_dir}")
            print(f"Attempt {attempt + 1}/{max_retries}")
            print(f"{'='*60}")

            try:
                # Don't capture output - let it stream to terminal in real-time
                result = subprocess.run(
                    cmd,
                    cwd=str(output_dir),
                    timeout=timeout_seconds,
                    check=False,
                )

                print("=" * 60)
                if result.returncode == 0:
                    print(f"✓ Successfully downloaded Zenodo record {zenodo_id}")
                    return True
                else:
                    print(f"✗ Download failed with return code {result.returncode}")
                    if attempt < max_retries - 1:
                        print(f"  (Will retry - this may be due to rate limiting)")

            except subprocess.TimeoutExpired:
                print(f"✗ Download timed out after {timeout_seconds} seconds")
                if attempt < max_retries - 1:
                    print(f"  (Will retry)")
            except KeyboardInterrupt:
                print(f"\n✗ Download interrupted by user")
                return False
            except Exception as e:
                print(f"✗ Error running zenodo_get: {e}")
                if attempt < max_retries - 1:
                    print(f"  (Will retry)")

        print(f"\n✗ Failed to download after {max_retries} attempts")
        return False

    def download_dataset(
        self,
        psr_name: str,
        group: str,
        version: str = "recent",
        force: bool = False,
    ) -> Optional[Path]:
        """
        Download a NICER dataset from Zenodo

        Parameters
        ----------
        psr_name : str
            Pulsar name ("J0030" or "J0740")
        group : str
            Analysis group ("amsterdam" or "maryland")
        version : str, optional
            Dataset version ("recent", "intermediate", "original")
        force : bool, optional
            If True, re-download even if files exist

        Returns
        -------
        Path or None
            Path to downloaded dataset directory, or None if download failed

        Examples
        --------
        >>> downloader = ZenodoDownloader()
        >>> path = downloader.download_dataset("J0030", "amsterdam", "recent")
        """
        # Get dataset info
        if psr_name not in ZENODO_DATASETS:
            print(f"Unknown pulsar: {psr_name}")
            return None

        if group not in ZENODO_DATASETS[psr_name]:
            print(f"Unknown group {group} for pulsar {psr_name}")
            return None

        if version not in ZENODO_DATASETS[psr_name][group]:
            print(f"Unknown version {version} for {psr_name}/{group}")
            print(f"Available versions: {list(ZENODO_DATASETS[psr_name][group].keys())}")
            return None

        dataset_info = ZENODO_DATASETS[psr_name][group][version]
        zenodo_id = dataset_info["zenodo_id"]

        # Create output directory
        output_dir = self.base_dir / psr_name / group / version
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if files already exist (skip md5sums file)
        existing_files = [f for f in output_dir.iterdir() if f.is_file() and f.name != "md5sums.txt"]

        if existing_files and not force:
            print(f"\n{'='*60}")
            print(f"Dataset already exists: {dataset_info['name']}")
            print(f"Pulsar: {psr_name}, Group: {group}, Version: {version}")
            print(f"Location: {output_dir}")
            print(f"Files found: {len(existing_files)}")
            print(f"{'='*60}")
            print(f"\n✓ Skipping download (files already exist)")
            print(f"  Use force=True to re-download")
            return output_dir

        if existing_files and force:
            print(f"\n{'='*60}")
            print(f"Re-downloading dataset (force=True): {dataset_info['name']}")
            print(f"Existing files will be overwritten")
            print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print(f"Downloading dataset: {dataset_info['name']}")
        print(f"Pulsar: {psr_name}, Group: {group}, Version: {version}")
        print(f"Zenodo ID: {zenodo_id}")
        print(f"URL: {dataset_info['url']}")
        print(f"{'='*60}\n")

        # Download using zenodo_get
        success = self.download_zenodo_record(zenodo_id, output_dir)

        if success:
            print(f"\nDownload complete! Files saved to: {output_dir}")
            return output_dir
        else:
            print(f"\nDownload failed for {psr_name}/{group}/{version}")
            print(f"Please download manually from: {dataset_info['url']}")
            return None

    def list_available_datasets(self) -> None:
        """Print information about all available datasets"""
        print("\n" + "="*80)
        print("AVAILABLE NICER DATASETS FROM ZENODO")
        print("="*80 + "\n")

        for psr_name, psr_data in ZENODO_DATASETS.items():
            print(f"\n{psr_name}:")
            print("-" * 60)
            for group, group_data in psr_data.items():
                print(f"\n  {group.upper()}:")
                for version, dataset_info in group_data.items():
                    print(f"    [{version}] {dataset_info['name']}")
                    print(f"      Zenodo ID: {dataset_info['zenodo_id']}")
                    print(f"      URL: {dataset_info['url']}")
                    print(f"      Description: {dataset_info['description']}")
                    if dataset_info['hotspot_models']:
                        print(f"      Hotspot models: {', '.join(dataset_info['hotspot_models'])}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    downloader = ZenodoDownloader()
    downloader.list_available_datasets()

    # Uncomment to download a specific dataset:
    # downloader.download_dataset("J0030", "amsterdam", "recent")
    # downloader.download_dataset("J0740", "maryland", "original")
