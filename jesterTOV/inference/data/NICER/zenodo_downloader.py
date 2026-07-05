"""Download NICER datasets from Zenodo.

Uses the ``zenodo_get`` package to fetch archives. Install with::

    uv pip install zenodo-get
"""

import subprocess
import time
from pathlib import Path
from typing import Optional


# Zenodo dataset registry for all NICER pulsars
ZENODO_DATASETS: dict = {
    "J0437": {
        "amsterdam": {
            "original": {
                "name": "Choudhury et al. 2024",
                "zenodo_id": "13766753",
                "url": "https://zenodo.org/records/13766753",
                "description": "Amsterdam analysis of PSR J0437-4715 (nearest and brightest MSP)",
            },
        },
    },
    "J0614": {
        "amsterdam": {
            "original": {
                "name": "Dittmann et al. 2025",
                "zenodo_id": "17380576",
                "url": "https://zenodo.org/records/17380576",
                "description": "Amsterdam analysis of PSR J0614-3329 (1.4 Msun edge-on pulsar)",
            },
        },
    },
    "J0030": {
        "amsterdam": {
            "intermediate": {
                "name": "Vinciguerra et al. 2023",
                "zenodo_id": "8239000",
                "url": "https://zenodo.org/records/8239000",
                "description": "Amsterdam analysis of PSR J0030+0451 up to 2018 NICER data",
            },
            "original": {
                "name": "Riley et al. 2019",
                "zenodo_id": "3473466",
                "url": "https://zenodo.org/records/3473466",
                "description": "Original Amsterdam analysis of PSR J0030+0451",
            },
        },
        "maryland": {
            "original": {
                "name": "Miller et al. 2019",
                "zenodo_id": "3473464",
                "url": "https://zenodo.org/records/3473464",
                "description": "Original Maryland analysis of PSR J0030+0451",
            },
        },
    },
    "J0740": {
        "amsterdam": {
            "recent": {
                "name": "Salmi et al. 2024",
                "zenodo_id": "10519473",
                "url": "https://zenodo.org/records/10519473",
                "description": "Most recent Amsterdam analysis of PSR J0740+6620",
            },
            "intermediate": {
                "name": "Salmi et al. 2022",
                "zenodo_id": "6827537",
                "url": "https://zenodo.org/records/6827537",
                "description": "Intermediate Amsterdam analysis of PSR J0740+6620",
            },
            "original": {
                "name": "Riley et al. 2022",
                "zenodo_id": "7096886",
                "url": "https://zenodo.org/records/7096886",
                "description": "Original Amsterdam analysis of PSR J0740+6620",
            },
        },
        "maryland": {
            "original": {
                "name": "Miller et al. 2021",
                "zenodo_id": "4670689",
                "url": "https://zenodo.org/records/4670689",
                "description": "Original Maryland analysis of PSR J0740+6620",
            },
        },
    },
}


class ZenodoDownloader:
    """Download and cache NICER datasets from Zenodo using ``zenodo_get``."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            base_dir = Path(__file__).parent / "zenodo_data"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _zenodo_get_available(self) -> bool:
        try:
            return (
                subprocess.run(
                    ["zenodo_get", "--version"], capture_output=True, check=False
                ).returncode
                == 0
            )
        except FileNotFoundError:
            return False

    def _download_record(
        self,
        zenodo_id: str,
        output_dir: Path,
        timeout_seconds: int = 10 * 3600,
        max_retries: int = 50,
        retry_delay: int = 5,
    ) -> bool:
        if not self._zenodo_get_available():
            print("ERROR: zenodo_get not installed. Run: uv pip install zenodo-get")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["zenodo_get", zenodo_id]

        for attempt in range(max_retries):
            if attempt > 0:
                print(
                    f"\nRetrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)

            print(f"\n{'=' * 60}")
            print(
                f"zenodo_get {zenodo_id}  →  {output_dir}  (attempt {attempt + 1}/{max_retries})"
            )
            print("=" * 60)

            try:
                result = subprocess.run(
                    cmd, cwd=str(output_dir), timeout=timeout_seconds, check=False
                )
                if result.returncode == 0:
                    print(f"Downloaded Zenodo record {zenodo_id}")
                    return True
                print(f"Download failed (returncode {result.returncode})")
            except subprocess.TimeoutExpired:
                print(f"Timed out after {timeout_seconds}s")
            except KeyboardInterrupt:
                print("Interrupted by user")
                return False
            except Exception as e:
                print(f"Error: {e}")

        print(f"Failed after {max_retries} attempts")
        return False

    def download_dataset(
        self,
        psr_name: str,
        group: str,
        version: str = "original",
        force: bool = False,
    ) -> Optional[Path]:
        """Download a NICER dataset from Zenodo.

        Parameters
        ----------
        psr_name:
            Pulsar name, e.g. ``"J0030"`` or ``"J0740"``.
        group:
            Analysis group: ``"amsterdam"`` or ``"maryland"``.
        version:
            Dataset version: ``"recent"``, ``"intermediate"``, or ``"original"``.
        force:
            Re-download even if files already exist.
        """
        try:
            dataset_info = ZENODO_DATASETS[psr_name][group][version]
        except KeyError:
            print(f"Unknown dataset: {psr_name}/{group}/{version}")
            return None

        output_dir = self.base_dir / psr_name / group / version
        output_dir.mkdir(parents=True, exist_ok=True)

        existing = [
            f for f in output_dir.iterdir() if f.is_file() and f.name != "md5sums.txt"
        ]
        if existing and not force:
            print(
                f"Already downloaded: {dataset_info['name']} ({len(existing)} files in {output_dir})"
            )
            return output_dir

        print(
            f"\nDownloading: {dataset_info['name']}  (Zenodo {dataset_info['zenodo_id']})"
        )
        success = self._download_record(dataset_info["zenodo_id"], output_dir)
        if success:
            return output_dir
        print(f"Download failed. Get it manually from: {dataset_info['url']}")
        return None

    def list_available_datasets(self) -> None:
        """Print all available datasets."""
        print("\n" + "=" * 70)
        print("AVAILABLE NICER DATASETS FROM ZENODO")
        print("=" * 70)
        for psr, psr_data in ZENODO_DATASETS.items():
            print(f"\n{psr}:")
            for group, group_data in psr_data.items():
                print(f"  {group.upper()}:")
                for version, info in group_data.items():
                    print(f"    [{version}] {info['name']}")
                    print(f"      Zenodo: {info['zenodo_id']}  —  {info['url']}")
        print()


if __name__ == "__main__":
    downloader = ZenodoDownloader()
    downloader.list_available_datasets()
