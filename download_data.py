from pathlib import Path
from urllib.request import Request, urlopen

from config import DATASET_URL, RAW_FILE


def download(url: str = DATASET_URL, destination: Path = RAW_FILE) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "NLP-homework/1.0"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())
    print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes)")
    return destination


if __name__ == "__main__":
    download()

