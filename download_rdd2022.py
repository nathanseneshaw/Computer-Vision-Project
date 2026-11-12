import urllib.request
import zipfile
from pathlib import Path

URL = "https://ndownloader.figshare.com/files/38030910"
SIZE = 13_264_172_619

DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "RDD2022.zip"
OUT_DIR = DATA_DIR / "RDD2022"


def download():
    done = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0
    if done >= SIZE:
        return
    request = urllib.request.Request(URL, headers={"Range": f"bytes={done}-"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            done = 0
        with open(ZIP_PATH, "ab" if done else "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                done += len(chunk)
                print(f"\rDownloading: {done / SIZE:.1%}", end="", flush=True)
    print()


def extract():
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for member in archive.namelist():
            if not member.endswith(".zip"):
                continue
            print(f"Extracting {member}")
            country_zip = Path(archive.extract(member, DATA_DIR))
            with zipfile.ZipFile(country_zip) as country:
                country.extractall(OUT_DIR)
            country_zip.unlink()
    ZIP_PATH.unlink()


def main():
    if OUT_DIR.exists() and not ZIP_PATH.exists():
        print(f"RDD2022 is already downloaded to {OUT_DIR}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    download()
    extract()
    print(f"Done: {OUT_DIR}")


if __name__ == "__main__":
    main()
