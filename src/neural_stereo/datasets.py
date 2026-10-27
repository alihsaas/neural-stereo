import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import DATA_DIR, DATA_ZIP_URL, GT_ZIP_URL, SCENE_FILES, SCENE_ROOT, SCENES

USER_AGENT = "neural-stereo-dataset-fetch/0.1"


def scene_dir(scene: str) -> Path:
    return SCENE_ROOT / scene


def scene_complete(scene: str) -> bool:
    return all((scene_dir(scene) / name).is_file() for name in SCENE_FILES)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as stream:
        shutil.copyfileobj(response, stream)
    partial.replace(destination)


def extract_scene_members(archive: Path) -> None:
    wanted_prefixes = tuple(f"MiddEval3/trainingQ/{scene}/" for scene in SCENES)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.startswith(wanted_prefixes) and not member.endswith("/"):
                bundle.extract(member, DATA_DIR)


def ensure_data() -> None:
    if all(scene_complete(scene) for scene in SCENES):
        print("dataset already present")
        return
    downloads = DATA_DIR / "downloads"
    for url in (DATA_ZIP_URL, GT_ZIP_URL):
        archive = downloads / url.rsplit("/", 1)[1]
        download_file(url, archive)
        extract_scene_members(archive)
    missing = [scene for scene in SCENES if not scene_complete(scene)]
    if missing:
        raise RuntimeError(f"scenes incomplete after extraction: {missing}")
    print(f"dataset ready under {SCENE_ROOT}")
