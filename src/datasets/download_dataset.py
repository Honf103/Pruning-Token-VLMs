#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import zipfile
import warnings
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URLS = {
    # VQA v2 annotations / questions
    "annotations_train": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "annotations_val":   "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "questions_train":   "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "questions_val":     "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "questions_test":    "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",

    # COCO images used by VQA v2
    "images_train":      "https://images.cocodataset.org/zips/train2014.zip",
    "images_val":        "https://images.cocodataset.org/zips/val2014.zip",
    "images_test":       "https://images.cocodataset.org/zips/test2015.zip",
}

SPLITS = {
    "trainval": [
        "annotations_train",
        "annotations_val",
        "questions_train",
        "questions_val",
        "images_train",
        "images_val",
    ],
    "test": [
        "questions_test",
        "images_test",
    ],
    "all": [
        "annotations_train",
        "annotations_val",
        "questions_train",
        "questions_val",
        "questions_test",
        "images_train",
        "images_val",
        "images_test",
    ],
}


def sizeof_fmt(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
    })
    return session


def is_coco_image_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "images.cocodataset.org" in host


def expected_extract_outputs(zip_name: str, extract_dir: Path):
    name = zip_name.lower()
    if "train2014.zip" in name:
        return [extract_dir / "train2014"]
    if "val2014.zip" in name:
        return [extract_dir / "val2014"]
    if "test2015.zip" in name:
        return [extract_dir / "test2015"]
    if "v2_annotations_train_mscoco.zip" in name:
        return [extract_dir / "v2_mscoco_train2014_annotations.json"]
    if "v2_annotations_val_mscoco.zip" in name:
        return [extract_dir / "v2_mscoco_val2014_annotations.json"]
    if "v2_questions_train_mscoco.zip" in name:
        return [extract_dir / "v2_OpenEnded_mscoco_train2014_questions.json"]
    if "v2_questions_val_mscoco.zip" in name:
        return [extract_dir / "v2_OpenEnded_mscoco_val2014_questions.json"]
    if "v2_questions_test_mscoco.zip" in name:
        return [extract_dir / "v2_OpenEnded_mscoco_test2015_questions.json"]
    return []


def already_extracted(zip_path: Path, extract_dir: Path) -> bool:
    outputs = expected_extract_outputs(zip_path.name, extract_dir)
    return len(outputs) > 0 and all(p.exists() for p in outputs)


def download_file(
    session: requests.Session,
    url: str,
    dst: Path,
    allow_insecure_coco_fallback: bool = True,
    chunk_size: int = 1024 * 1024,
):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    # thử chuẩn trước
    verify = True
    tried_insecure = False

    for attempt in range(2):
        try:
            if not verify:
                warnings.filterwarnings("ignore", message="Unverified HTTPS request")

            with session.get(url, stream=True, timeout=(30, 120), verify=verify) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0)) or None

                downloaded = 0
                start = time.time()

                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        elapsed = max(time.time() - start, 1e-6)
                        speed = downloaded / elapsed

                        if total is not None:
                            pct = downloaded * 100.0 / total
                            msg = (
                                f"\rDownloading {dst.name}: "
                                f"{pct:6.2f}% | {sizeof_fmt(downloaded)}/{sizeof_fmt(total)} "
                                f"| {sizeof_fmt(speed)}/s"
                            )
                        else:
                            msg = (
                                f"\rDownloading {dst.name}: "
                                f"{sizeof_fmt(downloaded)} | {sizeof_fmt(speed)}/s"
                            )
                        print(msg, end="", flush=True)

                print()
                tmp.replace(dst)
                return

        except requests.exceptions.SSLError as e:
            # chỉ fallback insecure cho ảnh COCO
            if allow_insecure_coco_fallback and verify and is_coco_image_url(url) and not tried_insecure:
                print(f"\n[SSL warning] {url}")
                print("  SSL hostname verification failed. Retrying this COCO image file with verify=False ...")
                verify = False
                tried_insecure = True
                continue
            raise RuntimeError(f"SSL error while downloading {url}: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Failed downloading {url}: {e}") from e


def unzip_file(zip_path: Path, extract_dir: Path, remove_zip: bool = False):
    print(f"Extracting {zip_path.name} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    if remove_zip:
        zip_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Download VQA v2 dataset")
    parser.add_argument("--root", type=str, default="./data/vqa_v2")
    parser.add_argument("--split", type=str, default="trainval", choices=["trainval", "test", "all"])
    parser.add_argument("--no-unzip", action="store_true")
    parser.add_argument("--remove-zip", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    archives_dir = root / "archives"
    extract_dir = root

    root.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    keys = SPLITS[args.split]

    print(f"Save to: {root.resolve()}")
    print(f"Split  : {args.split}")
    print("-" * 80)

    for key in keys:
        url = URLS[key]
        filename = url.split("/")[-1]
        dst = archives_dir / filename

        if dst.exists():
            print(f"[Skip zip exists] {dst.name}")
        elif (not args.no_unzip) and already_extracted(Path(filename), extract_dir):
            print(f"[Skip extracted] {filename}")
        else:
            download_file(session, url, dst)

        if not args.no_unzip:
            if already_extracted(dst if dst.exists() else Path(filename), extract_dir):
                print(f"[Skip extract exists] {filename}")
            elif dst.exists():
                unzip_file(dst, extract_dir, remove_zip=args.remove_zip)
            else:
                print(f"[Warn] Missing zip after download step: {filename}")

    print("\nDone.")
    print(f"Dataset root: {root.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)