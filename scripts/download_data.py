"""Download ReasonDrive annotations and required nuScenes images from official mirrors."""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
from data_utils import write_json

QA_REPO = 'ac4462/DriveLM-reasoning'
QA_REVISION = 'd9a27e1dfdd1f9441fa6a31e043b536d1798a4d4'
IMAGE_REPO = 'OpenDriveLab/DriveLM'
IMAGE_REVISION = 'ae973fbd4e8d4684af4ab234d504bd6c5e946868'
# Both mirrors are linked by OpenDriveLab/DriveLM's official data preparation docs.
ARCHIVES = [
    ('drivelm_nus_imgs_train.zip', 3483205396, '1DeosPGYeM2gXSChjMODGsQChZyYDmaUz'),
    ('drivelm_nus_imgs_val.zip', 704864335, '18f8ygNxGZWat-crUjroYuQbd39Sk9xCo'),
]


def annotations(path):
    path = Path(path)
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        from datasets import load_dataset
        rows = list(load_dataset(QA_REPO, revision=QA_REVISION, split='train'))
        write_json(path, rows)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Annotation file must contain a nonempty JSON array')
    return rows


def required_images(rows):
    required = set()
    for row in rows:
        images = row['image']
        if not isinstance(images, list) or not images:
            raise ValueError('Expected a nonempty image path list')
        for value in images:
            if not isinstance(value, str) or not value.startswith('../nuscenes/samples/'):
                raise ValueError(f'Unexpected dataset image path: {value!r}')
            relative = PurePosixPath(value.removeprefix('../nuscenes/'))
            if '..' in relative.parts or '\\' in value or len(relative.parts) != 3:
                raise ValueError(f'Unsafe dataset image path: {value!r}')
            required.add(str(relative))
    return required


def valid_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def destination(root, relative):
    root = Path(root).resolve()
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Image path escapes output directory: {relative}')
    return target


def missing_images(root, required):
    return {name for name in required if not valid_image(destination(root, name))}


def archive_key(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        return None
    parts = path.parts
    if 'samples' not in parts:
        return None
    suffix = parts[parts.index('samples'):]
    return str(PurePosixPath(*suffix)) if len(suffix) == 3 else None


def extract_required(archive, root, wanted):
    """Read selected members to EOF (ZIP CRC checked); never extract arbitrary paths."""
    found = set()
    with zipfile.ZipFile(archive) as bundle:
        members = {}
        for info in bundle.infolist():
            key = archive_key(info.filename)
            if key not in wanted or info.is_dir():
                continue
            if key in members:
                raise ValueError(f'Duplicate image in archive: {key}')
            if info.file_size > 100 * 1024**2:
                raise ValueError(f'Unexpected image size: {key}')
            members[key] = info
        from tqdm import tqdm
        for key, info in tqdm(members.items(), desc='Extract images'):
            target = destination(root, key)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.part')
            try:
                with bundle.open(info) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024*1024)
                if not valid_image(temporary):
                    raise ValueError(f'Invalid image inside archive: {key}')
                temporary.replace(target)
                found.add(key)
            finally:
                temporary.unlink(missing_ok=True)
    return found


def fetch_archive(name, size, drive_id, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / name
    if local.is_file() and zipfile.is_zipfile(local):
        return local
    if shutil.disk_usage(cache).free < 2*size + 512*1024**2:
        raise RuntimeError(f'Need at least {(2*size+512*1024**2)/1024**3:.1f} GiB free for {name} and extraction')
    print(f'Downloading official {name}: approximately {size/1e9:.2f} GB', flush=True)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    try:
        archive = Path(hf_hub_download(IMAGE_REPO, name, repo_type='dataset', revision=IMAGE_REVISION))
    except (HfHubHTTPError, OSError) as error:
        print(f'Hugging Face unavailable ({type(error).__name__}); trying official public Google Drive mirror.', flush=True)
        import gdown
        temporary = str(local) + '.download'
        if Path(temporary).is_file() and not zipfile.is_zipfile(temporary):
            Path(temporary).unlink()  # Invalid completed HTML/ZIP; interrupted .part files stay resumable.
        try:
            downloaded = gdown.download(id=drive_id, output=temporary, quiet=False, resume=True, use_cookies=False)
        except Exception as mirror_error:
            raise RuntimeError(
                '공식 이미지 다운로드를 완료하지 못했습니다.\n'
                'Hugging Face 로그인과 데이터셋 접근 승인은 별개입니다.\n'
                'https://huggingface.co/datasets/OpenDriveLab/DriveLM 에서 터미널과 같은 계정으로 '
                '로그인한 뒤 데이터셋 접근 신청/동의를 완료하고 승인 상태를 확인하세요.\n'
                'Google Drive 대체 경로도 현재 할당량 또는 접근 문제로 실패했습니다.\n'
                '이미 로그인했다면 재로그인할 필요 없이, 접근 승인 후 bash scripts/run_mac.sh 를 다시 실행하세요.\n'
                '기존 파일과 다운로드 중간 파일은 보존됩니다.'
            ) from mirror_error
        if not downloaded or not zipfile.is_zipfile(temporary):
            raise RuntimeError('Mirror returned no valid ZIP; retry later or authenticate to the official HF dataset')
        Path(temporary).replace(local)
        archive = local
    if not zipfile.is_zipfile(archive):
        raise ValueError(f'Not a valid ZIP archive: {archive}')
    return archive


def download(data_path='data/drivelm.json', image_root='data/nuscenes', cache='data/downloads'):
    rows = annotations(data_path)
    required = required_images(rows)
    root = Path(image_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    missing = {name for name in tqdm(sorted(required), desc='Check existing images')
               if not valid_image(destination(root, name))}
    print(f'{len(rows)} QA, {len(required)} unique images; {len(missing)} missing/corrupt', flush=True)
    sources = []
    for name, size, drive_id in ARCHIVES:
        if not missing:
            break
        archive = fetch_archive(name, size, drive_id, cache)
        available = shutil.disk_usage(root).free
        with zipfile.ZipFile(archive) as bundle:
            needed = sum(i.file_size for i in bundle.infolist() if archive_key(i.filename) in missing)
        if available < needed + 256*1024**2:
            raise RuntimeError('Insufficient disk space in the image output directory')
        restored = extract_required(archive, root, missing)
        missing -= restored
        sources.append({'archive': name, 'extracted': len(restored)})
    report = {'qa_count': len(rows), 'required_images': len(required), 'missing': sorted(missing),
              'image_root': str(root), 'qa_repo': QA_REPO, 'qa_revision': QA_REVISION,
              'image_repo': IMAGE_REPO, 'image_revision': IMAGE_REVISION, 'downloads_this_run': sources}
    write_json(Path(data_path).parent / 'download_report.json', report)
    if missing:
        raise RuntimeError(f'{len(missing)} required images unavailable in official archives; see download_report.json')
    print(f'All required images ready: {root}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='data/drivelm.json')
    parser.add_argument('--image_root', default='data/nuscenes')
    parser.add_argument('--cache', default='data/downloads')
    args = parser.parse_args()
    try:
        download(args.data_path, args.image_root, args.cache)
    except (RuntimeError, OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"\n데이터 준비 중단: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
