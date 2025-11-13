# model_fetch.py
import os
import shutil
from pathlib import Path

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def download_from_url(url: str, dst: Path):
    import urllib.request
    ensure_dir(dst)
    print(f"[model_fetch] Downloading model from URL: {url}")
    urllib.request.urlretrieve(url, dst.as_posix())
    print(f"[model_fetch] Saved to: {dst}")

def download_from_gdrive(file_id: str, dst: Path):
    import gdown
    ensure_dir(dst)
    print(f"[model_fetch] Downloading model from Google Drive file id: {file_id}")
    gdown.download(id=file_id, output=dst.as_posix(), quiet=False)
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("Download failed or empty file.")

def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists() and src.is_file() and src.stat().st_size > 0:
        ensure_dir(dst)
        shutil.copy2(src, dst)
        print(f"[model_fetch] Copied local model: {src} -> {dst}")
        return True
    return False

def ensure_model() -> str:
    """
    best.pt のローカルパスを返す。優先順位：
      1) ./models/best.pt が既にある
      2) 環境変数 MODEL_URL（直リンク）でダウンロード
      3) 環境変数 GDRIVE_FILE_ID（Google Drive）でダウンロード
      4) リポジトリ内の runs/**/best.pt を自動探索してコピー
    """
    target = Path("models/best.pt")
    if target.exists() and target.stat().st_size > 0:
        return target.as_posix()

    model_url = os.getenv("MODEL_URL", "").strip()
    if model_url:
        download_from_url(model_url, target)
        return target.as_posix()

    gdrive_id = os.getenv("GDRIVE_FILE_ID", "").strip()
    if gdrive_id:
        download_from_gdrive(gdrive_id, target)
        return target.as_posix()

    candidates = list(Path(".").glob("runs/**/best.pt")) + list(Path(".").glob("**/runs/**/best.pt"))
    for c in candidates:
        if copy_if_exists(c, target):
            return target.as_posix()

    raise FileNotFoundError(
        "best.pt が見つかりません。\n"
        "以下のいずれかを設定してください：\n"
        " - ./models/best.pt を直接配置\n"
        " - 環境変数 MODEL_URL に直リンクURL\n"
        " - 環境変数 GDRIVE_FILE_ID にGoogle Driveのfile id\n"
        " - リポジトリ内に runs/**/best.pt を置く（自動コピー）\n"
    )
