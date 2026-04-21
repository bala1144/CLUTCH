from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets"


def get_assets_root() -> Path:
    for env_key in ("ASSETS_PATH", "ASSESTS_PATH"):
        env_value = os.environ.get(env_key)
        if env_value:
            return Path(env_value).expanduser().resolve()
    return DEFAULT_ASSETS_DIR


def get_asset_path(*parts: str) -> Path:
    return get_assets_root().joinpath(*parts)


def get_mano_model_path() -> str:
    return str(get_asset_path("mano_v1_2"))
