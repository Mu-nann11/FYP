import json
from pathlib import Path

class Config:
    def __init__(self, config_path: Path = None):
        if config_path is None:
            # 默认指向当前目录下的 fiji_config.json
            config_path = Path(__file__).parent / "fiji_config.json"
        
        self.config_path = config_path
        self.data = {}
        self.load()

    def load(self):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            print(f"Warning: Config file {self.config_path} not found. Using empty config.")

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    @property
    def crop_root(self):
        return Path(self.get("CROP_OUTPUT_DIR", "/results/crop"))

    @property
    def batch_output_dir(self):
        return Path(self.get("BATCH_OUTPUT_DIR", "/results/batch_features"))

    @property
    def expansion_distance(self):
        return self.get("EXPANSION_DISTANCE", 15)

# 实例化全局配置对象
config = Config()
