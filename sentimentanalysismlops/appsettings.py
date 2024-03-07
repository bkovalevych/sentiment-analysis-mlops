import json


class AppSettings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppSettings, cls).__new__(cls)
            cls._instance.config_path = "sentimentanalysismlops/config.json"
            cls.config = cls._instance._read_config()
        return cls._instance

    def __getitem__(self, item: str):
        return self.config.get(item)

    def _read_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            return json.load(f)

