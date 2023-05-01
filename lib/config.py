from pathlib import Path

from lib.local_settings import BLOBS_PATH, RESULTS_PATH


class _Config:
    def __init__(self, project="wela-vae"):
        self.project_name = project

    @staticmethod
    def create(path):
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def blobs_path(self):
        path = Path(BLOBS_PATH)
        return self.create(path)

    @property
    def results_path(self):
        path = Path(RESULTS_PATH)
        return self.create(path)


Config = _Config()
