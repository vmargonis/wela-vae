from pathlib import Path


class _Config:
    def __init__(self, project="wela-vae"):
        self.project_name = project
        self.project_path = Path(__file__).parent.parent

    @staticmethod
    def create(path):
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def blobs_path(self):
        path = self.project_path / "blobs"
        return self.create(path)

    @property
    def results_path(self):
        path = self.project_path / "results"
        return self.create(path)


Config = _Config()
