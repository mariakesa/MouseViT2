import os
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

class AllenAPI:
    """Singleton class for accessing Allen Brain Observatory API"""
    
    _instance = None
    _boc = None  # Lazy-loaded BrainObservatoryCache instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def boc(self):
        """Lazy-load BrainObservatoryCache only when accessed."""
        if self._boc is None:
            allen_cache_path = os.environ.get('CAIM_ALLEN_CACHE_PATH')
            if not allen_cache_path:
                raise ValueError("AllenAPI requires a valid cache path. Set `CAIM_ALLEN_CACHE_PATH` in .env.")

            manifest_path = Path(allen_cache_path) / 'brain_observatory_manifest.json'
            self._boc = BrainObservatoryCache(manifest_file=str(manifest_path))

        return self._boc

    def get_boc(self):
        """Retrieve the BrainObservatoryCache object, ensuring it is initialized."""
        return self.boc

# Create a global instance so that all files can use it
allen_api = AllenAPI()
