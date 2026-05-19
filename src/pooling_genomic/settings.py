from pydantic import BaseSettings, validator
from pathlib import Path


class PoolingGenomicSettings(BaseSettings):
    path_data: Path = Path.home() / 'pooling_genomic' / 'data'
    path_results: Path = Path.home() / 'pooling_genomic' / 'results'

    class Config:
        env_prefix = 'POOLING_GENOMIC_'
        env_file = '.env'
