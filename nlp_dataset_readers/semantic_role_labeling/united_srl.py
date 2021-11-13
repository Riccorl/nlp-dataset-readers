from pathlib import Path
from typing import Union, Any

from semantic_role_labeling.srl_reader import SrlReader


class UnitedSrlReader(SrlReader):
    def read(self, file_path: Union[str, Path], *args, **kwargs) -> Any:
        pass
