from pathlib import Path
from typing import Union, Any

from common.reader import Reader


class SrlReader(Reader):
    def read(self, file_path: Union[str, Path]) -> Any:
        pass
