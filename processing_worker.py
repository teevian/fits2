"""Private worker entry point; requests are created only by processing_jobs."""
import importlib
from pathlib import Path
import pickle
import sys


def main():
    root = Path(sys.argv[1])
    try:
        with (root / 'request.pkl').open('rb') as stream:
            module, name, args, kwargs = pickle.load(stream)
        function = getattr(importlib.import_module(module), name)
        result = (True, function(*args, **kwargs))
    except Exception as exc:
        result = (False, str(exc) or type(exc).__name__)
    with (root / 'result.pkl').open('wb') as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
