from sys import argv

from weather_doc_extractor.cli import run


if __name__ == "__main__":
    raise SystemExit(run(argv[1:]))
