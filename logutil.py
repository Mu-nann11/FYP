import logging
from pathlib import Path


def get_logger(config):
    name = "fiji_stitcher"
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_name = str(config.get("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    stitched_parent = Path(config["STITCHED_PARENT_DIR"])
    log_dir = stitched_parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logfile = log_dir / config["RUN_LOG_FILENAME"]
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(str(logfile), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("Logging to: %s", logfile)
    return logger
