from pathlib import Path

CHANNELS = ("DAPI", "HER2", "PR", "ER")


def get_all_level1_directories(config):
    root = Path(config["DEFAULT_ROOT_DIR"])
    raw_name = str(config.get("RAW_DATA_DIR_NAME", "Raw_Data")).strip()

    candidate_roots = [
        root / raw_name,
        root,
    ]

    found = []
    seen = set()

    for base in candidate_roots:
        if not base.exists() or not base.is_dir():
            continue

        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue

            has_channel = any((p / ch).is_dir() for ch in CHANNELS)
            if has_channel:
                resolved = str(p.resolve())
                if resolved not in seen:
                    found.append(resolved)
                    seen.add(resolved)

    return found
