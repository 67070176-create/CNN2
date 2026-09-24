from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / 'dataset' / 'round2'
TEST_DEST = ROOT / 'test_dataset'
ARCHIVE_DEST = ROOT / 'archive2' / 'test_500'

FOLDER_TO_CLASS_ID = {
    '161': 101, '162': 102, '163': 103, '164': 104, '167': 105, '168': 106, '169': 107,
    '170': 108, '171': 109, '173': 110, '175': 111, '176': 112, '177': 113, '178': 114,
    '179': 115, '180': 116, '181': 117, '182': 118, '183': 119, '184': 120, '185': 121,
    '186': 122, '187': 123, '188': 124, '189': 125, '190': 126, '191': 127, '192': 128,
    '193': 129, '194': 130, '195': 131, '196': 132, '197': 133, '199': 134, '200': 135,
    '201': 136, '202': 137, '203': 138, '204': 139, '205': 140, '206': 141,
    '207': 201, '209': 202, '210': 203, '212': 204, '213': 205, '214': 206, '215': 207,
    '216': 208, '217': 209, '224': 210, '225': 211, '226': 212, '227': 213, '228': 214,
    '229': 215, '230': 216, '231': 217, '232': 218, '233': 219, '234': 220, '236': 221,
    '240': 301, '241': 302, '242': 303, '243': 304, '244': 305, '245': 306, '246': 307,
    '247': 308, '248': 309, '249': 310,
}


def collect_samples():
    samples = []
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f'Missing source folder: {SOURCE_ROOT}')

    for folder in sorted(SOURCE_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        class_id = FOLDER_TO_CLASS_ID.get(folder.name)
        if class_id is None:
            continue

        for file in sorted(folder.iterdir()):
            if not file.is_file():
                continue
            if file.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            samples.append((file, class_id))

    return samples


def prepare_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for child in folder.iterdir():
        if child.is_file() and child.name != 'test.csv':
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def main():
    selected = collect_samples()[:500]
    if len(selected) < 500:
        raise RuntimeError(f'Only {len(selected)} valid images were found in {SOURCE_ROOT}. Need at least 500.')

    prepare_folder(TEST_DEST)
    prepare_folder(ARCHIVE_DEST)

    rows = []
    for source_file, class_id in selected:
        dst_name = source_file.name
        shutil.copy2(source_file, TEST_DEST / dst_name)
        shutil.copy2(source_file, ARCHIVE_DEST / dst_name)

        rows.append({
            'id': source_file.stem,
            'gt_path': f'./test_dataset/{dst_name}',
            'gt_class_id': class_id,
        })

    df = pd.DataFrame(rows)
    df.to_csv(TEST_DEST / 'test.csv', index=False)
    df[['id', 'gt_class_id']].to_csv(ARCHIVE_DEST / 'test.csv', index=False)

    print(f'Created {len(rows)} images in {TEST_DEST}')
    print(f'Created {len(rows)} images in {ARCHIVE_DEST}')
    print(f'CSV sample:')
    print(df.head().to_string(index=False))


if __name__ == '__main__':
    main()
