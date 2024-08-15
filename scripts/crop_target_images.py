from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).parents[1]
NORMAL_DIR = ROOT_DIR.joinpath('data', 'necp_target', 'back', 'subset')
ANOMALY_DIR = ROOT_DIR.joinpath('data', 'necp', 'back', 'ng')
OUTPUT_DIR = NORMAL_DIR.parent

def crop_and_save_images(input_dir: Path, output_dir: Path, left_upper: tuple[int, int], size: int = 128):
    """画像の指定領域を切り抜いて保存する

    Args:
        input_dir (Path): 入力画像のディレクトリ
        output_dir (Path): 出力先のディレクトリ
        left_upper (tuple[int, int]): 指定領域の左上の点の座標.
        size (int, optional): 切り抜く画像サイズ. Defaults to 256.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    exts = ['png', 'jpg', 'jpeg']
    for ext in exts:
        images.extend(list(input_dir.glob(f'*.{ext}')))

    x, y = left_upper
    for image_path in tqdm(images):
        img = Image.open(image_path)
        img_crop = img.crop((x, y, x + size, y + size))
        file_name = f'{image_path.stem}_{out_dir.parents[0].stem}.png'
        img_crop.save(output_dir / file_name)


if __name__ == '__main__':
    left_uppers = {
        'CN300_1': (670, 90),
        'CN300_2': (840, 90),
        'CN500_1': (300, 90),
        'CN500_2': (430, 90),
        'SW200': (1020, 90),
    }  # 92, 23, 6

    for name, coords in left_uppers.items():
        out_dir = OUTPUT_DIR.joinpath(name, 'ng')
        crop_and_save_images(ANOMALY_DIR, out_dir, coords, size=256)
        # break
