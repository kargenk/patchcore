from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data', 'kintsugi', 'image')


def split_and_save_images(input_dir: Path, output_dir: Path, size: int = 256):
    """画像を分割して保存する

    Args:
        input_dir (Path): 入力画像のディレクトリ
        output_dir (Path): 出力先のディレクトリ
        size (int, optional): 分割後の画像サイズ. Defaults to 256.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg'))

    for image_path in tqdm(images):
        img = Image.open(image_path)

        # sizeの倍数に拡大してから分割
        scale_factor = int(max(img.size) / size) + 1
        up_size = size * scale_factor
        # scale_factor = 3
        # up_size = 768
        img = img.resize((up_size, up_size))
        for i in range(scale_factor):
            for j in range(scale_factor):
                # 各パッチを取得
                box = (j * 256, i * 256, (j + 1) * 256, (i + 1) * 256)
                patch = img.crop(box)

                # 分割された画像を保存
                patch_name = f'{image_path.stem}_{i}_{j}.png'
                patch_path = output_dir / patch_name
                patch.save(patch_path)


if __name__ == '__main__':
    input_dir = DATA_DIR.joinpath('normal')
    output_dir = DATA_DIR.with_name('splitted').joinpath('normal')

    split_and_save_images(input_dir, output_dir, size=256)
