import glob
from PIL import Image

out = r"D:\Development\TSAI\EVA\MaskRCNN Dataset\Foreground\masks\{}"

path = r"D:\Development\TSAI\EVA\MaskRCNN Dataset\Foreground\resize\*.*"

for file in glob.glob(path):
    im = Image.open(file, 'r')
    file_name = file.split("\\")[-1]
    rgb_data = im.tobytes("raw", "RGB")
    alpha_data = im.tobytes("raw", "A")
    alpha_image = Image.frombytes("L", im.size, alpha_data)
    alpha_image.save(out.format(file_name))
