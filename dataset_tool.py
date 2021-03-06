import os
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps, UnidentifiedImageError


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def get_all_files(folder, files=None):
    if files is None:
        files = []

    for file in os.listdir(folder):
        file = os.path.join(folder, file)

        if os.path.isdir(file):
            get_all_files(file, files)
        else:
            files.append(file)

    return files


def remove_folder(folder):
    for file in os.listdir(folder):
        file = os.path.join(folder, file)
        if os.path.isdir(file):
            remove_folder(file)
        else:
            os.remove(file)

    os.removedirs(folder)


def transform(img):
    if args.transform == 'center-crop':
        img_ratio = img.size[0] / float(img.size[1])
        ratio = args.width / args.height

        if ratio > img_ratio:
            img = img.resize((args.width, int(round(args.width * img.size[1] / img.size[0]))), Image.BICUBIC)
            img = img.crop((0, round((img.size[1] - args.height) / 2),
                            img.size[0], round((img.size[1] + args.height) / 2)))
        else:
            img = img.resize((int(round(args.height * img.size[0] / img.size[1])), args.height), Image.BICUBIC)
            img = img.crop((round((img.size[0] - args.width) / 2), 0,
                            round((img.size[0] + args.width) / 2), img.size[1]))

    if args.transform == 'resize-and-pad':
        img = ImageOps.pad(img, (args.width, args.height))

    return img


def build_dataset():
    if os.path.exists(args.dest) and len(os.listdir(args.dest)) > 0:
        if args.clear_dest:
            remove_folder(args.dest)
            os.makedirs(args.dest)
        else:
            print('Destination folder exists and is not empty!')
            return
    else:
        os.makedirs(args.dest, exist_ok=True)

    index = 0
    for file in tqdm(get_all_files(args.source), smoothing=0.0):
        try:
            img = Image.open(file).convert(args.format)
            img = transform(img)

            folder = index // 1000
            if index % 1000 == 0:
                os.makedirs(f"{args.dest}/{folder:04}")

            img.save(f"{args.dest}/{folder:04}/{index%1000:04}.png")
            index += 1
        except UnidentifiedImageError:
            print(f"Failed to load image '{file}!'")
            continue


parser = argparse.ArgumentParser(description='Dataset converter utility.')
parser.add_argument('--source', type=dir_path, help='The source folder.')
parser.add_argument('--dest', type=str, help='The destination folder.')
parser.add_argument('--width', type=int, help='The image width.')
parser.add_argument('--height', type=int, help='The image height.')
parser.add_argument('--format', choices=['RGB', 'L'], default='RGB', help='The image format.')
parser.add_argument('--transform', choices=['center-crop', 'resize-and-pad'],
                    default='center-crop', help='How to transform the image.')
parser.add_argument('--clear-dest', type=bool, default=False,
                    help='Force clear the destination folder if not empty.')
args = parser.parse_args()

if __name__ == '__main__':
    build_dataset()
