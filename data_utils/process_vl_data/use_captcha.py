# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""

from __future__ import annotations
import os
import secrets
import typing as t
from PIL import Image  # Ensure this is properly imported from PIL
from PIL.Image import new as createImage, Transform, Resampling
from PIL.ImageDraw import Draw, ImageDraw
from PIL.ImageFilter import SMOOTH
from PIL.ImageFont import FreeTypeFont, truetype
from io import BytesIO
import json
__all__ = ['ImageCaptcha']


ColorTuple = t.Union[t.Tuple[int, int, int], t.Tuple[int, int, int, int]]

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
# DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono.ttf')]
DEFAULT_FONTS = ["/user/share/fonts/Calibri.ttf"]



class ImageCaptcha:
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    lookup_table: list[int] = [int(i * 1.97) for i in range(256)]
    character_offset_dx: tuple[int, int] = (0, 2)
    character_offset_dy: tuple[int, int] = (0, 3)
    character_rotate: tuple[int, int] = (-0.02, 0.02)
    character_warp_dx: tuple[float, float] = (0.0, 0.1)
    character_warp_dy: tuple[float, float] = (0.0, 0.1)
    word_space_probability: float = 0.5
    word_offset_dx: float = 0.03

    def __init__(
            self,
            width: int = 160,
            height: int = 60,
            fonts: list[str] | None = None,
            font_sizes: tuple[int, ...] | None = None,
            background_image_path: str | None = None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (80, 90, 100)
        self._truefonts: list[FreeTypeFont] = []
        self._background_image_path = background_image_path

    @property
    def background_image(self) -> Image | None:
        """Load and return the background image if provided."""
        if self._background_image_path and os.path.exists(self._background_image_path):
            return Image.open(self._background_image_path)
        return None

    def create_captcha_image(
            self,
            chars: str,
            color: ColorTuple,
            background: ColorTuple) -> Image:
        """Create the CAPTCHA image itself."""

        # Load background image if available
        background_image = self.background_image
        if background_image:
            # Resize the background image to fit the CAPTCHA size
            background_image = background_image.resize((self._width, self._height))
            image = background_image.convert('RGB')
        else:
            # Create a solid color background if no image is provided
            image = createImage('RGB', (self._width, self._height), background)

        draw = Draw(image)

        # 保存每个字符的图像
        images: list[Image] = []
        for c in chars:
            if secrets.randbits(32) / (2 ** 32) > self.word_space_probability:
                images.append(self._draw_character(" ", draw, color))
            # 为每个字符传递一个新的随机颜色
            images.append(self._draw_character(c, draw, color))

        # 计算所有字符的总宽度
        text_width = sum([im.size[0] for im in images])

        # 计算居中位置
        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        # 设置字符位置，使其居中
        average = int(text_width / len(chars))  # 每个字符的平均宽度
        rand = int(self.word_offset_dx * average)  # 随机偏移量
        offset = int((self._width - text_width) / 2)  # 将文本居中

        # 将字符依次添加到图像中
        for im in images:
            w, h = im.size
            mask = im.convert('L').point(self.lookup_table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + (-secrets.randbelow(rand + 1))

        # 如果图像宽度大于预设宽度，重新调整图像大小
        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars: str,
                       bg_color: ColorTuple | None = None,
                       fg_color: ColorTuple | None = None) -> Image:
        """Generate the image of the given characters.

        :param chars: text to be generated.
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a).
        """
        background = bg_color if bg_color else self.create_random_color(238, 255)
        random_fg_color = self.create_random_color(10, 200, secrets.randbelow(36) + 220)
        color: ColorTuple = fg_color if fg_color else random_fg_color

        im = self.create_captcha_image(chars, color, background)
        # self.create_noise_dots(im, color)
        # self.create_noise_curve(im, color)
        im = im.filter(SMOOTH)
        return im
    @property
    def truefonts(self) -> list[FreeTypeFont]:
        if self._truefonts:
            return self._truefonts
        self._truefonts = [
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ]
        return self._truefonts

    @staticmethod
    def create_noise_curve(image: Image, color: ColorTuple) -> Image:
        w, h = image.size
        x1 = secrets.randbelow(int(w / 5) + 1)
        x2 = secrets.randbelow(w - int(w / 5) + 1) + int(w / 5)
        y1 = secrets.randbelow(h - 2 * int(h / 5) + 1) + int(h / 5)
        y2 = secrets.randbelow(h - y1 - int(h / 5) + 1) + y1
        points = [x1, y1, x2, y2]
        end = secrets.randbelow(41) + 160
        start = secrets.randbelow(21)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(
            image: Image,
            color: ColorTuple,
            width: int = 3,
            number: int = 30) -> Image:
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = secrets.randbelow(w + 1)
            y1 = secrets.randbelow(h + 1)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_random_color(
            self,
            start_id: int,
            end_id: int,
            opacity: int | None = None) -> ColorTuple:
        red = secrets.randbelow(end_id - start_id + 1) + start_id
        green = secrets.randbelow(end_id - start_id + 1) + start_id
        blue = secrets.randbelow(end_id - start_id + 1) + start_id
        if opacity is None:
            return red, green, blue
        return red, green, blue, opacity

    def _draw_character(
            self,
            c: str,
            draw: ImageDraw,
            color: ColorTuple) -> Image:
        font = secrets.choice(self.truefonts)
        _, _, w, h = draw.multiline_textbbox((1, 1), c, font=font)

        dx1 = secrets.randbelow(self.character_offset_dx[1] - self.character_offset_dx[0] + 1) + \
              self.character_offset_dx[0]
        dy1 = secrets.randbelow(self.character_offset_dy[1] - self.character_offset_dy[0] + 1) + \
              self.character_offset_dy[0]
        im = createImage('RGBA', (int(w) + dx1, int(h) + dy1))
        # 为每个字符选择一个随机颜色
        random_color = self.create_random_color(0, 255)  # 这里可以调整随机颜色的范围
        Draw(im).text((dx1, dy1), c, font=font, fill=random_color)

        # rotate
        im = im.crop(im.getbbox())
        im = im.rotate(
            self.character_rotate[0] + (secrets.randbits(32) / (2 ** 32)) * (
                        self.character_rotate[1] - self.character_rotate[0]),
            Resampling.BILINEAR,
            expand=True,
        )

        # warp
        dx2 = w * (secrets.randbits(32) / (2 ** 32)) * (self.character_warp_dx[1] - self.character_warp_dx[0]) + \
              self.character_warp_dx[0]
        dy2 = h * (secrets.randbits(32) / (2 ** 32)) * (self.character_warp_dy[1] - self.character_warp_dy[0]) + \
              self.character_warp_dy[0]
        x1 = int(secrets.randbits(32) / (2 ** 32) * (dx2 - (-dx2)) + (-dx2))
        y1 = int(secrets.randbits(32) / (2 ** 32) * (dy2 - (-dy2)) + (-dy2))
        x2 = int(secrets.randbits(32) / (2 ** 32) * (dx2 - (-dx2)) + (-dx2))
        y2 = int(secrets.randbits(32) / (2 ** 32) * (dy2 - (-dy2)) + (-dy2))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((w2, h2))
        im = im.transform((int(w), int(h)), Transform.QUAD, data)
        return im


    def generate(self, chars: str, format: str = 'png',
                 bg_color: ColorTuple | None = None,
                 fg_color: ColorTuple | None = None) -> BytesIO:
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a).
        """
        im = self.generate_image(chars, bg_color=bg_color, fg_color=fg_color)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars: str, output: str, format: str = 'png',
              bg_color: ColorTuple | None = None,
              fg_color: ColorTuple | None = None) -> None:
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a).
        """
        im = self.generate_image(chars, bg_color=bg_color, fg_color=fg_color)
        im.save(output, format=format)



save_dict = []
save_img_dir='/home/jz08/zhouzy/data/vl_data/verification_code/wooden'
os.makedirs(save_img_dir, exist_ok=True)
img = ImageCaptcha(width=640, height=480, fonts=None, font_sizes=(100, 120, 140), background_image_path='/home/jz08/zhouzy/data/wooden.jpg')
gen_img_flg = False
gen_llava_format = True
gen_vlmeval_format = False
q_list = [
    'Solve the question: point out the equation.'
]
cnt = 0
for i in range(1,46):
    for j in range(1,46):
        if gen_llava_format:
            x={}
            question = f"<image>\nSolve the question: point out the equation."
            answer = f"On the table is {str(i)}+{str(j)}. {str(i)} plus {str(j)} equals {str(i+j)}."
            template = [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
            x["id"] = 'verify_wooden_'+str(i)+'_'+str(j)
            x["image"] = [os.path.join('wooden', str(i) + '_' + str(j) + '.jpg')]
            x["conversations"] = template
            save_dict.append(x)

        elif gen_vlmeval_format:
            if len(save_dict) == 0:
                save_dict.append(['index','question','A','B','C','D','answer','category','abcLabel','image_path'])
            for q in q_list:
                save_dict.append([str(cnt), q, '', '', '', '', '', 'verification_train', '', str(i) + '_' + str(j) + '.jpg'])
                cnt += 1
        if gen_img_flg:
            image = img.generate_image(f'{str(i)}+{str(j)}={str(i+j)}')
            image.save(os.path.join(save_img_dir, str(i)+'_'+str(j)+'.jpg'))

if gen_llava_format:
    save_dir = '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/verification_code_0410.json'
    json.dump(save_dict, open(save_dir, 'w'), indent=4)
    print("Json created successfully.")

elif gen_vlmeval_format:
    import csv
    filename = '/home/jz08/LMUData/MMRO_mini_0408_veri_train.tsv'
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(save_dict)
    print(f"TSV file '{filename}' created successfully.")

