import math
import cv2
import numpy as np
import os
import torch
import hashlib

torch.backends.cuda.matmul.allow_tf32 = True


def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)


class SSHPicture:
    """
    A class for generating a collage from SSH keys using Stable Diffusion 2.1

    Attributes:
        key_path (str): Path to the SSH key file.
        pipeline (DiffusionPipeline): The Stable Diffusion pipeline for generating images.

    Methods:
        get_image(seed: int, guidance_scale: float, prompt: str) -> np.ndarray:
            Generates an image based on the given seed, guidance scale, and prompt.

        paint_ssh(images: list[np.ndarray], texture: np.ndarray) -> np.ndarray:
            Creates a collage of images with a background texture.

        __call__() -> np.ndarray:
            Processes the SSH key file to generate a collage image.
    """
    guidance_divisor = 20
    num_bytes_per_seed = 5
    num_inference_steps = 6

    def __init__(self, key_path: str, model: str = "stabilityai/stable-diffusion-2-1-base",
                 output_path: str = "output") -> None:

        from diffusers import DPMSolverMultistepScheduler
        from diffusers import StableDiffusionPipeline, AutoencoderTiny

        if not isinstance(key_path, str):
            raise ValueError("key_path must be a string")
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"The file at {key_path} does not exist")

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model,
            use_safetensors=True,
        )

        self.pipeline.vae = AutoencoderTiny.from_pretrained(
            "sayakpaul/taesd-diffusers", use_safetensors=True,
        )

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.pipeline.safety_checker = dummy_safety_checker

        self.device = "cpu"

        self.pipeline = self.pipeline.to(self.device)
        self.key_path = key_path
        self.output_path = output_path

        self.cache_dir = "cache_images"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _make_cache_filename(self, seed: int, guidance_scale: float, prompt: str) -> str:
        key = f"{seed}_{guidance_scale}_{prompt}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.png")

    def get_image(self, seed: int, guidance_scale: float, prompt: str) -> np.ndarray:
        """
        Generates an image based on the given seed, guidance scale, and prompt.

        Args:
            seed (int): Random seed for image generation.
            guidance_scale (float): Scale for guidance.
            prompt (str): Text prompt for the image.

        Returns:
            np.ndarray: Generated image as a NumPy array.
        """

        cache_file = self._make_cache_filename(seed, guidance_scale, prompt)
        if os.path.exists(cache_file):
            return cv2.imread(cache_file, cv2.IMREAD_COLOR)

        generator = torch.Generator(self.device).manual_seed(seed)
        image = self.pipeline(
            prompt,
            num_inference_steps=self.__class__.num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale / self.__class__.guidance_divisor,
            height=512,
            width=512
        ).images[0]

        arr = np.array(image)
        cv2.imwrite(cache_file, arr)
        return arr

    def paint_ssh(self, images: list[np.ndarray], texture: np.ndarray) -> np.ndarray:
        """
        Creates a collage of images with a background texture.

        Args:
            images (list[np.ndarray]): List of images to include in the collage.
            texture (np.ndarray): Background texture image.

        Returns:
            np.ndarray: Final collage image.
        """
        target_img_size = (256, 256)
        spacing = 20
        border = 30
        img_border = 10

        resized_imgs = [cv2.resize(img, target_img_size, interpolation=cv2.INTER_AREA) for img in images]
        texture_size = (100, 100)
        texture = cv2.resize(texture, texture_size, interpolation=cv2.INTER_AREA)

        num_imgs = len(resized_imgs)
        grid_size = int(math.ceil(math.sqrt(num_imgs)))

        img_w, img_h = target_img_size
        sub_w = img_w + 2 * img_border
        sub_h = img_h + 2 * img_border

        total_width = grid_size * sub_w + (grid_size + 1) * spacing + 2 * border
        total_height = grid_size * sub_h + (grid_size + 1) * spacing + 2 * border

        final_image = 255 * np.ones((total_height, total_width, 3), dtype=np.uint8)

        fill_area = final_image[border:-border, border:-border]
        fill_h, fill_w, _ = fill_area.shape

        for y in range(0, fill_h, texture_size[1]):
            for x in range(0, fill_w, texture_size[0]):
                end_y = min(y + texture_size[1], fill_h)
                end_x = min(x + texture_size[0], fill_w)
                fill_area[y:end_y, x:end_x] = texture[0:(end_y - y), 0:(end_x - x)]

        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= num_imgs:
                    break

                rimg = resized_imgs[idx]
                sub_canvas = 255 * np.ones((sub_h, sub_w, 3), dtype=np.uint8)
                sub_canvas[img_border:img_border + img_h, img_border:img_border + img_w] = rimg

                top_left_x = border + spacing + col * (sub_w + spacing)
                top_left_y = border + spacing + row * (sub_h + spacing)

                final_image[top_left_y: top_left_y + sub_h, top_left_x: top_left_x + sub_w] = sub_canvas

                idx += 1

        return final_image

    def get_ssh_pic(self) -> np.ndarray:
        """
        Processes the SSH key file to generate a collage image.

        Returns:
            np.ndarray: Final collage image.
        """
        import struct

        with open(self.key_path, "r") as file:
            ssh_str_fingerprint = file.readline().split(":")

        num_pics = len(ssh_str_fingerprint) // self.__class__.num_bytes_per_seed

        str_hex_seeds = [
            (
                "".join(ssh_str_fingerprint[
                        i * self.__class__.num_bytes_per_seed: (i + 1) * self.__class__.num_bytes_per_seed - 1]),
                ssh_str_fingerprint[(i + 1) * self.__class__.num_bytes_per_seed - 1]
            )
            for i in range(num_pics)
        ]

        int_seeds = [
            struct.unpack("I", bytes.fromhex(str_hex_num[0]))[0]
            for str_hex_num in str_hex_seeds
        ]

        guidance_scales = [
            struct.unpack("B", bytes.fromhex(str_hex_num[1]))[0]
            for str_hex_num in str_hex_seeds
        ]

        str_hex_resresidual = "".join(
            ssh_str_fingerprint[num_pics * self.__class__.num_bytes_per_seed:]
        )

        int_resresidual = int(str_hex_resresidual, 16)

        images = [
            self.get_image(
                int_seeds[i],
                guidance_scales[i],
                f"Random picture"
            )
            for i in range(len(int_seeds))
        ]

        texture = self.get_image(
            int_resresidual,
            int_resresidual / self.__class__.guidance_divisor ** (math.ceil(int_resresidual.bit_length() / 8) - 1),
            f"Random texture"
        )

        return self.paint_ssh(images, texture)

    def run(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        key_filename = os.path.splitext(os.path.basename(self.output_path))[0]
        result = self.get_ssh_pic()

        output_path = os.path.join(self.output_path, f"{key_filename}.png")
        cv2.imwrite(output_path, result)

        print(f"Result saved to: {output_path}")
