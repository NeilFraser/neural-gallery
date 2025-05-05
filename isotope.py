import colorsys
import os
import glob
import random
import shutil
from collections import Counter
from PIL import Image


def rotate_image(img, rnd):
  # 1. Random Rotation
  angle = rnd * 180
  if random.random() > 0.5:
    angle = -angle
  return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)


def translate_image(img, rnd):
  # 2. Random Translation
  MAX_TRANSLATION = 6  # 1/n of the image.
  width, height = img.size
  max_translation_x = width // MAX_TRANSLATION
  max_translation_y = height // MAX_TRANSLATION
  tx = int(rnd * (max_translation_x * 2) - max_translation_x)
  ty = int(rnd * (max_translation_y * 2) - max_translation_y)

  # Create a new image with enough space for the translated image
  translated_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

  # Paste the image onto the new canvas with the translation offset
  translated_img.paste(img, (tx, ty))
  return translated_img


def stretch_image(img, rnd):
  # 3. Random stretch/zoom
  MIN_STRETCH = 0.8
  MAX_STRETCH = 1.2
  width, height = img.size
  horizontal_factor = rnd * (MAX_STRETCH - MIN_STRETCH) + MIN_STRETCH
  vertical_factor = rnd * (MAX_STRETCH - MIN_STRETCH) + MIN_STRETCH
  new_width = int(width * horizontal_factor)
  new_height = int(height * vertical_factor)

  # Resize the image
  img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
  # Crop the image to the original size
  left = (new_width - width) // 2
  top = (new_height - height) // 2
  right = left + width
  bottom = top + height
  return img.crop((left, top, right, bottom))


def hue_image(img, rnd):
  # 4. Random hue rotation
  width, height = img.size
  hue_shift = rnd
  pixels = img.load()

  for y in range(height):
    for x in range(width):
      r, g, b = pixels[x, y]
      h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

      # Rotate the hue
      new_h = (h + hue_shift) % 1.0

      # Convert back to RGB
      new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, s, v)

      # Scale back to 0-255 integer values
      pixels[x, y] = (int(new_r * 255), int(new_g * 255), int(new_b * 255))
  return img


def get_background(img):
  # Find the most likely background colour.
  width, height = img.size
  pixels = img.load()

  points = [
    (0, 0),  # Corners.
    (width - 1, 0),
    (0, height - 1),
    (width - 1, height - 1),
    (0, height // 4),  # Left column.
    (0, height // 4 * 3),
    (width - 1, height // 4),  # Right column.
    (width - 1, height // 4 * 3),
    (width // 4, 0),  # Top row.
    (width // 4 * 3, 0),
    (width // 4, height - 1),  # Bottom row.
    (width // 4 * 3, height - 1),
  ]
  rgb_array = []
  for point in points:
    rgb_array.append(pixels[point[0], point[1]])

  color_counts = Counter(rgb_array)
  max_count = max(color_counts.values())
  modes = [color for color, count in color_counts.items() if count == max_count]
  return modes[0]


def rgb_image(img, bg):
  width, height = img.size
  # Remove the alpha channel if it exists
  if img.mode != "RGBA":
    return img

  bg_img = Image.new("RGB", (width, height), bg)  # Create background
  # Extract the alpha channel as a mask
  alpha_mask = img.split()[-1]

  # Ensure the alpha mask is a single channel (L mode)
  if alpha_mask.mode != 'L':
    alpha_mask = alpha_mask.convert('L')

  # Composite the original RGB channels onto the red background using the alpha mask
  rgb_image = img.convert("RGB")
  return Image.composite(rgb_image, bg_img, alpha_mask)


def get_image_name(image_path):
  base_name = os.path.basename(image_path)
  return os.path.splitext(base_name)[0]


def mutate_image(OUT_DIR, input_image_path, n):
  img_name = get_image_name(input_image_path)
  img = None
  try:
    img = Image.open(input_image_path).convert("RGBA")
  except FileNotFoundError:
    print(f"Error: Input file '{input_image_path}' not found.")
    return
  except Exception as e:
    print(f"An error occurred: {e}")
    return

  bg = get_background(img)

  mutation = 0
  if random.random() > 0.5:
    rnd = random.random()
    img = rotate_image(img, rnd)
    mutation += rnd
  if random.random() > 0.5:
    rnd = random.random()
    img = translate_image(img, rnd)
    mutation += rnd
  if random.random() > 0.5:
    rnd = random.random()
    img = stretch_image(img, rnd)
    mutation += rnd
  img = rgb_image(img, bg)
  if random.random() > 0.5:
    rnd = random.random()
    img = hue_image(img, rnd)
    mutation += rnd

  # Mutation is a number between 0 and 4.
  # Create a score penalty from 0 to 100.
  score = int(mutation * 25)

  dir = os.path.join(OUT_DIR, img_name)
  # Create directory if it doesn't exist
  if not os.path.exists(dir):
    os.makedirs(dir)
  # Save the transformed image
  output_image_path = os.path.join(dir, "%d_%d.png" % (n, score))
  img.save(output_image_path)


if __name__ == "__main__":
  SRC_DIR = "Pool"
  TRAINING_DIR = "Training"
  VALIDATION_DIR = "Validation"
  TEST_DIR = "Test"
  VALIDATION_FRACTION = 0.15
  TEST_FRACTION = 0.15

  # Open each png in the 'Pool' directory.
  pool_files = glob.glob(os.path.join(SRC_DIR, "*.png"))
  pool_files.sort()
  print("Found %d Pool images." % len(pool_files))

  validation_len = int(len(pool_files) * VALIDATION_FRACTION)
  test_len = int(len(pool_files) * TEST_FRACTION)
  validation_files = random.sample(pool_files, validation_len)
  training_files = [item for item in pool_files if item not in validation_files]
  test_files = random.sample(training_files, test_len)
  training_files = [item for item in training_files if item not in test_files]
  print("Split Pool into %d Training images, %d Validation images, and %d Test images." %
        (len(training_files), len(validation_files), len(test_files)))

  groups = [(training_files, TRAINING_DIR), (validation_files, VALIDATION_DIR), (test_files, TEST_DIR)]
  for (files, DIR) in groups:
    files.sort()
    try:
      shutil.rmtree(DIR)
    except:
      pass
    os.makedirs(DIR)
    for png_image in files:
      img_name = get_image_name(png_image)
      shutil.copy2(png_image, os.path.join(DIR, img_name + ".png"))
      for i in range(16):
        mutate_image(DIR, png_image, i)
      print(f"Image '{png_image}' transformed and saved to '{DIR}'")
