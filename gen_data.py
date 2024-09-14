import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
from matplotlib.font_manager import findSystemFonts

# Characters to include in the dataset
characters = '0123456789'
# Mapping of characters to class indices
char_to_idx = {char: idx for idx, char in enumerate(characters)}

def list_system_fonts():
    """List all available system fonts."""
    fonts = findSystemFonts(fontpaths=None, fontext='ttf')
    return fonts

def select_font(font_name):
    """Select a specific font by name."""
    fonts = list_system_fonts()
    for font in fonts:
        if font_name.lower() in os.path.basename(font).lower():
            return font
    raise ValueError(f"Font '{font_name}' not found on the system.")

def create_image(image_path, label_path, num_chars=5, img_size=(256, 64), font_name="arial", font_size=32):
    # Create a white image
    image = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(image)

    # Select the font
    font_path = select_font(font_name)
    font = ImageFont.truetype(font_path, font_size)

    # Generate random text
    text = ''.join(random.choice(characters) for _ in range(num_chars))

    # Calculate the bounding box of the entire text to center it
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # Position the text in the center
    text_x = (image.width - text_width) // 2
    text_y = (image.height - text_height) // 2

    labels = []
    x_offset = text_x

    for char in text:
        # Calculate the bounding box for each character
        char_bbox = draw.textbbox((x_offset, text_y), char, font=font)
        char_width = char_bbox[2] - char_bbox[0]
        char_height = char_bbox[3] - char_bbox[1]

        # Draw the character
        draw.text((x_offset, text_y), char, font=font, fill='black')

        # Calculate normalized bounding box coordinates
        x_center = (char_bbox[0] + char_bbox[2]) / (2 * img_size[0])
        y_center = (char_bbox[1] + char_bbox[3]) / (2 * img_size[1])
        width = (char_bbox[2] - char_bbox[0]) / img_size[0]
        height = (char_bbox[3] - char_bbox[1]) / img_size[1]

        # Create label
        label_line = f"{char_to_idx[char]} {x_center} {y_center} {width} {height}"
        labels.append(label_line)

        # Move to the next character
        x_offset += char_width

    # Save the image
    image.save(image_path)

    # Write labels to file
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

# Generate training data
os.makedirs("data/characters/images/train", exist_ok=True)
os.makedirs("data/characters/labels/train", exist_ok=True)
for i in tqdm(range(10000)):
    image_path = f"data/characters/images/train/image_{i}.jpg"
    label_path = f"data/characters/labels/train/image_{i}.txt"
    create_image(image_path, label_path)

# Generate validation data
os.makedirs("data/characters/images/val", exist_ok=True)
os.makedirs("data/characters/labels/val", exist_ok=True)
for i in tqdm(range(2000)):
    image_path = f"data/characters/images/val/image_{i}.jpg"
    label_path = f"data/characters/labels/val/image_{i}.txt"
    create_image(image_path, label_path)

print("Dataset creation completed.")
