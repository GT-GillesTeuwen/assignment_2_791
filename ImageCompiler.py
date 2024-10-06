from PIL import Image
import os

# Directories
images_dir = 'images'
histograms_dir = 'histograms'
results_dirs_o = [
    'results/otsu/sa/k{}',
    'results/otsu/sa/k{}/histogram',
    'results/otsu/vns/k{}',
    'results/otsu/vns/k{}/histogram'
]

results_dirs_k = [
    'results/kapur/sa/k{}',
    'results/kapur/sa/k{}/histogram',
    'results/kapur/vns/k{}',
    'results/kapur/vns/k{}/histogram'
]

# Output directory for the compiled images
output_dir = 'compiled_images'
os.makedirs(output_dir, exist_ok=True)

# Function to find the corresponding image file
def find_image(base_name, dir_path):
    for file_name in os.listdir(dir_path):
        if file_name.startswith(base_name):
            return os.path.join(dir_path, file_name)
    return None

# Function to resize an image
def resize_image(image, scale_factor):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Margin size in pixels
margin = 100

def doThing(dirs):
    # Iterate over images in the 'images' directory
    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith(('.jpg', '.png')):
            continue

        # Base name without extension
        base_name = os.path.splitext(image_file)[0]
        dir_name=""
        # Add original and histogram images
        original_image_path = os.path.join(images_dir, image_file)
        original_image = Image.open(original_image_path)
        original_image = resize_image(original_image, 1.5)  # Make the original image larger

        histogram_image_path = find_image(base_name, histograms_dir)
        if histogram_image_path:
            histogram_image = Image.open(histogram_image_path)  # Keep histogram image size as is
        else:
            continue  # Skip this image if no histogram is found

        # Iterate over n in {2, 3, 4, 5}
        for n in range(2, 6):
            images_to_compile = [original_image, histogram_image]

            # Add result images for the current value of n
            for dir_template in dirs:
                dir_parts = dir_template.split('/')
            
                dir_name = dir_parts[1]
                dir_path = dir_template.format(n)
                image_path = find_image(base_name, dir_path)
                if image_path:
                    img = Image.open(image_path)
                    # Resize images not from histogram directories
                    if 'histogram' not in dir_template:
                        img = resize_image(img, 1.5)
                    images_to_compile.append(img)

            # Skip if no additional images were found
            if len(images_to_compile) <= 2:
                continue

            # Calculate dimensions for the 3x2 grid
            cell_width = max(img.width for img in images_to_compile)
            cell_height = max(img.height for img in images_to_compile)
            grid_width = cell_width * 2  # 2 images per row
            grid_height = cell_height * 3  # 3 rows

            # Create a new blank image for the grid with margins
            combined_image = Image.new('RGB', (grid_width + 2 * margin, grid_height + 2 * margin), (255, 255, 255))

            # Paste images into the 3x2 grid, centered in each cell with margins
            for index, img in enumerate(images_to_compile):
                row = index // 2
                col = index % 2
                x_offset = margin + col * cell_width + (cell_width - img.width) // 2  # Center horizontally
                y_offset = margin + row * cell_height + (cell_height - img.height) // 2  # Center vertically
                combined_image.paste(img, (x_offset, y_offset))

            # Save the compiled image
            compiled_image_path = os.path.join(output_dir, f'{base_name}_k{n}_{dir_name}_compiled.jpg')
            combined_image.save(compiled_image_path)
            print(f'Compiled image saved: {compiled_image_path}')

doThing(results_dirs_o)
doThing(results_dirs_k)