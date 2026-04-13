from PIL import Image
import os


def combine_plots(
    image_paths: list[str], output_dir: str, output_file_name: str, direction="vertical"
) -> None:
    """
    Stitches multiple images together into a single image.

    Args:
        image_paths (list[str]): List of file paths to the images to be combined.
        output_dir (str): Directory where the combined image will be saved.
        output_file_name (str): Name of the output file.
        direction (str, optional): Direction to stack images, either "vertical" or "horizontal".
            Defaults to "vertical".
    """
    if not image_paths:
        return

    try:
        images = [Image.open(p) for p in image_paths if os.path.exists(p)]
        if len(images) < 2:
            return

        # Resize logic
        if direction == "vertical":
            max_width = max(i.width for i in images)
            # Resize all to max width
            images = [
                i.resize((max_width, int(i.height * max_width / i.width)))
                for i in images
            ]

            total_height = sum(i.height for i in images)
            new_img = Image.new("RGB", (max_width, total_height), (255, 255, 255))

            y_offset = 0
            for i in images:
                new_img.paste(i, (0, y_offset))
                y_offset += i.height
        else:
            # Horizontal logic (simple implementation)
            max_height = max(i.height for i in images)
            images = [
                i.resize((int(i.width * max_height / i.height), max_height))
                for i in images
            ]
            total_width = sum(i.width for i in images)
            new_img = Image.new("RGB", (total_width, max_height), (255, 255, 255))
            x_offset = 0
            for i in images:
                new_img.paste(i, (x_offset, 0))
                x_offset += i.width

        os.makedirs(output_dir, exist_ok=True)
        new_img.save(os.path.join(output_dir, output_file_name))
        print(f"Combined image saved: {output_file_name}")

    except Exception as e:
        print(f"Error combining images: {e}")
