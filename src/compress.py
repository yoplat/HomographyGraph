import os
import cv2 as cv

def compress_dataset(filepaths, output_dir, scale=0.25, jpeg_quality=70):
    """
    Compress images from a list of filepaths and save to output_dir.

    Parameters:
        filepaths (list): list of image paths
        output_dir (str): folder to save compressed images
        scale (float): resize factor (e.g. 0.25 = 25% size)
        jpeg_quality (int): JPEG quality (0–100)
    """

    os.makedirs(output_dir, exist_ok=True)

    for path in filepaths:
        try:
            # Read as grayscale (change if needed)
            img = cv.imread(path, cv.IMREAD_COLOR)

            if img is None:
                print(f"Skipping (failed to load): {path}")
                continue

            # Resize
            img_small = cv.resize(img, (0, 0), fx=scale, fy=scale)

            # Build output path (force .jpg for compression)
            filename = os.path.splitext(os.path.basename(path))[0] + ".jpg"
            out_path = os.path.join(output_dir, filename)

            # Save compressed
            cv.imwrite(out_path, img_small, [cv.IMWRITE_JPEG_QUALITY, jpeg_quality])

            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    dataset = [
        f"iacv/IMG_{i}.png"
        for i in range(4714, 4714 + 30)
    ]

    compress_dataset(dataset, "compressed_images", scale=0.25, jpeg_quality=100)
