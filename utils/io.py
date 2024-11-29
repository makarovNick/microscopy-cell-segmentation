import SimpleITK as sitk
from tifffile import imread
import numpy as np


def read_image(path: str) -> sitk.Image:
    """Read a multi-dimensional TIFF file and convert it to a SimpleITK image."""
    numpy_data = imread(path)
    numpy_data = numpy_data.transpose(0, 2, 3, 1).astype(np.uint16)  # Adjust axis order and type
    return sitk.GetImageFromArray(numpy_data)


def write_image(image: sitk.Image, path: str, compression: bool = True) -> None:
    """Write a SimpleITK image to a file."""
    sitk.WriteImage(image, path, useCompression=compression)
