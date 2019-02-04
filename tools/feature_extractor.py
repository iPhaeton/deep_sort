from generate_detections import ImageEncoder
import os
import argparse
import cv2
import numpy as np

def get_split_path(path):
    return list(filter(lambda x: x != '', path.split('/')))

def extract_features_from_dir(encoder, shape, source, target, filter_dirs=['.DS_Store'], recompute = False):
    """
    Extracts features from the images in single directory or calls itself recursively for nested directories

    Parameters
    ----------
    encoder: Callable[image] -> ndarray
        The encoder function takes as input a BGR color image 
        and returns a matrix of corresponding feature vectors.
    source : string
        Path to a source directory
    target : string
        Path to a target directory
    filter_dirs: [string]
        Directories to be filtered out
    recompute: boolean
        Whether a targer directory should be recomputed if it already exists

    Returns
    -------
    None
    """

    dir_name = get_split_path(source)[-1]
    if dir_name in filter_dirs:
        return

    target_path_step = ''
    for step in get_split_path(target):
        target_path_step += step + '/'
        if os.path.exists(target_path_step) == False:
            os.mkdir(target_path_step)

    target_path = os.path.join(target, dir_name)
    
    if os.path.exists(target_path) == False:
        os.mkdir(target_path)
    elif recompute == False:
        print(target_path + 'already exists')
        return
    
    for name in os.listdir(source):
        source_path = os.path.join(source, name)
        if os.path.isfile(source_path):
            features_path = os.path.join(target_path, name.split('.')[0])
            if os.path.exists(features_path) == True:
                continue

            bgr_image = cv2.imread(source_path, cv2.IMREAD_COLOR)
            bgr_image = cv2.resize(bgr_image, tuple(reversed(shape)))
            features = encoder([bgr_image], batch_size=1)
            np.save(features_path, features)
        else:
            extract_features_from_dir(encoder, shape, source_path, target_path)
    print(target_path + ' extracted')

def extract_features(encoder, shape, source, target):
    """
    Extracts features from the images. Target directory preserves the structure of the source directory
    
    Parameters
    ----------
    encoder: Callable[image] -> ndarray
        The encoder function takes as input a BGR color image 
        and returns a matrix of corresponding feature vectors.
    source : string
        Path to a source directory
    target : string
        Path to a target directory

    Returns
    -------
    None
    """

    if os.path.exists(source) == False:
        raise Exception('[extract_features] Source directory "' + source + '" does not exist')
    
    extract_features_from_dir(encoder, shape, source, target, recompute=True)

def create_box_encoder(model_filename):
    encoder = ImageEncoder(model_filename)
    return encoder

def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--shape",
        default=(128, 64),
        help="Path to a target directory.")
    parser.add_argument(
        "--source",
        help="Path to a source directory.")
    parser.add_argument(
        "--target",
        help="Path to a target directory.")

    return parser.parse_args()

def main():
    args = parse_args()
    encoder = create_box_encoder(args.model)
    extract_features(encoder, args.shape, args.source, args.target)

if __name__ == "__main__":
    main()
