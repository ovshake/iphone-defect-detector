import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from skimage.feature import hog
import os
import glob
from tqdm import tqdm

def compare_patches(patch1, patch2):
    # Convert patches to grayscale
    patch1_gray = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    patch2_gray = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

    hog1 = hog(patch1_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    hog2 = hog(patch2_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    similarity = cv2.compareHist(hog1.astype(np.float32), hog2.astype(np.float32), cv2.HISTCMP_CORREL)
    return similarity

# Read the reference image (perfect iPhone)
reference_img = cv2.imread('assets/perfect_iphone.jpeg')
# Read the query image (iPhone with defect)
query_img = cv2.imread('assets/69.jpg')
def find_similarity_patches(reference_img, query_img):
    # Convert both images to grayscale
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors
    keypoints_reference, descriptors_reference = sift.detectAndCompute(reference_gray, None)
    keypoints_query, descriptors_query = sift.detectAndCompute(query_gray, None)

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using FLANN
    matches = flann.knnMatch(descriptors_reference, descriptors_query, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    reference_points = np.float32([keypoints_reference[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    query_points = np.float32([keypoints_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    M, mask = cv2.findHomography(query_points, reference_points, cv2.RANSAC, 5.0)

    # Align the query image with the reference image
    aligned_query_img = cv2.warpPerspective(query_img, M, (reference_img.shape[1], reference_img.shape[0]))

    # Remove 15% of the border on all 4 sides
    border_ratio = 0.15
    border_width = int(reference_img.shape[1] * border_ratio)
    border_height = int(reference_img.shape[0] * border_ratio)
    reference_img_cropped = reference_img[border_height:-border_height, border_width:-border_width]
    aligned_query_img_cropped = aligned_query_img[border_height:-border_height, border_width:-border_width]

    # Convert the cropped images to grayscale
    reference_gray_cropped = cv2.cvtColor(reference_img_cropped, cv2.COLOR_BGR2GRAY)
    aligned_query_gray_cropped = cv2.cvtColor(aligned_query_img_cropped, cv2.COLOR_BGR2GRAY)

    # Set patch size
    patch_size = (100, 100)

    # Perform patch-wise similarity comparison
    similarity_threshold = 0.17
    dissimilar_patches = []
    similar_patches = []
    for y in range(0, reference_img_cropped.shape[0], patch_size[1]):
        for x in range(0, reference_img_cropped.shape[1], patch_size[0]):
            patch_reference = reference_img_cropped[y:y+patch_size[1], x:x+patch_size[0]]
            patch_query = aligned_query_img_cropped[y:y+patch_size[1], x:x+patch_size[0]]
            similarity = compare_patches(patch_reference, patch_query)
            if similarity < similarity_threshold:
                dissimilar_patches.append((x + border_width, y + border_height, patch_size[0], patch_size[1], similarity))
            else:
                similar_patches.append((x + border_width, y + border_height, patch_size[0], patch_size[1], similarity))
    return dissimilar_patches, similar_patches, aligned_query_img

images_set_path = 'images_set/*'
output_folder = 'output_images'
patch_threshold = 30

if os.path.exists(output_folder):
    for file in glob.glob(os.path.join(output_folder, '*')):
        os.remove(file)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


num_dissimilar_patches_array = []
for query_img_path in tqdm(glob.glob(images_set_path), desc="Processing images"):
    query_img = cv2.imread(query_img_path)
    # Assuming all the necessary steps to get `aligned_query_img`, `dissimilar_patches`, and `similar_patches` are performed here
    dissimilar_patches, similar_patches, aligned_query_img = find_similarity_patches(reference_img, query_img)
    num_dissimilar_patches_array.append(len(dissimilar_patches))
    if len(dissimilar_patches) >= patch_threshold:
        for x, y, w, h, similarity in dissimilar_patches:
            cv2.rectangle(aligned_query_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(aligned_query_img, f"{similarity:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for x, y, w, h, similarity in similar_patches:
            cv2.rectangle(aligned_query_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(aligned_query_img, f"{similarity:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(output_folder, os.path.basename(query_img_path))
        cv2.imwrite(output_path, aligned_query_img)


plt.hist(num_dissimilar_patches_array, bins=20)
plt.xlabel('Number of dissimilar patches')
plt.ylabel('Frequency')
plt.savefig(f'dissimilar_patches_histogram017.png')

