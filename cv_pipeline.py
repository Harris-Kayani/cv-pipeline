import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# img processing _><><>-
# ==========================================
# load and resize image to 512x512
img_bgr = cv2.imread('use_your_image.jpg')
if img_bgr is None:
    # fallback if img isnt found
    img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img_bgr, (150, 200), 60, (30, 30, 220), -1)
    cv2.circle(img_bgr, (360, 320), 85, (25, 25, 210), -1)
    cv2.rectangle(img_bgr, (80, 380), (180, 460), (40, 200, 40), -1)
else:
    img_bgr = cv2.resize(img_bgr, (512, 512))

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# conversion to grayscale, HSV and LAB
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

# gamma correction (gamma = 1.5 for brightness/contrast adjustment)
gamma = 1.5
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
gamma_corrected = cv2.LUT(img_bgr, table)

# CLAHE on L-channel of LAB image to enhance local contrast
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
clahe_bgr = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

# bilateral filter to smooth while preserving edges
bilateral = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

# ==========================================
# colour based segmentation
# ==========================================
# extract red objects using HSV color space
lower_red1 = np.array([0, 100, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# erosion and dialation to clean binary mask
kernel = np.ones((5, 5), np.uint8)
mask_eroded = cv2.erode(mask, kernel, iterations=1)
mask_clean = cv2.dilate(mask_eroded, kernel, iterations=2)

# detect object boundaries
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img_bgr.copy()
detected_objects = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected_objects.append({'area': area, 'centroid': (cx, cy), 'contour': cnt})

            # draw contour and centroid on original image
            cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(contour_img, (cx, cy), 5, (255, 0, 0), -1)

print(f"Total Detected Objects: {len(detected_objects)}")
for idx, obj in enumerate(detected_objects, 1):
    print(f"Object {idx}: Area = {obj['area']:.2f} px, Centroid = {obj['centroid']}")

# ==========================================
# feature extraction and matching
# ==========================================
# create transformed version by rotating image 45 degrees
center = (256, 256)
rot_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_bgr = cv2.warpAffine(img_bgr, rot_matrix, (512, 512))

gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_rot = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)

# detect keypoints and descriptors using ORB
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(gray_orig, None)
kp2, des2 = orb.detectAndCompute(gray_rot, None)

# feature matching using brute force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# visualize top 50 matches
matched_img = cv2.drawMatches(
    img_bgr, kp1, rotated_bgr, kp2, matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# ==========================================
# line drawing and region highlighting
# ==========================================

# custom implementations
def bresenham_line(img, x0, y0, x1, y1, color=(255, 255, 0)):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0]:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def midpoint_circle(img, xc, yc, r, color=(0, 255, 255)):
    x = 0
    y = r
    p = 1 - r

    def plot_octants(cx, cy, x, y):
        points = [
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x),
            (cx + y, cy - x), (cx - y, cy - x)
        ]
        for px, py in points:
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                img[py, px] = color

    plot_octants(xc, yc, x, y)
    while x < y:
        x += 1
        if p < 0:
            p += 2 * x + 1
        else:
            y -= 1
            p += 2 * (x - y) + 1
        plot_octants(xc, yc, x, y)


algo_img = img_bgr.copy()


if len(detected_objects) >= 2:
    c1 = detected_objects[0]['centroid']
    c2 = detected_objects[1]['centroid']
    bresenham_line(algo_img, c1[0], c1[1], c2[0], c2[1], color=(255, 255, 0))

if len(detected_objects) >= 1:
    largest_obj = max(detected_objects, key=lambda o: o['area'])
    lx, ly = largest_obj['centroid']
    radius = int(np.sqrt(largest_obj['area'] / np.pi)) + 5
    midpoint_circle(algo_img, lx, ly, radius, color=(0, 255, 255))

# display results
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes[0, 0].imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB));
axes[0, 0].set_title("Part 1: Gamma Corrected")
axes[0, 1].imshow(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB));
axes[0, 1].set_title("Part 1: CLAHE Enhanced")
axes[0, 2].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB));
axes[0, 2].set_title("Part 1: Bilateral Filter")
axes[1, 0].imshow(mask_clean, cmap='gray');
axes[1, 0].set_title("Part 2: Cleaned Binary Mask")
axes[1, 1].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB));
axes[1, 1].set_title("Part 2: Contours & Centroids")
axes[1, 2].imshow(cv2.cvtColor(algo_img, cv2.COLOR_BGR2RGB));
axes[1, 2].set_title("Part 4: Bresenham & Midpoint")
for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title("Part 3: Top 50 ORB Feature Matches (Original vs 45° Rotated)")
plt.axis('off')
plt.show()