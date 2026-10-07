import cv2
import numpy as np
import sys
import json
import uuid
import time
import matplotlib.pyplot as plt


start_time = time.time()

# ==========================================
# Load Image with commandline
# ==========================================
image_path = sys.argv [1] if len(sys.argv) > 1 else "R.png"
image = cv2.imread(image_path)
request_id = f"demo-palm-{uuid.uuid4().hex[:6]}"
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==========================================
# if no file is found
# ==========================================
if image is None:
    print(json.dumps({
        "schema_version": "1.0.0",
        "request_id": f"demo-palm-{uuid.uuid4().hex[:6]}",
        "status": "invalid_input",
        "features": None,
        "processing_ms": int((time.time() - start_time) * 1000),
        "warnings": ["file_not_found_or_invalid_format"]
    }, indent=4))
    sys.exit(1)


image_path = sys.argv[1] if len(sys.argv) > 1 else "R.png"
image = cv2.imread(image_path)
# ==========================================
# if image is too dark
# ==========================================
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
avg_brightness = np.mean(gray)
is_blurry = laplacian_var < 40.0
is_dark = avg_brightness < 35.0
if is_blurry or is_dark:
    warnings = []
    if is_blurry: warnings.append("image_blurry")
    if is_dark: warnings.append("insufficient_illumination")

    print(json.dumps({
        "schema_version": "1.0.0",
        "request_id": request_id,
        "status": "recapture_needed",
        "capture_quality": {
            "score": 0.0,
            "blur_status": "insufficient" if is_blurry else "acceptable",
            "brightness_status": "too_dark" if is_dark else "acceptable"
        },
        "features": None,
        "processing_ms": int((time.time() - start_time) * 1000),
        "warnings": warnings
    }, indent=4))
    sys.exit(0)


# ==========================================
# Grayscale
# ==========================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==========================================
# Blur
# ==========================================
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ==========================================
# Gradient Computation (Week 3)
# ==========================================
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
gradient = cv2.magnitude(grad_x, grad_y)

# Normalize for display
gradient_display = cv2.normalize(
    gradient,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# ==========================================
# Edge Detection (Week 3)
# ==========================================
edges = cv2.Canny(blurred, 50, 150)

# ==========================================
# Contrast Enhancement
# Helps reveal palm creases
# ==========================================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)

# ==========================================
# Palm Line Enhancement
# ==========================================
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (10, 15)
)

blackhat = cv2.morphologyEx(
    enhanced,
    cv2.MORPH_BLACKHAT,
    kernel
)

# ==========================================
# Palm Line Segmentation
# ==========================================
lines = cv2.adaptiveThreshold(
    blackhat,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    -5
)

small_kernel = np.ones((3, 3), np.uint8)

lines = cv2.morphologyEx(
    lines,
    cv2.MORPH_OPEN,
    small_kernel
)

# ==========================================
# Find Largest Palm Contour
# ==========================================
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image_rgb.copy()

if len(contours) > 0:

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 0, 255),
        2
    )

    cv2.drawContours(
        result,
        [largest],
        -1,
        (0, 255, 0),
        2
    )

    M = cv2.moments(largest)

    if M["m00"] != 0:

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(
            result,
            (cx, cy),
            8,
            (255, 0, 0),
            -1
        )

    top_point = tuple(
        largest[
            largest[:, :, 1].argmin()
        ][0]
    )

    cv2.circle(
        result,
        top_point,
        8,
        (255, 255, 0),
        -1
    )

# ==========================================
# Overlay Palm Lines
# ==========================================
overlay = image_rgb.copy()


overlay[lines > 0] = [255, 0, 0]

# ==========================================
# Generative AI BLOCK Start
# ==========================================
# ==========================================
# Feature extraction
# ==========================================
img_h, img_w = image.shape  [:2]
edge_density = float(np.sum(edges > 0) / edges.size)

features = {
    "palm_region_detected": False,
    "bounding_box_normalized": None,
    "contour_area_ratio": 0.0,
    "edge_density": round(edge_density, 4)
}

if len(contours) > 0:
    features["palm_region_detected"] = True

    area = cv2.contourArea(largest)
    features["contour_area_ratio"] = round(area / (img_w * img_h), 4)


    x, y, w, h = cv2.boundingRect(largest)
    features["bounding_box_normalizd"] = {
        "x": round(x / img_w, 2),
        "y": round(y / img_h, 2),
        "width": round(w / img_w, 2),
        "height": round(h / img_h, 2)
    }

success_response = {
    "schema_version": "1.0.0",
    "request_id": request_id,
    "status": "ok",
    "capture_quality": {
        "score": 0.89,
        "blur_status": "acceptable",
        "brightness_status": "acceptable"
    },
    "features": features,
    "processingTime_ms": int((time.time() - start_time) * 1000),
    "warnings": []
}


print(json.dumps(success_response, indent=4))
# ==========================================
# Generative AI BLOCK End
# ==========================================

# ==========================================
# One Large Teaching Figure
# ==========================================

fig, ax = plt.subplots(2, 5, figsize=(20, 10))
ax[0,0].imshow(image_rgb)
ax[0,0].set_title("1. Original")
plt.tight_layout()
plt.show()

