import cv2
import numpy as np

# -------------- 1️⃣ FUNCTION TO CONVERT BGR TO HSV ----------------
def convert_bgr_to_hsv(bgr_color):
    color_bgr = np.uint8([[bgr_color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    return color_hsv[0][0]

# -------------- 2️⃣ DEFINE YOUR TWO COLORS (in BGR format) ----------------
color1_bgr = (29.4, 94.1, 100)     # Example: dark violet-ish
color2_bgr = (29.4, 94.1, 100)   # Example: white

# Convert them to HSV
hsv1 = convert_bgr_to_hsv(color1_bgr)
hsv2 = convert_bgr_to_hsv(color2_bgr)

# -------------- 3️⃣ FUNCTION TO DETECT COLORS BASED ON 2 RANGES ----------------
def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges with margin
    margin = 8
    hsv_ranges = [
        ([max(hsv1[0] - margin, 0), 50, 50], [min(hsv1[0] + margin, 179), 255, 255]),
        ([max(hsv2[0] - margin, 0), 50, 50], [min(hsv2[0] + margin, 179), 255, 255])
    ]

    # Combine masks
    mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in hsv_ranges:
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask_part = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_or(mask, mask_part)

    # Optional: clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# -------------- 4️⃣ FUNCTION TO GET CONTOUR CENTER ----------------
def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    return None

# -------------- 5️⃣ FUNCTION TO COMPARE CONTOURS ----------------
def compare_contours(captured_contour, reference_contour, threshold=0.1):
    shape_match_score = cv2.matchShapes(captured_contour, reference_contour, 1, 0.0)
    print(f"Shape Match Score: {shape_match_score}")
    return shape_match_score < threshold

# -------------- 6️⃣ LOAD REFERENCE IMAGE & PROCESS ----------------
reference_image = cv2.imread("./pato.png")

if reference_image is None:
    print("Error: Could not load reference image.")
    exit()

reference_masked = detect_colors(reference_image)
cv2.imwrite("reference_masked_binary.png", reference_masked)

contours_reference, _ = cv2.findContours(reference_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours_reference) == 0:
    print("Error: No contours found in reference image.")
    exit()

reference_contour = max(contours_reference, key=cv2.contourArea)
reference_image_copy = reference_image.copy()
cv2.drawContours(reference_image_copy, [reference_contour], -1, (0, 255, 0), 3)

# -------------- 7️⃣ REAL-TIME CAPTURE AND ANALYSIS ----------------
cap = cv2.VideoCapture(0)

print("Press 'C' to capture and compare.")
print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot capture video.")
        break

    cv2.imshow("Original Image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        print("Processing captured image...")

        combined_mask = detect_colors(frame)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            center = get_contour_center(largest_contour)
            if center:
                print(f"Largest Contour Center: {center}")

            frame_with_contour = frame.copy()
            cv2.drawContours(frame_with_contour, [largest_contour], -1, (0, 255, 0), 3)

            # Resize for side-by-side comparison
            height = min(frame_with_contour.shape[0], reference_image_copy.shape[0])
            frame_resized = cv2.resize(frame_with_contour, (int(frame_with_contour.shape[1] * height / frame_with_contour.shape[0]), height))
            reference_resized = cv2.resize(reference_image_copy, (int(reference_image_copy.shape[1] * height / reference_image_copy.shape[0]), height))

            comparison_image = np.hstack((frame_resized, reference_resized))
            cv2.imshow("Captured vs Reference", comparison_image)

            # Compare contours
            match_result = compare_contours(largest_contour, reference_contour, threshold=0.5)
            if match_result:
                print("Contours Match: ✅ YES")
            else:
                print("Contours Do Not Match: ❌ NO")
        else:
            print("No contours detected in captured image.")

        cv2.imshow("Detected Colors Mask", combined_mask)

    if key == ord('q'):
        break

# -------------- 🔚 CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
