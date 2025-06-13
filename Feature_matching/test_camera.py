import cv2
import numpy as np
import time
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera

def convert_bgr_to_hsv(bgr_color):
    color_bgr = np.uint8([[bgr_color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    return color_hsv[0][0]

def detect_colors(frame, hsv1, hsv2):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    margin = 8
    hsv_ranges = [
        ([max(hsv1[0] - margin, 0), 50, 50], [min(hsv1[0] + margin, 179), 255, 255]),
        ([max(hsv2[0] - margin, 0), 50, 50], [min(hsv2[0] + margin, 179), 255, 255])
    ]
    mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in hsv_ranges:
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask_part = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.bitwise_or(mask, mask_part)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    return None

def compare_contours(captured_contour, reference_contour, threshold=0.1):
    shape_match_score = cv2.matchShapes(captured_contour, reference_contour, 1, 0.0)
    print(f"Shape Match Score: {shape_match_score}")
    return shape_match_score < threshold

def main():
    color1_bgr = (29.4, 94.1, 100)
    color2_bgr = (29.4, 94.1, 100)
    hsv1 = convert_bgr_to_hsv(color1_bgr)
    hsv2 = convert_bgr_to_hsv(color2_bgr)

    reference_image = cv2.imread("./pato.png")
    if reference_image is None:
        print("Error: Could not load reference image.")
        sys.exit(1)

    reference_masked = detect_colors(reference_image, hsv1, hsv2)
    cv2.imwrite("reference_masked_binary.png", reference_masked)

    contours_reference, _ = cv2.findContours(reference_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_reference) == 0:
        print("Error: No contours found in reference image.")
        sys.exit(1)

    reference_contour = max(contours_reference, key=cv2.contourArea)
    reference_image_copy = reference_image.copy()
    cv2.drawContours(reference_image_copy, [reference_contour], -1, (0, 255, 0), 3)

    # ----------- Initialisation caméra avec picamera ------------
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    camera.awb_mode = 'auto'  # balance des blancs auto
    camera.exposure_mode = 'auto'
    raw_capture = PiRGBArray(camera, size=(640, 480))
    time.sleep(2)  # attendre que l'exposition se stabilise
    # ------------------------------------------------------------

    print("Press 'C' to capture and compare.")
    print("Press 'Q' to quit.")

    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Original Image", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Processing captured image...")
            combined_mask = detect_colors(image, hsv1, hsv2)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                center = get_contour_center(largest_contour)
                if center:
                    print(f"Largest Contour Center: {center}")
                image_with_contour = image.copy()
                cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 3)
                height = min(image_with_contour.shape[0], reference_image_copy.shape[0])
                image_resized = cv2.resize(image_with_contour, (int(image_with_contour.shape[1] * height / image_with_contour.shape[0]), height))
                reference_resized = cv2.resize(reference_image_copy, (int(reference_image_copy.shape[1] * height / reference_image_copy.shape[0]), height))
                comparison_image = np.hstack((image_resized, reference_resized))
                cv2.imshow("Captured vs Reference", comparison_image)
                match_result = compare_contours(largest_contour, reference_contour, threshold=0.5)
                if match_result:
                    print("Contours Match: ✅ YES")
                else:
                    print("Contours Do Not Match: ❌ NO")
            else:
                print("No contours detected in captured image.")
            cv2.imshow("Detected Colors Mask", combined_mask)

        raw_capture.truncate(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.close()

if __name__ == "__main__":
    main()
