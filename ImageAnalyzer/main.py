import cv2 as cv
import numpy as np

# Ouverture du webcam
camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'c' to capture an image or 'q' to quit.")

while True:
    # Capture frame par frame du flux d'images 
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Afficher l'image vue par la webcam en direct
    cv.imshow("Live Feed", frame)

    # Attendre une action utilisateur
    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        # Transformer l'image en espace couleur HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Determiner le spectre de couleurs utilisés pour créer des masques
        lower_red1 = np.array([0, 150, 100]) 
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        # Création des masques pour la couleur rouge
        mask1 = cv.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        # Application du masque à l'image
        red_result = cv.bitwise_and(frame, frame, mask=red_mask)

        # Trouver des contours 
        contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv.boundingRect(largest_contour)
            print(f"Largest red area coordinates: Top-left ({x}, {y}), Width: {w}, Height: {h}")

            if (w * h) < 300:  # Adjusted the threshold for significant area
                print("No significant red area detected.")
                continue

            # Draw a rectangle around the largest red area on the original image
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the image with the rectangle
            detected_image_filename = "detected_red_area.jpg"
            cv.imwrite(detected_image_filename, frame)
            print(f"Image with detected red area saved as {detected_image_filename}")
        else:
            print("No significant red area detected.")

        # Show the masked image and the result
        cv.imshow("Red Mask", red_result)
        cv.imshow("Detected Red Area", frame)

    elif key == ord('q'):
        # Exit the program when 'q' is pressed
        print("Quitting...")
        break

# Release the camera and close all OpenCV windows
camera.release()
cv.destroyAllWindows()
