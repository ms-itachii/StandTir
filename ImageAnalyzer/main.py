# Programme du traitement d'image avec OpenCV
import cv2 as cv

# Open the webcam (0 is the default camera)
camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'c' to capture an image or 'q' to quit.")
c
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show the live video feed
    cv.imshow("Live Feed", frame)

    # Wait for user input
    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save the image when 'c' is pressed
        image_filename = "captured_image.jpg"
        cv.imwrite(image_filename, frame)
        print(f"Image captured and saved as {image_filename}")
    elif key == ord('q'):
        # Exit the program when 'q' is pressed
        print("Quitting...")
        break

# Release the camera and close all OpenCV windows
camera.release()
cv.destroyAllWindows()
