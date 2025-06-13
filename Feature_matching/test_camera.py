from picamera2 import Picamera2, Preview
import time, cv2

def test_awb():
    picam2 = Picamera2()
    
    # Utilisation du pipeline "still" (identique à libcamera-hello)
    config = picam2.create_still_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    
    # Activation automatique de l'exposition et balance des blancs
    picam2.set_controls({
        "AwbEnable": True,
        "AwbMode": 0,       # 0 = auto
        "AeEnable": True
    })
    
    # Aperçu QTGL (optionnel)
    picam2.start_preview(Preview.QTGL)
    picam2.start()
    time.sleep(2)  # Laisse AWB/AE s'ajuster
    
    img = picam2.capture_array()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Test AWB Image", bgr)
    cv2.waitKey(0)
    
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_awb()
