import cv2
from picamera2 import Picamera2
import time

def main():
    # Initialisation de la caméra
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)

    # Départ en mode balance des blancs manuelle
    gain_r = 1.5
    gain_b = 1.5

    picam2.set_controls({
        "AwbMode": 0,  # Mode manuel
        "ColourGains": (gain_r, gain_b),
        "AeEnable": True
    })

    picam2.start()
    time.sleep(3)

    print("Calibration de la balance des blancs :")
    print("Utilise les touches pour ajuster :")
    print("  Z / S : augmenter / diminuer le gain Rouge")
    print("  E / D : augmenter / diminuer le gain Bleu")
    print("  Q : quitter et sauvegarder")

    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        info = f"Gain Rouge: {gain_r:.2f} | Gain Bleu: {gain_b:.2f}"
        cv2.putText(frame_bgr, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Calibration AWB", frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        updated = False
        step = 0.05  # ajustement fin

        if key == ord('z'):  # Augmenter R
            gain_r += step
            updated = True
        elif key == ord('s'):  # Diminuer R
            gain_r = max(0.1, gain_r - step)
            updated = True
        elif key == ord('e'):  # Augmenter B
            gain_b += step
            updated = True
        elif key == ord('d'):  # Diminuer B
            gain_b = max(0.1, gain_b - step)
            updated = True
        elif key == ord('q'):  # Quitter
            break

        if updated:
            print(f"Nouvelle configuration : gain_r = {gain_r:.2f}, gain_b = {gain_b:.2f}")
            picam2.set_controls({
                "ColourGains": (gain_r, gain_b)
            })

    cv2.destroyAllWindows()
    picam2.stop()

    print("Calibration terminée.")
    print(f"➡️  Gains finaux à utiliser dans ton code : ColourGains = ({gain_r:.2f}, {gain_b:.2f})")

if __name__ == "__main__":
    main()
