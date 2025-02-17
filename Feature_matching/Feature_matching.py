import cv2
import numpy as np

# Fonction pour détecter les couleurs jaune et orange
def detect_colors(frame):
    # Convertir l'image en espace de couleur HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Plage de couleurs jaunes (en HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Plage de couleurs orange (en HSV)
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([20, 255, 255])

    # Masques pour détecter le jaune et l'orange
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Combiner les deux masques
    combined_mask = cv2.bitwise_or(mask_yellow, mask_orange)

    return combined_mask

# Fonction pour calculer le centre du contour
def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    else:
        return None

# Fonction pour comparer les contours et retourner un résultat booléen
def compare_contours(captured_contour, reference_contour, threshold=0.1):
    # Effectuer le shape matching entre les deux contours
    shape_match_score = cv2.matchShapes(captured_contour, reference_contour, 1, 0.0)
    print(f"Score de correspondance de forme : {shape_match_score}")
    
    # Si le score de correspondance est inférieur au seuil, considérer que les contours coïncident
    if shape_match_score < threshold:
        return True
    else:
        return False

# Initialiser la capture vidéo de la webcam
cap = cv2.VideoCapture(0)

# Charger l'image de référence pour appliquer le masque
reference_image = cv2.imread("pato.png")

# Appliquer le masque jaune et orange à l'image de référence
reference_masked = detect_colors(reference_image)

# Sauvegarder l'image binaire résultante de la référence
cv2.imwrite("reference_masked_binary.png", reference_masked)

# Trouver les contours dans le masque de référence
contours_reference, _ = cv2.findContours(reference_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
reference_contour = max(contours_reference, key=cv2.contourArea)  # Contour de référence le plus grand

# Dessiner le contour de la référence sur l'image de référence
reference_image_copy = reference_image.copy()
cv2.drawContours(reference_image_copy, [reference_contour], -1, (0, 255, 0), 3)  # Vert, épaisseur 3

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur de capture vidéo")
        break

    # Afficher l'image en temps réel
    cv2.imshow("Image originale", frame)

    # Attendre que l'utilisateur appuie sur 'C' pour capturer l'image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Appuyer sur 'C' pour capturer l'image
        # Effectuer l'analyse sur l'image capturée
        print("Analyse de l'image capturée...")

        # Appliquer le masque jaune et orange à l'image capturée
        combined_mask = detect_colors(frame)

        # Trouver les contours dans le masque capturé
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Si des contours sont trouvés
        if contours:
            # Trouver le plus grand contour basé sur la superficie
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculer le centre du plus grand contour
            center = get_contour_center(largest_contour)
            if center:
                print(f"Position du plus grand contour : {center}")

            # Dessiner le contour sur l'image capturée
            frame_with_contour = frame.copy()
            cv2.drawContours(frame_with_contour, [largest_contour], -1, (0, 255, 0), 3)  # Vert, épaisseur 3

            # Redimensionner l'image de la webcam et l'image de référence pour qu'elles aient la même hauteur
            height = min(frame_with_contour.shape[0], reference_image_copy.shape[0])
            frame_resized = cv2.resize(frame_with_contour, (int(frame_with_contour.shape[1] * height / frame_with_contour.shape[0]), height))
            reference_resized = cv2.resize(reference_image_copy, (int(reference_image_copy.shape[1] * height / reference_image_copy.shape[0]), height))

            # Comparaison entre l'image capturée et l'image de référence
            comparison_image = np.hstack((frame_resized, reference_resized))  # Superposer les deux images

            # Afficher la comparaison entre l'image capturée et la référence
            cv2.imshow("Comparaison entre l'image capturée et la référence", comparison_image)

            # Comparer les contours
            match_result = compare_contours(largest_contour, reference_contour, threshold=0.5)
            if match_result:
                print("Les contours coïncident : True")
            else:
                print("Les contours ne coïncident pas : False")

        # Afficher le masque combiné pour l'image capturée
        cv2.imshow("Masque combiné Jaune + Orange", combined_mask)

    # Quitter si la touche 'q' est pressée
    if key == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
