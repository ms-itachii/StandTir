import cv2
import numpy as np
import os


def find_zombie_percentage_in_center(
    ref_img_path,
    scene_img_path,
    ratio_thresh=4,  # Seuil du ratio test (plus haut = plus permissif)
    resize_factor=None,  # Facteur de redimensionnement (ex: 0.5 pour réduire à 50%)
    detector_type="SIFT",  # Type de détecteur ("ORB", "SIFT", "AKAZE")
    nfeatures=500,  # Nombre de points d'intérêt maximum (utilisé pour ORB et autres détecteurs)
    log_dir="logs"  # Répertoire où sauvegarder les images des étapes
):
    """
    Calcule le pourcentage d'occupation du "zombie" (défini par l'image de référence)
    dans la zone centrale de l'image capturée (scene_img_path), avec des logs pour chaque étape.
    """

    # Créer un sous-dossier pour cette image dans le dossier de logs
    scene_name = os.path.splitext(os.path.basename(scene_img_path))[0]
    scene_log_dir = os.path.join(log_dir, scene_name)
    os.makedirs(scene_log_dir, exist_ok=True)

    # Charger les images en niveaux de gris
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    scene_img = cv2.imread(scene_img_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or scene_img is None:
        raise FileNotFoundError("Impossible de charger l'une des images.")

    # Redimensionner les images si un facteur est spécifié
    if resize_factor is not None:
        ref_img = cv2.resize(ref_img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
        scene_img = cv2.resize(scene_img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

    # Initialiser le détecteur de points clés
    if detector_type == "ORB":
        detector = cv2.ORB_create(nfeatures=nfeatures)
        norm_type = cv2.NORM_HAMMING  # ORB utilise NORM_HAMMING
    elif detector_type == "SIFT":
        detector = cv2.SIFT_create(nfeatures=nfeatures)
        norm_type = cv2.NORM_L2  # SIFT utilise NORM_L2
    elif detector_type == "AKAZE":
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_L2  # AKAZE utilise aussi NORM_L2
    else:
        raise ValueError(f"Type de détecteur '{detector_type}' non reconnu.")

    # Détecter les points clés et calculer les descripteurs
    kp1, des1 = detector.detectAndCompute(ref_img, None)
    kp2, des2 = detector.detectAndCompute(scene_img, None)

    # Sauvegarder les points clés détectés
    img_kp1 = cv2.drawKeypoints(ref_img, kp1, None, color=(0, 255, 0))
    img_kp2 = cv2.drawKeypoints(scene_img, kp2, None, color=(0, 255, 0))
    cv2.imwrite(os.path.join(scene_log_dir, "ref_keypoints.jpg"), img_kp1)
    cv2.imwrite(os.path.join(scene_log_dir, "scene_keypoints.jpg"), img_kp2)

    # Vérifier que les descripteurs ne sont pas None ou vides
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0

    # Convertir en float32 si nécessaire
    if norm_type == cv2.NORM_L2:  # SIFT et AKAZE nécessitent float32
        des1 = np.asarray(des1, dtype=np.float32)
        des2 = np.asarray(des2, dtype=np.float32)

    # Initialiser le matcher avec le bon type
    bf = cv2.BFMatcher(norm_type, crossCheck=False)

    # Faire le matching des descripteurs
    matches_knn = bf.knnMatch(des1, des2, k=2)

    # Filtrer les correspondances avec le ratio test
    good_matches = [m for m, n in matches_knn if m.distance < ratio_thresh * n.distance]

    # Sauvegarder les correspondances avant filtrage
    matches_img = cv2.drawMatches(ref_img, kp1, scene_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(scene_log_dir, "matches.jpg"), matches_img)

    if len(good_matches) < 4:
        return 0.0

    # Points pour l'homographie
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calcul de l'homographie
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return 0.0

    # Définir les coins de la bounding box de l'image de référence
    h_ref, w_ref = ref_img.shape
    ref_corners = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)

    # Projeter les coins dans l'image de la scène
    projected_corners = cv2.perspectiveTransform(ref_corners, M)
    scene_with_box = scene_img.copy()
    cv2.polylines(scene_with_box, [np.int32(projected_corners)], True, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(scene_log_dir, "scene_with_box.jpg"), scene_with_box)

    zombie_x, zombie_y, zombie_w, zombie_h = cv2.boundingRect(projected_corners)

    # Zone centrale de l'image de la scène
    x1, y1, x2, y2 = 0, 0, 160, 160

    scene_with_center = scene_img.copy()
    cv2.rectangle(scene_with_center, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(scene_log_dir, "scene_with_center.jpg"), scene_with_center)

    # Calcul de l'intersection
    inter_x1 = max(x1, zombie_x)
    inter_y1 = max(y1, zombie_y)
    inter_x2 = min(x2, zombie_x + zombie_w)
    inter_y2 = min(y2, zombie_y + zombie_h)

    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    scene_with_intersection = scene_img.copy()
    if intersection_area > 0:
        cv2.rectangle(scene_with_intersection, (inter_x1, inter_y1), (inter_x2, inter_y2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(scene_log_dir, "scene_with_intersection.jpg"), scene_with_intersection)

    center_area = (x2 - x1) * (y2 - y1)
    if center_area == 0:
        return 0.0

    return (intersection_area / center_area) * 100.0

def main():
    ref_path = "/home/mahmoud/Ecole/PX/Z-Shooter/assets/ennemis/aigle/fly1.png"
    dossier_scene = "/home/mahmoud/Ecole/PX/StandTir/Scene_aigle/images_decoupees" #

    fichiers = sorted(
        [f for f in os.listdir(dossier_scene) if f.endswith(".png")], 
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for fichier in fichiers:
        scene_path = os.path.join(dossier_scene, fichier)
        pourcentage = find_zombie_percentage_in_center(ref_path, scene_path, log_dir="logs")
        print(f"Pourcentage de correspondance pour {fichier} : {pourcentage:.2f} %")

if __name__ == '__main__':
    main()
