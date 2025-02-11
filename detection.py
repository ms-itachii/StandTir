import cv2
import numpy as np

def find_zombie_percentage_in_center(ref_img_path, scene_img_path):
    """
    Calcule le pourcentage d'occupation du "zombie" (défini par l'image de référence)
    dans la zone centrale de l'image capturée (scene_img_path).

    :param ref_img_path: Chemin vers l'image de référence (ex : 'zombie_reference.png')
    :param scene_img_path: Chemin vers l'image capturée (ex : 'camera_shot.jpg')
    :return: Pourcentage (float) d'occupation du zombie dans la zone centrale.
    """

    # 1) Chargement en niveaux de gris (plus simple pour ORB)
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    scene_img = cv2.imread(scene_img_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or scene_img is None:
        raise FileNotFoundError("Impossible de charger l'une des images.")

    # 2) Création de l'objet ORB (500 features max, à ajuster selon vos besoins)
    orb = cv2.ORB_create(nfeatures=500)

    # 3) Détection et description des keypoints
    kp1, des1 = orb.detectAndCompute(ref_img, None)
    kp2, des2 = orb.detectAndCompute(scene_img, None)

    if des1 is None or des2 is None:
        print("Pas de descripteurs détectés dans l'une des images.")
        return 0.0

    # 4) Matching avec un BFMatcher pour descripteurs de type ORB (Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # On utilise knnMatch pour réaliser le ratio test (Lowe)
    matches_knn = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_thresh = 0.75  # ratio test threshold (classique ~ 0.7-0.8)
    for m,n in matches_knn:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 5) On a besoin d'au moins 4 bons matches pour estimer une homographie
    if len(good_matches) < 4:
        print("Pas assez de bons matches pour estimer une homographie.")
        return 0.0

    # 6) Construction des ensembles de points correspondants
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    # 7) Calcul de l'homographie via RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("Impossible de calculer l'homographie.")
        return 0.0

    # 8) Définition des coins de l'image de référence (zombie)
    h_ref, w_ref = ref_img.shape
    ref_corners = np.float32([[0,0], [w_ref,0], [w_ref,h_ref], [0,h_ref]]).reshape(-1,1,2)

    # 9) Projection des coins dans l'image courante
    projected_corners = cv2.perspectiveTransform(ref_corners, M)  # shape: (4,1,2)

    # 10) Calcul de la bounding box du zombie projeté (approximatif mais simple)
    zombie_x, zombie_y, zombie_w, zombie_h = cv2.boundingRect(projected_corners)

    # 11) Définir la zone centrale de la scene (ex: 50% de largeur/hauteur au centre)
    h_scene, w_scene = scene_img.shape
    # Par exemple, on prend 25% de marge à gauche et à droite => zone centrale = 50% du milieu
    margin_x = 0.25
    margin_y = 0.25
    x1 = int(w_scene * margin_x)
    y1 = int(h_scene * margin_y)
    x2 = int(w_scene * (1.0 - margin_x))
    y2 = int(h_scene * (1.0 - margin_y))

    # 12) Bounding box de la zone centrale
    center_w = x2 - x1
    center_h = y2 - y1

    # 13) Calcul de l'intersection entre les deux rectangles
    # Intersection rect (zombie) vs rect (centre)
    inter_x1 = max(x1, zombie_x)
    inter_y1 = max(y1, zombie_y)
    inter_x2 = min(x2, zombie_x + zombie_w)
    inter_y2 = min(y2, zombie_y + zombie_h)

    intersection_area = 0
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # 14) Aire de la zone centrale
    center_area = center_w * center_h
    if center_area == 0:
        print("Zone centrale invalide (aire = 0)")
        return 0.0

    # 15) Pourcentage d'occupation
    percentage = (intersection_area / center_area) * 100.0

    return percentage

def main():
    # Chemins d'exemple (à adapter)
    ref_path = "/home/mahmoud/Ecole/PX/Z-Shooter/assets/ennemis/aigle/fly1.png"
    scene_path = "/home/mahmoud/Ecole/PX/Z-Shooter/assets/ennemis/aigle/fly2.png"

    pourcentage = find_zombie_percentage_in_center(ref_path, scene_path)
    print(f"Pourcentage d'occupation du zombie dans la zone centrale : {pourcentage:.2f} %")

if __name__ == '__main__':
    main()
