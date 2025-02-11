import os
from PIL import Image

def decouper_images_centrales(dossier_scene, largeur_decoupe=160, hauteur_decoupe=160):
    """
    Découpe une zone centrale de chaque image dans un dossier et enregistre les résultats.
    
    :param dossier_scene: Chemin du dossier contenant les images de la scène.
    :param largeur_decoupe: Largeur de la zone centrale à extraire (par défaut 400).
    :param hauteur_decoupe: Hauteur de la zone centrale à extraire (par défaut 400).
    """
    # Créer un dossier pour enregistrer les images découpées
    dossier_sortie = os.path.join(dossier_scene, "images_decoupees")
    os.makedirs(dossier_sortie, exist_ok=True)

    # Lister les fichiers PNG dans le dossier scène
    fichiers = sorted(
        [f for f in os.listdir(dossier_scene) if f.endswith(".png")], 
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for fichier in fichiers:
        image_path = os.path.join(dossier_scene, fichier)
        try:
            # Charger l'image
            image = Image.open(image_path)

            # Obtenir les dimensions de l'image
            largeur, hauteur = image.size

            # Calculer les coordonnées de la zone centrale
            gauche = (largeur - largeur_decoupe) // 2
            haut = (hauteur - hauteur_decoupe) // 2
            droite = gauche + largeur_decoupe
            bas = haut + hauteur_decoupe
            zone = (gauche, haut, droite, bas)

            # Découper l'image
            image_coupee = image.crop(zone)

            # Enregistrer l'image découpée dans le dossier de sortie
            sortie_path = os.path.join(dossier_sortie, fichier)
            image_coupee.save(sortie_path)
            print(f"✅ Image découpée enregistrée : {sortie_path}")

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {fichier} : {e}")

def main():
    # Chemin du dossier contenant les images de la scène
    dossier_scene = "/home/mahmoud/Ecole/PX/StandTir/Scene"
    
    # Exécuter la fonction
    decouper_images_centrales(dossier_scene)

if __name__ == '__main__':
    main()
