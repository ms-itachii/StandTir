from PIL import Image

# Charger l'image
image = Image.open("/home/mahmoud/Ecole/PX/StandTir/image.png")

# Obtenir les dimensions de l'image
largeur, hauteur = image.size

# Définir la taille de la découpe (par exemple, 200x200 pixels)
largeur_decoupe = 400
hauteur_decoupe = 400

# Calculer les coordonnées de la zone centrale
gauche = (largeur - largeur_decoupe) // 2
haut = (hauteur - hauteur_decoupe) // 2
droite = gauche + largeur_decoupe
bas = haut + hauteur_decoupe

# Définir la zone de découpe
zone = (gauche, haut, droite, bas)

# Découper l'image
image_coupée = image.crop(zone)

# Enregistrer ou afficher l'image découpée
image_coupée.save("./StandTir/image_decoupee.png")
image_coupée.show()
