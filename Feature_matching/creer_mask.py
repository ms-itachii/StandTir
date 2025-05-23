import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_main_colors(image_path, k=3):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans clustering to find k dominant colors
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    return colors

def generate_mask_code(colors):
    mask_code = []
    for i, color in enumerate(colors):
        lower = np.clip(color - 20, 0, 255)  # Adjust range as needed
        upper = np.clip(color + 20, 0, 255)

        mask_code.append(f"""
# Mask for Color {i+1} - RGB {tuple(color)}
lower_bound = np.array({list(lower)}, dtype=np.uint8)
upper_bound = np.array({list(upper)}, dtype=np.uint8)
mask{i+1} = cv2.inRange(image, lower_bound, upper_bound)
        """)
    
    return "\n".join(mask_code)

if __name__ == "__main__":
    image_path = "./pato.png"  # Change this to your file path
    colors = extract_main_colors(image_path)
    
    print("### Generated Python Code for Color Masks ###\n")
    print("import cv2\nimport numpy as np\n")
    print("image = cv2.imread('your_image.png')\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n")
    print(generate_mask_code(colors))
