from PIL import Image
from collections import Counter

def get_dominant_colors(image_path, num_colors=2):
    image = Image.open(image_path)
    # Resize for faster processing
    image = image.resize((150, 150))
    # Convert to RGB (in case of PNG with alpha)
    image = image.convert("RGB")
    
    pixels = list(image.getdata())
    counts = Counter(pixels)
    dominant = counts.most_common(num_colors)
    
    hex_colors = []
    for rgb, count in dominant:
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        hex_colors.append(hex_color)
    
    return hex_colors

if __name__ == "__main__":
    image_path = "/Users/muhammadismail/.gemini/antigravity/brain/1baa72f8-3b51-4f6e-93c6-6aa32ed77788/uploaded_media_1770135279156.png"
    try:
        colors = get_dominant_colors(image_path, num_colors=5)
        print("Dominant Colors:", colors)
    except Exception as e:
        print(f"Error: {e}")
