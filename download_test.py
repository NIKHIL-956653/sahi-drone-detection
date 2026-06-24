"""Download a sample drone/aerial image with many small objects."""
import urllib.request
import os

# SAHI's official demo image — small cars/people from aerial view
URL = "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg"

OUT = "test.jpg"

if os.path.exists(OUT):
    print(f"✅ {OUT} already exists!")
else:
    print(f"Downloading test image...")
    try:
        urllib.request.urlretrieve(URL, OUT)
        print(f"✅ Saved as {OUT}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Manually download any aerial/drone image and save as test.jpg")