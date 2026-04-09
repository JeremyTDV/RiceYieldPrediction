import gzip
import shutil

# Compress the GeoJSON file
with open('static/calabarzon.geojson', 'rb') as f_in:
    with gzip.open('static/calabarzon.geojson.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("GeoJSON file compressed successfully")
print(f"Original size: {191678} bytes")
print(f"Compressed size will be much smaller")
