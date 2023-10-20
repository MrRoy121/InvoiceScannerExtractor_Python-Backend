import cv2

import numpy as np
import urllib.request

url = str("https://firebasestorage.googleapis.com/v0/b/invoice-scanner-68292.appspot.com/o/images%2Fin.jpg?alt=media&token=d94ed9ac-a4d3-4597-9705-a6c5fe03ed6e")
url_response = urllib.request.urlopen(url)
imgs = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)

new_size = (750, 1061)
imgs = cv2.resize(imgs, new_size)
print(imgs)