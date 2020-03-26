import sys
from poly_img import PolyImg

img = PolyImg(sys.argv[1], blur=4, max_points=200, rate=0.8)
# img.show_low_poly()
img.plot_nodes()