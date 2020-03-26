import cv2 as cv
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

class PolyImg:
  def __init__(self, path, blur, max_points, rate):
    self.blur = blur
    self.max_points = max_points
    self.rate = rate
    self.orig = cv.imread(path)

  def _preprocess(self, img):
    # Convert image to greyscale
    gs_luminosities = [0.07, 0.72, 0.21]
    grey = cv.transform(img, np.array(gs_luminosities).reshape((1, 3)))

    # Blur image
    blur = cv.blur(grey, (self.blur, self.blur))

    # Get the edges
    canny_kernel = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]])
    edges = cv.filter2D(blur, -1, canny_kernel)
    return edges

  def _get_nodes(self, img):
    nodes_convolution = signal.convolve2d(img, np.ones((3, 3))/9)
    sparse_nodes = nodes_convolution > 0.1
    nodes = np.transpose(sparse_nodes.nonzero())
    selected_idx = np.random.randint(len(nodes), size=self.max_points)
    return nodes[selected_idx, :]

  def _create_triangles(self, nodes):
    tri = Delaunay(nodes)
    return tri

  def plot_nodes(self):
    pp_img = self._preprocess(self.orig)
    nodes = self._get_nodes(pp_img)
    triangles = self._create_triangles(nodes)
    # plt.scatter(nodes[0], nodes[1])
    plt.triplot(nodes[:,0], nodes[:,1], triangles.simplices.copy())
    plt.plot(nodes[:,0], nodes[:,1], 'o')
    plt.show()
  
  def get_low_poly(self):
    pp_img = self._preprocess(self.orig)
    # nodes = self._get_nodes(pp_img)
    return pp_img
  
  def show_low_poly(self):
    cv.imshow("Low Poly image", self.get_low_poly())
    cv.waitKey(0)