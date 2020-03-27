import cv2 as cv
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
import math

class PolyImg:
  def __init__(self, path, blur, max_points, rate):
    self.blur = blur
    self.max_points = max_points
    self.rate = rate
    self.orig = cv.imread(path)
    self.low_poly_img = np.ones((self.orig.shape[0], self.orig.shape[1], 3))*255

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
    # Find edges, and threshold them
    nodes_convolution = signal.convolve2d(img, np.ones((3, 3))/9)
    sparse_nodes = nodes_convolution > math.ceil(np.mean(nodes_convolution))
    nodes = np.transpose(sparse_nodes.nonzero())
    
    # Randomly sample edge pixles to get nodes
    if self.max_points > len(nodes):
      selected_idx = np.random.choice(len(nodes), 
                                      size=len(nodes), 
                                      replace=False)
    else:
      selected_idx = np.random.choice(len(nodes), 
                                      size=self.max_points,
                                      replace=False)
    img_dimensions = self.orig.shape
    nodes = nodes[selected_idx, :]
    
    # Add corners
    nodes = np.append(nodes, [[0, 0]], axis=0)
    nodes = np.append(nodes, [[img_dimensions[0], 0]], axis=0)
    nodes = np.append(nodes, [[0, img_dimensions[1]]], axis=0)
    nodes = np.append(nodes, [[img_dimensions[0], img_dimensions[1]]], axis=0)
    return nodes

  def _create_triangles(self, nodes):
    tri = Delaunay(nodes)
    return tri

  def _draw_poly_lines(self, tri):
    # Draw only the lines of the poly image
    points = tri.points
    simplices = tri.simplices
    all_poly = []
    for simplex in simplices:
      tri_points = []
      for vertex in simplex:
        tri_points.append(points[vertex])
      all_poly.append(tri_points)
    cv.polylines(self.low_poly_img, 
                np.array(all_poly, dtype=np.int32), 
                True,
                (0, 0, 1))

  def _draw_poly_points(self, tri):
    # Draw only the vertices of the poly image
    points = tri.points
    for point in points:
      cv.circle(self.low_poly_img, 
                (int(point[0]), int(point[1])),
                1, 
                [0, 0, 1],
                -1)
  
  def _draw_poly_img(self, tri):
    points = np.array(tri.points)
    simplices = tri.simplices
    for simplex in simplices:
      tri_points = []
      for vertex in simplex:
        tri_points.append([int(points[vertex][1]), int(points[vertex][0])])
      centroid = (
        int((tri_points[0][1] + tri_points[1][1] + tri_points[2][1])/3),
        int((tri_points[0][0] + tri_points[1][0] + tri_points[2][0])/3)
      )
      if centroid[0] >= self.low_poly_img.shape[0]:
        centroid = (self.low_poly_img.shape[0]-1, centroid[1])
      if centroid[1] >= self.low_poly_img.shape[1]:
        centroid = (centroid[0], self.low_poly_img.shape[1]-1)
      
      color = tuple([x/255 for x in self.orig[centroid]])
      cv.fillPoly(self.low_poly_img, 
                  np.array([tri_points], dtype=np.int32), 
                  color)

  def plot_nodes(self):
    pp_img = self._preprocess(self.orig)
    nodes = self._get_nodes(pp_img)
    triangles = self._create_triangles(nodes)
    plt.scatter(nodes[0], nodes[1])
    plt.triplot(nodes[:,0], nodes[:,1], triangles.simplices.copy())
    plt.plot(nodes[:,0], nodes[:,1], 'o')
    plt.show()

  def get_low_poly(self):
    pp_img = self._preprocess(self.orig)
    nodes = self._get_nodes(pp_img)
    triangles = self._create_triangles(nodes)
    self._draw_poly_img(triangles)
    return self.low_poly_img
  
  def show_low_poly(self):
    cv.imshow("Low Poly image", self.get_low_poly())
    cv.waitKey(0)

  def save_low_poly(self, path):
    cv.imwrite(path, self.get_low_poly() * 255)