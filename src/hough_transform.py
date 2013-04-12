import cv2 as cv
import scipy.sparse;
import numpy as np
from matplotlib.image import imread
import scipy;
import scipy.sparse;
import time


def benchmark(func, *args):
    start_time = time.time()
    output = func(*args)
    elapsed_time = time.time() - start_time
    print func.func_name, elapsed_time, 'sec'
    return output 

def get_2nd_point(grad_direction, i, j, l):
    ii = i - l*np.sin(grad_direction[i, j])
    jj = j + l*np.cos(grad_direction[i, j])
    return (int(jj), int(ii))

def drawL(image, i, j, l):
    cv.line(image, (j, i), get_2nd_point(i ,j, l), 150)            


def grad_CHT(image, r, vote_space_ratio):
    smoothed = cv.medianBlur(image, 7)
    edges = cv.Canny(smoothed, 250, 280)
    
    grad_y = cv.Sobel(smoothed, cv.CV_16S, 1, 0)
    grad_x = cv.Sobel(smoothed, cv.CV_16S, 0, 1)
    grad_direction = np.arctan2(grad_y, grad_x) - np.pi/2 
    
    rows = np.size(edges, 0)
    cols = np.size(edges, 1)
    
    vote_rows = int(rows / vote_space_ratio)
    vote_cols = int(cols / vote_space_ratio)
    
    result = np.zeros([vote_rows, vote_cols])
    
    for i in np.arange(rows):
        for j in np.arange(cols):
            if edges[i, j]:
                a = i - r * np.sin(grad_direction[i, j])
                b = j + r * np.cos(grad_direction[i, j])
                
                a = int(a / vote_space_ratio)
                b = int(b / vote_space_ratio)
                
                if a >= 0 and b >= 0 and a < vote_rows and b < vote_cols:
                    result[a, b] = result[a, b] + 1
    return result
                


def CHT(image, R):
    smoothed = cv.medianBlur(image, 7)
    edges = cv.Canny(smoothed, 250, 280)
    rows = np.size(edges, 0)
    cols = np.size(edges, 1)
    
    result = np.zeros((rows, cols))
    
    for row in np.arange(rows):
        for col in np.arange(cols):
            if edges[row, col]:
                min_height = row - R
                max_height = row + R
                
                if min_height < 0:
                    min_height = 0
                    
                if max_height >= rows:
                    max_height = rows - 1
                
                for a in np.arange(min_height, max_height):
                    squared_dif = R**2 - (row - a)**2
                    
                    if squared_dif >= 0:
                        dif = np.sqrt(squared_dif)
                        b1 = int(col - dif)
                        b2 = int(col + dif)
                        
                        if b1 >= 0 and b1 < cols:
                            result[a, b1] = result[a, b1] + 1
                        
                        if b2 >= 0 and b2 < cols:
                            result[a, b2] = result[a, b2] + 1
    return result

def non_maximum_suppression_and_prunning(centers):
    rows = np.size(centers, 0)
    cols = np.size(centers, 1)
    
    def is_valid(ar):
        return ar[0] >= 0 and ar[1] >= 0 and ar[0] < rows and ar[1] < cols
    
    delta = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    
    # less than 50% of the max vote will be considered as outlier
    t = np.max(centers) * 0.5
    centers = (centers > t) * centers
    
    for i in np.arange(rows):
        for j in np.arange(cols):
            if centers[i, j]:
                neighbours = delta + [i, j]
                for n in neighbours:
                    if is_valid(n):
                        if centers[i, j] < centers[n[0], n[1]]:
                            centers[i, j] = 0
                            break
                        
    return centers

def draw_circules(image, centers, r, color, vote_space_ratio):
    rows = np.size(centers, 0)
    cols = np.size(centers, 1)
    
    for i in np.arange(rows):
        for j in np.arange(cols):
            if centers[i, j]:
                point = (j*vote_space_ratio + vote_space_ratio/2, i*vote_space_ratio + vote_space_ratio/2)
                cv.circle(image, point, r, color)
                cv.circle(image, point, r+1, color)
                cv.circle(image, point, r+2, color)
                cv.circle(image, point, r+3, color)

def detec_circles(image_input, image_output, radius, color):
    ratio = 4
    result = benchmark(grad_CHT, image_input, radius, ratio)
    result = benchmark(non_maximum_suppression_and_prunning, result)
    draw_circules(image_output, result, radius, color, ratio)

image_input = cv.imread("../test_cases/coins_1.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
image_output = cv.imread("../test_cases/coins_1.jpg")
import cv as cv2
detec_circles(image_input, image_output, 135, cv2.RGB(255, 125, 85))
detec_circles(image_input, image_output, 120, cv2.RGB(150, 125, 220))
detec_circles(image_input, image_output, 107, cv2.RGB(200, 220, 85))
cv.imwrite("../output/1.jpg", image_output)

#print np.max(result)
#print scipy.sparse.coo_matrix(result3)
