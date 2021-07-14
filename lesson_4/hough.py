import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2  


image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

#Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color=255

#This time we are defining a four sided polygon to mask
imshape = image.shape
print(imshape)
#(0,imshape[0]), (450, 290), (490,290), (imshape[1], imshape[0])
vertices = np.array([[(80,imshape[0]), (450,290), (490, 290), (890, imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
maske_edges = cv2.bitwise_and(edges, mask)

#Define the hough transform parameters
# Make a blank the same size as our image to draw on
rho=1 #2
theta=np.deg2rad(1) #1
threshold=10 #15
min_line_length=25 #40
max_line_gap=5 #20
line_image = np.copy(image)*0

#Run hough on edges detected image
lines = cv2.HoughLinesP(maske_edges,
                        rho,
                        theta,
                        threshold,
                        np.array([]),
                        min_line_length,
                        max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

# Display the image
plt.imshow(combo)
plt.show()