# Image Processing Functions

Studying notes taken from Udacity Computer Vision Nanodegree

---

### Display multiple images

```python3
import matplotlib.pyplot as plt

%matplotlib inline

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('figure A')
ax1.imshow(image_A, cmap='gray')

ax2.set_title('figure B')
ax2.imshow(image_B, cmap='gray')
```


### Load image

```python3
import matplotlib.image as mpimg

image = mpimg.imread('images/image1.jpg')
```

```python3
import cv2

image = cv2.imread('images/image1.jpg')
```

### Covert image types

```python3
import cv2

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
```

### Apply simple filter to images

```python3
filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # sobel filter
filtered_image = cv2.filter2D(gray_image, -1, filter)
```

### Gaussian blur

```python3
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
```

### Create binary mask

```python3
ret, binary_image = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)
```

### Canny edge detector

```python3
lower = 120
upper = 240  # usually (2~3) x lower

edges = cv2.Canny(gray_image, lower, upper)
```

### Hough line detector

```python3
rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 50
max_line_gap = 5

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

lined_image = np.copy(gray_image)

for line in lines:
  for x1, y1, x2, y2 in line:
    cv2.line(lined_image, (x1,y1), (x2,y2), (255,0,0), 5)

```

### Hough circle detector

[cv2 documentation](https://docs.opencv.org/master/d3/de5/tutorial_js_houghcircles.html)

```python3
detected_circles = cv2.HoughCircles(image=gray_image,
                                    method=cv2.HOUGH_GRADIENT,
                                    dp=1,
                                    minDist=45,
                                    param1=70,
                                    param2=11,
                                    minRadius=20,
                                    maxRadius=40)
                                    
detected_circles = np.uint16(np.around(circles))

# draw detected circles on image
for cir in circles[0, :]:
  # draw outer circle
  cv2.circle(gray_image, center=(cir[0], cir[1]), radius=cir[2], color=(0,255,0), thickness=2)
  # mark the circle center
  cv2.circle(gray_image, center=(cir[0], cir[1]), radius=2, color=(0,0,255), thickness=3)

```

### Haar cascades

```python3
# load the classifier
clf = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# detect faces
detected_faces = clf.detectMultiScale(gray_image, scaleFactor=4, minNeighbors=6)

# plot bounding boxes on the image
for (x, y, w, h) in detected_faces:
  cv2.rectangle(gray_image, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=5)

plt.imshow(gray_image)
```




