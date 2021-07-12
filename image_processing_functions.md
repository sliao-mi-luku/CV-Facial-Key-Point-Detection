# Image Processing Functions


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

### Hough transform

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

### 






