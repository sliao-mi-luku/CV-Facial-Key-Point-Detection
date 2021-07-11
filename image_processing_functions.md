# Image Processing Functions


### Load image

```python3
import matplotlib.image as mpimg

image = mpimg.imread('images/image1.jpg')
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

### 
