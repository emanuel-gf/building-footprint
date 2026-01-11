Creating a rectangle from a bitmap image

After retrieving the contour of the image, the centroid is being select
by the momentum in cv2. Then a bounding box is being created by the given centroid
and w,d. However, a few choice was implemented:

The bounding box can have as aspect ratio diferent than 1. 
But the choice was a perfect squared bounding box. 


Context: Rooftop classifiers often need to see the "gutter" or the edge where the roof meets the ground. Padding ensures you don't cut too close.Distortion Prevention: By forcing a 1.0 aspect ratio (square), you prevent the image from being "squashed" or "stretched" when you resize the crop to feed it into a neural network (e.g., $224 \times 224$).Centricity: Using the centroid ensures the building is the focal point, even if the building is L-shaped or U-shaped