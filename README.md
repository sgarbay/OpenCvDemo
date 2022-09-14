# OpenCvDemo
Sample to explore OpenCV for real-time image processing on android.

## Hand detection
The hand is detected based on its colour. The so computed mask is then processed by Dilation and Erosion to get a mostly black image with only white for the hand. Followingly the center of the largest blob is computed, which is assumed to be the hand.

## Performance
As only simple OpenCV the hand detection masters to run at 30FPS. The accuracy of the hand detection can be further improved by initialising the algorithm with pictures of the hand.