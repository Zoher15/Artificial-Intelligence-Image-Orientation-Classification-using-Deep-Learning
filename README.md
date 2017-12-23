# Machine Learning: Image Classification
These days, all modern digital cameras include a sensor that detects which way the camera is being held when a photo is taken. This meta-
data is then included in the image file, so that image organization programs know the correct orientation - i.e., which way is "up" in the image. But for photos scanned in from film or from older digital cameras, rotating images to be in the correct orientation must typically be done by hand.

# Data
A dataset of images from the Flickr photo sharing website. The images were taken and uploaded by real users from across the world, making this a challenging task on a very realistic dataset. The raw images have been treated as numerical feature vectors, on which we standard machine learning techniques can be applied. In particular, we are using take an n x m x 3 color image (the third dimension is because color images are stored as three separate planes { red, green, and blue), and append all of the rows together to produce a single vector of size 1 x 3mn. We've done this work and simply treat images as vectors and do not care about them being images at all. The GitHub repo includes two ASCI dataset and one for testing, that contain the feature vectors. To generate this fille, each image has been rescaled to a very tiny "micro-thumbnail" of 8x8 pixels, resulting in an 8x8x3 = 192 dimensional feature vector.

Data and Supervision by **David Crandall**

# Part 1: KNN
At test time, for each image to be classified, the program finds the k "nearest" images in the training file, i.e. the ones with the closest distance (least vector difference) in Euclidean space, and have them vote on the correct orientation.

Code by **Ankita Alshi**

# Part 2: AdaBoost
Used very simple decision stumps that simply compare one entry in the image matrix to another, e.g. compare the red pixel at position 1,1 to the green pixel value at position 3,8.

Code by **Murtaza Khambaty**

# Part 3: Neural Networks
A fully-connected feed-forward network to classify image orientation, and implements the backpropagation algorithm to train the network using gradient descent.

Code by **Zoher Kachwala**
