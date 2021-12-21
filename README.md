# Project Road Segmentation

We get a set of satellite images acquired from GoogleMaps. We also get ground-truth images where each pixel is labeled 
as road or background. 

The task is to train a classifier to segment roads in these images, i.e. 
assigns a label `road=1, background=0` to each pixel.

The dataset is available from the 
[CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)

TODO