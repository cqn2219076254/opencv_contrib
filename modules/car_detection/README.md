#### Introduction

This project is to provide a 3D point cloud data-based vehicle detection method for OpenCV. The original model used in this project is the [Point-based 3D Single Stage Object Detector](https://arxiv.org/abs/2002.10187) (3DSSD) model. .

Model inference accuracy: 3DSSD model validation results on KITTI dataset (3,769 test samples) achieved **91.71% (easy), 80.44% (hard)**

| Easy AP | Moderate AP | Hard AP |
| :-----: | :---------: | :-----: |
|  91.71  |    83.30    |  80.44  |

At present, we have obtained two models adapted to opencv 4.5.1 based on the original model, and the predicted results are identical to those of the original model with the same inputs.

#### Environmental requirements

* Python 3.7
* OpenCV 4.5.1
* g++ 5.4
