# GestureRecognitionSurgery

The aim of this project is to identify surgical gestures and tools used during an open surgery suturing simulation.
Based on motion sensor data and video data that captured using two cameras, one closeup camera focusing on the simulation material and one overview camera that included the surrounding area.  
This is a class project as part of 097222 Computer Vision Seminar @ Technion.  

<p align="center">
    <a href="https://www.linkedin.com/in/nitzan-bar-9ab896146/">Nitzan Bar</a> | 
    <a href="https://www.linkedin.com/in/ido-levi-869a96177/">Ido Levi</a>
</p>


- [GestureRecognitionSurgery](#gesture-recognition-surgery)
  * [Files in The Repository](#files-in-the-repository)
  * [Dataset](#dataset) 
  * [Models](#models)
  * [Results](#results)
  * [References](#references)



## Files in the repository
|File name         | Purpsoe |
|----------------------|------|
|`clean_env.yml`| conda enviornment file|
|`main.py`| train the model|
|`eval.py`| predict and evaluate performence|
|`model.py`| models defintion|
|`batch_gen.py`| create batched data|
|`predcit_on_video.py`| create videos with predictions|
|`images`| Images used for preview in README.md file|



## Dataset
The data consists of 4 folds, the labels distribution towards the folds is:
![alt text](https://github.com/NitzanBar1/GestureRecognitionSurgery/blob/main/images/folds_table.png)


## Models
•	**MSTCN++ architecture:**  
We used MSTCN++ as the baseline architecture.
MS-TCN++ is a temporal convolutional neural network (TCN) that was designed originally for activity recognition in video data. 
The input to the network was the pre processed video frames data.
The first stage is the prediction generation stage, and the following stages are the refinement stages. 
More specifically, that first stage consists of a prediction generator that produces the initial predictions of the network, which is fed into several refinement stages, such that each stage outputs a prediction of the network. 


An illustration of the MSTCN++ architecture is shown below:

![alt text](https://github.com/NitzanBar1/GestureRecognitionSurgery/blob/main/images/mstcn2.png)


## Results
The models were compiled in Microsoft Azure using PyTorch packages with 10 epochs.   
**MSTCN++ Baseline Results:**  

![alt text](https://github.com/NitzanBar1/GestureRecognitionSurgery/blob/main/images/graphs1.png)
![alt text](https://github.com/NitzanBar1/GestureRecognitionSurgery/blob/main/images/graphs2.png)
![alt text](https://github.com/NitzanBar1/GestureRecognitionSurgery/blob/main/images/baseline_table.png)

Model detections:  
![alt text](https://github.com/NitzanBar1/GestureRecognitionSurgery/blob/main/images/prediction.png)


Evaluation metrics:  
We adopted three evaluation metrics in our experiments: frame-wise accuracy, edit score, and segmented F1 score. Frame-wise accuracy is to measure the performance in frame level. However, long gesture segments tend to have more impact than short gesture segments, and the frame-wise accuracy is not sensitive to the over-segmentation error. Therefore, we use the edit score and F1 score to assess the model at the segmental level. 
The edit score is defined as the normalized Levenshtein distance between the prediction and the ground truth. In contrast, F1 score is the harmonic mean of precision and recall with the threshold 10%, 25%, and 50%.


## References
[1] Farha, Y.A. and Gall, J., 2019. Ms-tcn: Multi-stage temporal convolutional network for action segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 3575-3584).

