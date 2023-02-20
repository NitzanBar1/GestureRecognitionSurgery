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
|`images`| Images used for preview in README.md file|



## Dataset
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/vis.png)
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/insights.png)



## Models
•	**MSTCN++ architecture:**  
We used MSTCN++ as the baseline architecture.
MS-TCN++ is a temporal convolutional neural network (TCN) that was designed originally for activity recognition in video data. 
The input to the network was the pre processed video frames data.
The first stage is the prediction generation stage, and the following stages are the refinement stages. 
More specifically, that first stage consists of a prediction generator that produces the initial predictions of the network, which is fed into several refinement stages, such that each stage outputs a prediction of the network. 


An illustration of the MSTCN++ architecture is shown below:

![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/mstcn2.png)


## Results
The models were compiled in Microsoft Azure using PyTorch packages with 25 epochs and early stopping.   
**MSTCN++ Baseline Results:**  

![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/graphs1.png)
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/graphs2.png)

Model detections:  
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/results.png)

Evaluation metrics:  


## References
[1] Farha, Y.A. and Gall, J., 2019. Ms-tcn: Multi-stage temporal convolutional network for action segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 3575-3584).

