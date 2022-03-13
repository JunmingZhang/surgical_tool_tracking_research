# surgical_tool_tracking_research
This is the research project (CSC494 in University of Toronto) supervised by Lueder Kahrs and Radian Gondokaryono, for surgical tool tracking.<br/>
Please view the project description at: https://usra.cs.toronto.edu/public/project/342?sort_by=0&opportunity_id=6 <br/>
Please view our organization web page at: https://www.cs.toronto.edu/~lakahrs/ <br/>
My main jobs are:
  <ol>
  <li> Data simulation in Unity, include randomization, stereo camera implementation and camera calibration scene. </li>
  <li> Collect synthetic data with key points (by perception package), then use the data for triangulation (get 3D key points from 2D key points). </li>
  <li> Design camera calibration and triangulation pipeline. </li>
  <li> Create and label the training data, then use the data to train the 2D key point detection model (in deeplabcut). </li>
  </ol>
 Special thanks for Mustafa Haiderbhai in MedCVR, who contributes many scripts I adapted for my work.

## Important packages
* [unity perception package](https://github.com/Unity-Technologies/com.unity.perception)
* [datainsights](https://github.com/Unity-Technologies/datasetinsights)
* [camera calibration in OpenCV)(https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html)
* [3D reconstruction in OpenCV)(https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html)
* [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)

## Important tutorials
* [Camera Calibration in ROS](http://wiki.ros.org/camera_calibration/Tutorials/StereoCalibration)
* [Perception tutorial](https://github.com/Unity-Technologies/com.unity.perception/blob/main/com.unity.perception/Documentation~/Tutorial/TUTORIAL.md)
* [Robotics Object Pose Estimation](https://github.com/Unity-Technologies/Robotics-Object-Pose-Estimation)
* [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/docs/intro.html)

## Data Simulation
Data (calibration and model training) are simulated by unity.
* [unity project](https://mcsscm.utm.utoronto.ca/medcvr/medcvr-unity/-/tree/wild/junming/data_simulation)

## Base repos (from MedCVR organization)
- [MedCVR Unity](https://mcsscm.utm.utoronto.ca/medcvr/medcvr-unity/-/tree/feature/new_tool_dataset/)
- [MedCVR ML](https://mcsscm.utm.utoronto.ca/medcvr/medcvr-ml)

## References
_copied and pasted from project description page_
- D'Ettorre et al. "Accelerating Surgical Robotics Research: Reviewing 10 Years of Research with the dVRK." arXiv preprint arXiv:2104.09869 (2021).  https://arxiv.org/abs/2104.09869  
- Lu et al. "SuPer Deep: A Surgical Perception Framework for Robotic Tissue Manipulation using Deep Learning for Feature Extraction" https://sites.google.com/ucsd.edu/super-framework 
- Kazanzides et al. "An Open-Source Research Kit for the da Vinci Surgical System", https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki/kazanzides-chen-etal-icra-2014.pdf 
- Lu et al. "Robust keypoint detection and pose estimation of robot manipulators with self-occlusions via sim-to-real transfer." arXiv preprint arXiv:2010.08054 (2020)  https://arxiv.org/abs/2010.08054  
- Liu et al. "Real-to-Sim Registration of Deformable Soft Tissue with Position-Based Dynamics for Surgical Robot Autonomy." 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021. https://arxiv.org/abs/2011.00800  
