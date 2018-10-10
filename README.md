## Point-On-Object-Detection

### Describe
This project is to detection a point which on a target object and extract the moving track of the point.
This project is simply used L-K Tracker, SVM and sliding windows. 


![image text](https://github.com/BLUECARVIN/Point-On-Object-Detection/blob/master/Test.png)

### How to use 
1. You need a series of training photos of the object you need to detect. Including some positive photos and negative photos. 
2. Put them in the root folder of this project.
3. Preparing a video needed to detection, and put it in the root folder of the project.
4. Run 'main.py'(before running, you should confirm the video name in the 'main.py' is correct)
5. Adjust the parameters of LKTrack or SVM or Sliding windows to make the performance better.


