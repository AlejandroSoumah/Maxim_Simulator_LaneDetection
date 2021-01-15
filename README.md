## Creation of an Lane Detection used for lattice planner in self-driving car in Carla Simulator


[![Watch the video](https://github.com/AlejandroSoumah/Maxim_Simulator_LaneDetection/blob/master/Screenshot_from_Lane_Detection_Works_2.mov.png)](https://youtu.be/XxF9UrvCx5w)

This shows the functioning of the Lane Detection and the short-term waypoint creation to maintain the vehicle at the center of the lane.

My algorithm needed the visual-perception to be of a high accuracy thus making it one of the modules of this project that I have worked in the most.

Furthermore for the short-term creation I used camera-geometry to convert certain pixels to a global coordinate frame. I took several weeks to correctly make this transformations.

### To RUN:
   1. Install Python x3.6
   2. Install Carla-Simulator
   3. Install the following libraries:
        - OpenCV
        - Tensorflow 2.0
        - Numpy
   4.Run Carla-Simulator
   5.Run module7.py
