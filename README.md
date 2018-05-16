## installation

- Install robosherlock

Detailed information is [here](https://robosherlock.org/install.html)

```bash
mkdir catkin_ws/src -p
cd catkin_ws/src
git clone https://github.com/Robosherlok/robosherlock.git
git clone https://github.com/Robosherlok/robosherlock_msgs.git
```

- Clone this repo

```bash
git clone https://github.com/knorth55/rs_test.git
cd catkin_ws
catkin build
```

- Run the tutorial

Download `test.bag` from [here](https://robosherlock.org/tutorials/pipeline.html)

```
roscore
rosbag play test.bag --loop --clock
roslaunch robosherlock rs.launch ae:=my_demo 
```
