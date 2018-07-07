## Requirement

- boost::python
- boost::numpy
- chainer
- chainercv

## Installation

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

```bash
roscore
roslaunch robosherlock rs.launch ae:=my_demo
rosbag play test.bag --loop --clock
```

- Run boost::python and boost::numpy example

```bash
roscore
roslaunch robosherlock rs.launch ae:=python_test
rosbag play test.bag --loop --clock
```

- Run Faster-RCNN demo

```bash
roscore
roslaunch robosherlock rs.launch ae:=faster_test
rosbag play test.bag --loop --clock
```

- Run SSD demo

```bash
roscore
roslaunch robosherlock rs.launch ae:=ssd_test
rosbag play test.bag --loop --clock
```

- Run FCIS demo

```bash
roscore
roslaunch robosherlock rs.launch ae:=fcis_test
rosbag play test.bag --loop --clock
```
