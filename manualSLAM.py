# Manually drive the robot inside the arena and perform SLAM using ARUCO markers

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import json
import time

# import csv

# import object detection
from my_module import yolo_detection as YOLO

# Import keyboard teleoperation components
import penguinPiC
import keyboardControlARtestStarter as Keyboard

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
import slam.Slam as Slam
import slam.Robot as Robot
import slam.aruco_detector as aruco
import slam.Measurements as Measurements

coke_list = []
sheep_list = []

seen_coke_list = []
seen_sheep_list = []

# coke_list = ["coke1", "coke2"]

# Manual SLAM
class Operate:
    def __init__(self, datadir, ppi):
        # Initialise
        self.ppi = ppi
        self.ppi.set_velocity(0, 0)
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)

        # Keyboard teleoperation components
        self.keyboard = Keyboard.Keyboard(self.ppi)

        # Get camera / wheel calibration info for SLAM
        camera_matrix, dist_coeffs, scale, baseline = self.getCalibParams(datadir)

        # SLAM components
        self.pibot = Robot.Robot(baseline, scale, camera_matrix, dist_coeffs)
        self.aruco_det = aruco.aruco_detector(self.pibot, marker_length=0.1)
        self.slam = Slam.Slam(self.pibot)

        self.classes = ["coke", "sheep"]


    #def __del__(self):
        #self.ppi.set_velocity(0, 0)

    def getCalibParams(self, datadir):
        # Imports camera / wheel calibration parameters
        fileK = "{}camera_calibration/intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}camera_calibration/distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}wheel_calibration/scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "{}wheel_calibration/baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')

        return camera_matrix, dist_coeffs, scale, baseline

    def control(self, dt):
        # Import teleoperation control signals
        lv, rv = self.keyboard.latest_drive_signal()
        drive_meas = Measurements.DriveMeasurement(lv, rv, dt=dt)
        self.slam.predict(drive_meas)

    def vision(self):
        # Import camera input and ARUCO marker info
        self.img = self.ppi.get_image()
        lms, aruco_image = self.aruco_det.detect_marker_positions(self.img)
        self.slam.add_landmarks(lms)
        self.slam.update(lms)

    def get_object_location(self):
        object = YOLO.get_cordinate(self.img)
        if(object):
            
            rpos = self.slam.robot.state.tolist()
            pos = [rpos[0]+object[1]*np.cos(rpos[2]), rpos[1]+object[1]*np.sin(rpos[2])]
            objpos = [pos[0][0], pos[1][0]]

            if(object[0] == 0):
                print("Found coke at", objpos)
                coke_list.append(objpos)
            else:
                print("Found sheep at", objpos)
                sheep_list.append(objpos)

        else: 
            print("There is no object.")

    def display(self, fig, ax):
        # Visualize SLAM
        ax[0].cla()
        self.slam.draw_slam_state(ax[0])

        ax[1].cla()
        ax[1].imshow(self.img[:, :, -1::-1])

        plt.pause(0.01)

    def write_map(self, slam):
    
        # Output SLAM map as a json file
        map_dict = {"AR_tag_list":slam.taglist,
                    "map":slam.markers.tolist(),
                    "State":slam.robot.state.tolist(),
                    "Coke": coke_list, 
                    "Sheep": sheep_list}
                    #"covariance":slam.P[3:,3:].tolist()}
        with open("Lab01_M5_Map_Group05.txt", 'w') as map_f:
            json.dump(map_dict, map_f, indent=2)


    def process(self):
        # Show SLAM and camera feed side by side
        fig, ax = plt.subplots(1, 2)
        img_artist = ax[1].imshow(self.img)

        # print("process")
        #starttime = time.time()]
        dt = 0.3
	
        # Main loop
        while True:
            # Run SLAM
            start_time = time.time()

            self.control(dt)
            self.vision()

            # starting object detcion
            # print(self.keyboard.capture_img)
            if(self.keyboard.capture_img):
                self.get_object_location()
            self.keyboard.capture_img = False
            # ending object detection


            # Save SLAM map
            self.write_map(self.slam)

            # Output visualisation
            # self.display(fig, ax)
            #time.sleep(0.5-((time.time()-starttime)%0.5))
            dt = (time.time()-start_time) * 0.5
            # print(dt)

if __name__ == "__main__":
    # Location of the calibration files
    currentDir = os.getcwd()
    datadir = "{}/calibration/".format(currentDir)
    # connect to the robot
    ppi = penguinPiC.PenguinPi()

    # Perform Manual SLAM
    operate = Operate(datadir, ppi)
    operate.process()



        # field_names= ['object', 'x', 'y']
        #     manual_slam_map =[
        #     {'object':AR_tag_list[1], 'x':map[1][1], 'y':map[1][2]},
        #     {'object':AR_tag_list[2], 'x':map[2][1], 'y':map[2][2]},
        #     {'object':AR_tag_list[3], 'x':map[3][1], 'y':map[3][2]},
        #     {'object':AR_tag_list[4], 'x':map[4][1], 'y':map[4][2]},
        #     {'object':AR_tag_list[5], 'x':map[5][1], 'y':map[5][2]},
        #     {'object':AR_tag_list[6], 'x':map[6][1], 'y':map[6][2]},
        #     ]
        # with open('manual_slam_map.csv', 'w') as csvfile:
        #     writer = csv.dictWriter(csvfile, fieldnames=field_names)
        #     writer.writerow(manual_slam_map)    
