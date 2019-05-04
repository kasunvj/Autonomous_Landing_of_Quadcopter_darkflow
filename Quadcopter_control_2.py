import vrep
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from darkflow.net.build import TFNet

_, ax = plt.subplots(figsize=(20, 10))
##################################################################
def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
            
    return newImage

def go(x,y,z,res):

	if q[0] > x:
		q[0] = q[0] - res
	elif q[0] < x:
		q[0] = q[0] + res

	if q[1] > y:
		q[1] = q[1] - res
	elif q[1] < y:
		q[1] < q[1] + res

	if q[2] > z:
		q[2] = q[2] - res
	elif q[2] < z:
		q[2] = q[2] + res

	if ( q[0] == x) & ( q[1] == y) & ( q[2] == z) :
		q[0] = x
		q[1] = y
		q[2] = z
		done =  1
	else :
		err_code = vrep.simxSetObjectPosition(clientID,target_handle,-1,[q[0],q[1],q[2]],vrep.simx_opmode_streaming)
		time.sleep(0.1)
		done =  0 

##################################################################
q = [0,0,0]
A = [2 ,2 ,3] # this is be near to the Quadcopter object placed on V-REP. Beacause you have to initiate from closer
B = [2, 1 ,3]
C = [1, 1, 3]
D = []
E = []
done = 0

#################################################################

plot_floorcamera_pos = [0,1]

vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
print(clientID) # if 1, then we are connected.
if clientID!=-1:
	print ("Connected to remote API server")
else:
	print("Not connected to remote API server")

#getting the handels of thigs we control over here 
err_code,floorcamera_handle = vrep.simxGetObjectHandle(clientID,"Quadricopter_floorCamera",vrep.simx_opmode_blocking)
err_code,target_handle = vrep.simxGetObjectHandle(clientID,"Quadricopter_target",vrep.simx_opmode_blocking)
err_code,camera = vrep.simxGetObjectHandle(clientID,"Vision_sensor",vrep.simx_opmode_oneshot_wait)

#Get what we need from the things we hold here 
err_code,target_pos = vrep.simxGetObjectPosition(clientID,target_handle,-1,vrep.simx_opmode_blocking)
err_code,floorcamera_pos = vrep.simxGetObjectPosition(clientID,floorcamera_handle,-1,vrep.simx_opmode_blocking)
err_code,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_streaming)


###################################################################
#Initiate darkflow engine 
options = {	"load": "bin/yolov2.weights",
		    "model": "cfg/yolo.cfg",
            "threshold": 0.05, 
            "demo": "arial2.avi",
            "saveVideo": True
		}


tfnet = TFNet(options)

#Give 1st target
visit = 0

print("On your mark")
time.sleep(2)
err_code = vrep.simxSetObjectPosition(clientID,target_handle,-1,[A[0],A[1],A[2]],vrep.simx_opmode_streaming)

q = A

print("Get set")
time.sleep(2)

print("Go!!")
t1 = time.time()

while (True):
		#getting image
		err_code,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,0,vrep.simx_opmode_buffer)
		if err_code == vrep.simx_return_ok:
			#print("image Ok")
			img = np.array(image, dtype = np.uint8)
			img.resize([resolution[1],resolution[0],3])

			results = tfnet.return_predict(img)
			new_img = boxing(img, results)

			cv2.imshow('image',new_img)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			elif err_code == vrep.simx_return_novalue_flag:
				print("no image")
				pass
			

		#get position odf target and copter
		err_code,target_pos = vrep.simxGetObjectPosition(clientID,target_handle,-1,vrep.simx_opmode_blocking)
		err_code,floorcamera_pos = vrep.simxGetObjectPosition(clientID,floorcamera_handle,-1,vrep.simx_opmode_blocking)
		#print(target_pos)
		print(floorcamera_pos)

		#plotting graphs of movement
		plot_floorcamera_pos = floorcamera_pos.append(floorcamera_pos[0])
		#plt.plot(target_pos[0])
		#plt.plot(plot_floorcamera_pos)
		#plt.show()
		
		#print(time.time()-t1)
		#if visit == 0 :
		#			err_code = vrep.simxSetObjectPosition(clientID,target_handle,-1,[1,2,3],vrep.simx_opmode_streaming)
		#			time.sleep(0.2)
		#			visit = 1

		
		#if done == 0:
		#	go(2,1,3,0.5)
		#else:
		#	print("Done")

		err_code = vrep.simxSetObjectPosition(clientID,target_handle,-1,[2,2,4],vrep.simx_opmode_streaming)
		time.sleep(3)

		





print("Done")