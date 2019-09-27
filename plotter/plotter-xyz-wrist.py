import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

fig, axs = plt.subplots(3, sharex=True, sharey=False)

o1_parent = [1, 1,2,3,4, 1, 6,7,8, 1, 10,11,12, 1, 14,15,16, 1, 18,19,20]
o1_parent = [x-1 for x in o1_parent]

baseline = sys.argv[1]
new_result = sys.argv[2]
keys = 'wristPosition0X wristPosition0Y wristPosition0Z'

keys = keys.split(' ')
places_dict = {}
places_dict1 = {}

with open(baseline,'r') as orig:
    lines_orig = orig.readlines()
    cells = lines_orig[0].split(',')
    for key in keys:
        a = cells.index(key)
        places_dict1[key] = a

    with open(new_result,'r') as res:
        lines = res.readlines()
        cells = lines[0].split(',')
        for key in keys:
            a = cells.index(key)
            places_dict[key] = a
        
        # for line,line1 in zip(lines[1:],lines_orig[1:]):
        x_list = []
        y_list = []
        z_list = []
        x_list1 = []
        y_list1 = []
        z_list1 = []
        t_list = []
        t_list1 = []
        for line in lines[1:]:
            line=line.split(',')
            joints = np.zeros((3,))
            for i in range(0,len(keys)):
                joints[i%3] = float(line[places_dict[keys[i]]])
            t_list.append(line[1])
            x_list.append(joints[0])
            y_list.append(joints[1])
            z_list.append(joints[2])

        for line in lines_orig[1:]:
            line=line.split(',')
            joints = np.zeros((3,))
            for i in range(0,len(keys)):
                joints[i%3] = float(line[places_dict1[keys[i]]])
            t_list1.append(line[1])
            x_list1.append(joints[0])
            y_list1.append(joints[1])
            z_list1.append(joints[2])

        t_list = np.array(t_list)
        x_list = np.array(x_list)
        t_list1=np.array(t_list1)
        x_list1=np.array(x_list1)
        # if len(t_list) > len(t_list1):
   
        # df = pd.DataFrame({"time":t_list, "orig": x_list1, "result": x_list})
        
        # df.plot(x='time')
        axs[0].plot(t_list,x_list, color='r')
        axs[0].plot(t_list1,x_list1,color='b')
        
        axs[1].plot(t_list,y_list, color='r')
        axs[1].plot(t_list1,y_list1,color='b')


        axs[2].plot(t_list,z_list, color='r')
        axs[2].plot(t_list1,z_list1,color='b')
        # ax2.set_title("orig")
        # ax.set_title("result")
        plt.show()

        # plt.pause(0.1)
            
            # plt.cla()
