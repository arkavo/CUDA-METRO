import numpy as np



dmi6 = [[ 0.8660254,0.5,-2.0],
 [-0.8660254,0.5,-2.0],
 [ 0.,1.,2.0],
 [ 0.8660254,-0.5,2.0],
 [-0.8660254,-0.5, 2.0],
 [ 0.,-1.,-2.0]]

dmi3 = [[-0.8660254, 0.5, -2.0],
        [ 0.8660254,-0.5,2.0],
        [ 0.,-1.,-2.0]]

print(dmi6)
print(dmi3)
np.save('dmi_3.npy', dmi3)
np.save('dmi_6.npy', dmi6)
di3 = np.load('dmi_3.npy')
print(di3)
di6 = np.load("dmi_6.npy")
print(di6)
