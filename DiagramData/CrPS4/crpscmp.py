import numpy as np
import matplotlib.pyplot as plt

ref_data = np.loadtxt("CrPS4_0K-213K_M-X.dat",delimiter=",")

T_ref = np.array(ref_data[:,0]).astype(np.float32)
M_ref = np.array(ref_data[:,1]/ref_data[:,1].max()).astype(np.float32)
X_ref = np.array(ref_data[:,2]/ref_data[:,2].max()).astype(np.float32)

print(T_ref.shape)
print(M_ref.shape)

m_data_625 = np.load("BQM_64_256_625.npy")/np.load("BQM_64_256_625.npy").max()
x_data_625 = np.load("BQX_64_256_625.npy")
x_data_625[0] = 0.0
x_data_625 = x_data_625/x_data_625.max()
m_data_25 = np.load("BQM_64_1024_25.npy")/np.load("BQM_64_1024_25.npy").max()
x_data_25 = np.load("BQX_64_1024_25.npy")
x_data_25[0] = 0.0
x_data_25 = x_data_25/x_data_25.max()
m_data_220_8192 = np.load("BQM_220_8192.npy")/np.load("BQM_220_8192.npy").max()
x_data_220_8192 = np.load("BQX_220_8192.npy")
x_data_220_8192[0] = 0.0
x_data_220_8192 = x_data_220_8192/x_data_220_8192.max()
m_data_220_1024 = np.load("BQM_220_1024.npy")/np.load("BQM_220_1024.npy").max()
x_data_220_1024 = np.load("BQX_220_1024.npy")
x_data_220_1024[0] = 0.0
x_data_220_1024 = x_data_220_1024/x_data_220_1024.max()

tm = 46.65*2.0
T = np.linspace(0.01,tm,21)
T2 = np.linspace(0.01,tm,11)
figure = plt.figure(figsize=(8,6))
ax = figure.add_subplot(111)
ax.plot(T_ref,M_ref,"k-",label="Ref")
ax.plot(T,m_data_625,"r-",label="6.25%")
ax.plot(T,m_data_25,"b-",label="25%")
ax.plot(T,m_data_220_8192,"g-",label="17%")
ax.plot(T2,m_data_220_1024,"c-",label="2%")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Normalized Magnetization")
ax.legend()
ax.set_xlim(0,tm)
figure.savefig("BQM_M.png",dpi=600,bbox_inches="tight",pad_inches=0.1,transparent=False)
plt.close()

figure = plt.figure(figsize=(8,6))
ax = figure.add_subplot(111)
ax.plot(T_ref,X_ref,"k-",label="Ref")
ax.plot(T,x_data_625,"r-",label="6.25%")
ax.plot(T,x_data_25,"b-",label="25%")
ax.plot(T,x_data_220_8192,"g-",label="17%")
ax.plot(T2,x_data_220_1024,"c-",label="2%")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Normalized Susceptibility")
ax.legend()
ax.set_xlim(0,tm)
figure.savefig("BQM_X.png",dpi=600,bbox_inches="tight",pad_inches=0.1,transparent=False)
plt.close()