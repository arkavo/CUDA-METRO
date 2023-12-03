import numpy as np
import pandas as pd

T = [12.50,45.00]
for t in T:
    bl_a_256 = np.load(f"BL_Acceptance_256_{t:.2f}.npy")
    bl_a_512 = np.load(f"BL_Acceptance_512_{t:.2f}.npy")
    bl_a_1024 = np.load(f"BL_Acceptance_1024_{t:.2f}.npy")
    bl_a_2048 = np.load(f"BL_Acceptance_2048_{t:.2f}.npy")
    bl_a_4096 = np.load(f"BL_Acceptance_4096_{t:.2f}.npy")
    
    bl = pd.DataFrame(np.transpose(np.array([bl_a_256,bl_a_512,bl_a_1024,bl_a_2048,bl_a_4096])), columns=["256","512","1024","2048","4096"])
    bl.to_csv(f"csvs/BL_Acceptance_{t:.2f}.csv", index=False)

    bq_a_256 = np.load(f"BQ_Acceptance_256_{t:.2f}.npy")
    bq_a_512 = np.load(f"BQ_Acceptance_512_{t:.2f}.npy")
    bq_a_1024 = np.load(f"BQ_Acceptance_1024_{t:.2f}.npy")
    bq_a_2048 = np.load(f"BQ_Acceptance_2048_{t:.2f}.npy")
    bq_a_4096 = np.load(f"BQ_Acceptance_4096_{t:.2f}.npy")

    bq = pd.DataFrame(np.transpose(np.array([bq_a_256,bq_a_512,bq_a_1024,bq_a_2048,bq_a_4096])), columns=["256","512","1024","2048","4096"])
    bq.to_csv(f"csvs/BQ_Acceptance_{t:.2f}.csv", index=False)

    bl_c_256 = np.load(f"BL_Correlation_256_{t:.2f}.npy")
    bl_c_512 = np.load(f"BL_Correlation_512_{t:.2f}.npy")
    bl_c_1024 = np.load(f"BL_Correlation_1024_{t:.2f}.npy")
    bl_c_2048 = np.load(f"BL_Correlation_2048_{t:.2f}.npy")
    bl_c_4096 = np.load(f"BL_Correlation_4096_{t:.2f}.npy")

    bl = pd.DataFrame(np.transpose(np.array([bl_c_256,bl_c_512,bl_c_1024,bl_c_2048,bl_c_4096])), columns=["256","512","1024","2048","4096"])
    bl.to_csv(f"csvs/BL_Correlation_{t:.2f}.csv", index=False)

    bq_c_256 = np.load(f"BQ_Correlation_256_{t:.2f}.npy")
    bq_c_512 = np.load(f"BQ_Correlation_512_{t:.2f}.npy")
    bq_c_1024 = np.load(f"BQ_Correlation_1024_{t:.2f}.npy")
    bq_c_2048 = np.load(f"BQ_Correlation_2048_{t:.2f}.npy")
    bq_c_4096 = np.load(f"BQ_Correlation_4096_{t:.2f}.npy")

    bq = pd.DataFrame(np.transpose(np.array([bq_c_256,bq_c_512,bq_c_1024,bq_c_2048,bq_c_4096])), columns=["256","512","1024","2048","4096"])
    bq.to_csv(f"csvs/BQ_Correlation_{t:.2f}.csv", index=False)

    bl_e_256 = np.load(f"BL_Energy_256_{t:.2f}.npy")
    bl_e_512 = np.load(f"BL_Energy_512_{t:.2f}.npy")
    bl_e_1024 = np.load(f"BL_Energy_1024_{t:.2f}.npy")
    bl_e_2048 = np.load(f"BL_Energy_2048_{t:.2f}.npy")
    bl_e_4096 = np.load(f"BL_Energy_4096_{t:.2f}.npy")
    
    bl = pd.DataFrame(np.transpose(np.array([-bl_e_256,-bl_e_512,-bl_e_1024,-bl_e_2048,-bl_e_4096])), columns=["256","512","1024","2048","4096"])
    bl.to_csv(f"csvs/BL_Energy_{t:.2f}.csv", index=False)

    bl_de_256 = -np.gradient(bl_e_256)
    bl_de_512 = -np.gradient(bl_e_512)
    bl_de_1024 = -np.gradient(bl_e_1024)
    bl_de_2048 = -np.gradient(bl_e_2048)
    bl_de_4096 = -np.gradient(bl_e_4096)

    blde = pd.DataFrame(np.transpose(np.array([bl_de_256,bl_de_512,bl_de_1024,bl_de_2048,bl_de_4096])), columns=["256","512","1024","2048","4096"])
    blde.to_csv(f"csvs/BL_dEnergy_{t:.2f}.csv", index=False)

    bq_e_256 = np.load(f"BQ_Energy_256_{t:.2f}.npy")
    bq_e_512 = np.load(f"BQ_Energy_512_{t:.2f}.npy")
    bq_e_1024 = np.load(f"BQ_Energy_1024_{t:.2f}.npy")
    bq_e_2048 = np.load(f"BQ_Energy_2048_{t:.2f}.npy")
    bq_e_4096 = np.load(f"BQ_Energy_4096_{t:.2f}.npy")

    bq = pd.DataFrame(np.transpose(np.array([-bq_e_256,-bq_e_512,-bq_e_1024,-bq_e_2048,-bq_e_4096])), columns=["256","512","1024","2048","4096"])
    bq.to_csv(f"csvs/BQ_Energy_{t:.2f}.csv", index=False)

    bq_de_256 = -np.gradient(bq_e_256)
    bq_de_512 = -np.gradient(bq_e_512)
    bq_de_1024 = -np.gradient(bq_e_1024)
    bq_de_2048 = -np.gradient(bq_e_2048)
    bq_de_4096 = -np.gradient(bq_e_4096)

    bqde = pd.DataFrame(np.transpose(np.array([bq_de_256,bq_de_512,bq_de_1024,bq_de_2048,bq_de_4096])), columns=["256","512","1024","2048","4096"])
    bqde.to_csv(f"csvs/BQ_dEnergy_{t:.2f}.csv", index=False)


