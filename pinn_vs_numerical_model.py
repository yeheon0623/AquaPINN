# pinn_vs_numerical_model.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from model import model
from model_2 import model_2

mat_data = scipy.io.loadmat('results_lower_observations(90)_200epochs_25000CP.mat')
parameters = mat_data['parameters']
parameters2 = mat_data['parameters2']

df = pd.read_excel("Observations_homogeneous_isotropic.xlsx", sheet_name="Foglio1")
X = df.iloc[:, 0].to_numpy()
Z = df.iloc[:, 2].to_numpy()
H0 = df.iloc[:, 7].to_numpy().reshape(20,20).T
H25 = df.iloc[:, 8].to_numpy().reshape(20,20).T
H50 = df.iloc[:, 9].to_numpy().reshape(20,20).T
H1 = df.iloc[:, 10].to_numpy().reshape(20,20).T

discr = 200
XTest = np.linspace(0, 1, discr)
ZTest = np.linspace(0, 1, discr)
TTest00 = np.zeros(discr)
TTest0 = np.full(discr, 0.01)
TTest025 = np.full(discr, 0.25)
TTest05 = np.full(discr, 0.5)
TTest1 = np.full(discr, 1.0)

dlXTest = torch.tensor(XTest, dtype=torch.float32).unsqueeze(0)
dlZTest = torch.tensor(ZTest, dtype=torch.float32).unsqueeze(0)
dlTTest00 = torch.tensor(TTest00, dtype=torch.float32).unsqueeze(0)
dlTTest0 = torch.tensor(TTest0, dtype=torch.float32).unsqueeze(0)
dlTTest025 = torch.tensor(TTest025, dtype=torch.float32).unsqueeze(0)
dlTTest05 = torch.tensor(TTest05, dtype=torch.float32).unsqueeze(0)
dlTTest1 = torch.tensor(TTest1, dtype=torch.float32).unsqueeze(0)

Xgrid = np.repeat(XTest[:, np.newaxis], discr, axis=1)
Z_grid = np.tile(np.linspace(0,1,discr), (discr,1))
Xgrid_flat = Xgrid.flatten()[np.newaxis, :]
Z_grid_flat = Z_grid.flatten()[np.newaxis, :]

Tgrid0 = np.full(Xgrid_flat.shape, 0.01)
Tgrid025 = np.full(Xgrid_flat.shape, 0.25)
Tgrid05 = np.full(Xgrid_flat.shape, 0.5)
Tgrid1 = np.full(Xgrid_flat.shape, 1.0)

dlXgrid = torch.tensor(Xgrid_flat, dtype=torch.float32)
dlZgrid = torch.tensor(Z_grid_flat, dtype=torch.float32)
dlTgrid0 = torch.tensor(Tgrid0, dtype=torch.float32)
dlTgrid025 = torch.tensor(Tgrid025, dtype=torch.float32)
dlTgrid05 = torch.tensor(Tgrid05, dtype=torch.float32)
dlTgrid1 = torch.tensor(Tgrid1, dtype=torch.float32)

dlHPred0 = model(parameters, dlXgrid, dlZgrid, dlTgrid0)
dlHPred025 = model(parameters, dlXgrid, dlZgrid, dlTgrid025)
dlHPred05 = model(parameters, dlXgrid, dlZgrid, dlTgrid05)
dlHPred1 = model(parameters, dlXgrid, dlZgrid, dlTgrid1)

HPred0 = dlHPred0.detach().numpy().flatten()
HPred025 = dlHPred025.detach().numpy().flatten()
HPred05 = dlHPred05.detach().numpy().flatten()
HPred1 = dlHPred1.detach().numpy().flatten()

dlS0 = model_2(parameters2, dlXTest, dlTTest00)
dlS025 = model_2(parameters2, dlXTest, dlTTest025)
dlS05 = model_2(parameters2, dlXTest, dlTTest05)
dlS1 = model_2(parameters2, dlXTest, dlTTest1)

S0 = dlS0.detach().numpy().flatten()
S025 = dlS025.detach().numpy().flatten()
S05 = dlS05.detach().numpy().flatten()
S1 = dlS1.detach().numpy().flatten()

HPred0_map = np.flip(np.reshape(HPred0, (discr, discr)), axis=0)
HPred025_map = np.flip(np.reshape(HPred025, (discr, discr)), axis=0)
HPred05_map = np.flip(np.reshape(HPred05, (discr, discr)), axis=0)
HPred1_map = np.flip(np.reshape(HPred1, (discr, discr)), axis=0)

X_estimation0 = np.flip(np.reshape(Xgrid_flat, (discr, discr)), axis=0)
Z_estimation0 = np.flip(np.reshape(Z_grid_flat, (discr, discr)), axis=0)

idx_close0 = (discr - np.round(S0 * discr)).astype(int)
idx_close025 = (discr - np.round(S025 * discr)).astype(int)
idx_close05 = (discr - np.round(S05 * discr)).astype(int)
idx_close1 = (discr - np.round(S1 * discr)).astype(int)
for i in range(discr):
    HPred0_map[:idx_close0[i], i] = np.nan
    HPred025_map[:idx_close025[i], i] = np.nan
    HPred05_map[:idx_close05[i], i] = np.nan
    HPred1_map[:idx_close1[i], i] = np.nan

dpi = 300
axes_ticks = np.round(np.linspace(0, discr, num=11)).astype(int)
x_tick_labels = [''] + [f'{x/100:.1f}' for x in np.linspace(0.1, 1, 10)]
y_tick_labels = [str(i) for i in np.linspace(1, 0, 11)]

def plot_and_save(image, clim, title, filename):
    plt.figure()
    cmap = plt.get_cmap('jet')
    cmap.set_under(color='white')
    im = plt.imshow(image, cmap=cmap, vmin=clim[0], vmax=clim[1])
    plt.title(title)
    plt.colorbar(im)
    plt.xticks(axes_ticks, x_tick_labels)
    plt.yticks(axes_ticks, y_tick_labels)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, format='tiff')
    plt.close()

plot_and_save(HPred0_map, (0.3, 1), 'PINN t=0.01', 'piezoPINN_t0_eter.tif')
plot_and_save(HPred025_map, (0.37, 0.6), 'PINN t=0.25', 'piezoPINN_t25_eter.tif')
plot_and_save(HPred05_map, (0.37, 0.6), 'PINN t=0.50', 'piezoPINN_t50_eter.tif')
plot_and_save(HPred1_map, (0.37, 0.6), 'PINN t=1', 'piezoPINN_t1_eter.tif')

plot_and_save(H0, (0.395, 1), 'MODFLOW t=0.01', 'piezoGMS_t0_eter.tif')
plot_and_save(H25, (0.395, 0.6), 'MODFLOW t=0.25', 'piezoGMS_t25_eter.tif')
plot_and_save(H50, (0.395, 0.6), 'MODFLOW t=0.50', 'piezoGMS_t50_eter.tif')
plot_and_save(H1, (0.395, 0.6), 'MODFLOW t=1', 'piezoGMS_t1_eter.tif')


