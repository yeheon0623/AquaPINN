# unconfined_homogeneous_isotropic.py
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine

from initialize_he import initialize_he
from initialize_zeros import initialize_zeros
from model import model
from model_2 import model_2
from model_gradients import model_gradients
from model_gradients_parameters2 import model_gradients_parameters2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定头和初始头设定
fixhead_west = 0.4
fixhead_east = 0.6
starting_head = 1.0

df = pd.read_excel("Observations_homogeneous_isotropic.xlsx", sheet_name="Foglio1")
X = df.iloc[:, 0].to_numpy().flatten()
Z = df.iloc[:, 2].to_numpy().flatten()
H0 = df.iloc[:, 7].to_numpy().flatten()
H25 = df.iloc[:, 8].to_numpy().flatten()
H50 = df.iloc[:, 9].to_numpy().flatten()
H1 = df.iloc[:, 10].to_numpy().flatten()

mask0 = H0 != -888
mask25 = H25 != -888
mask50 = H50 != -888
mask1 = H1 != -888
Xobs0, Xobs25, Xobs50, Xobs1 = X[mask0], X[mask25], X[mask50], X[mask1]
Zobs0, Zobs25, Zobs50, Zobs1 = Z[mask0], Z[mask25], Z[mask50], Z[mask1]
H0, H25, H50, H1 = H0[mask0], H25[mask25], H50[mask50], H1[mask1]

t0 = np.full(Xobs0.shape, 0.01)
t25 = np.full(Xobs25.shape, 0.25)
t50 = np.full(Xobs50.shape, 0.5)
t1 = np.full(Xobs1.shape, 1.0)

T_param = 0.001
S_param = 0.001

numBC_fixed = 1000
numBC_imp = 1000
x0BC1 = np.zeros(numBC_fixed)
z0BC1 = np.linspace(0, fixhead_west, numBC_fixed)
x0BC2 = np.ones(numBC_fixed)
z0BC2 = np.linspace(0, fixhead_east, numBC_fixed)
x0BC3 = np.linspace(0, 1, numBC_imp)
z0BC3 = np.zeros(numBC_imp)
x0BC4 = np.linspace(0, 1, numBC_imp)
z0BC4 = np.ones(numBC_imp)
x0BC5 = np.zeros(numBC_imp)
z0BC5 = np.linspace(fixhead_west, 1, numBC_imp)
x0BC6 = np.ones(numBC_imp)
z0BC6 = np.linspace(fixhead_east, 1, numBC_imp)
t0BC1 = np.linspace(0, 1, numBC_fixed)
t0BC2 = np.linspace(0, 1, numBC_fixed)
t0BC3 = np.linspace(0, 1, numBC_imp)
t0BC4 = np.linspace(0, 1, numBC_imp)
t0BC5 = np.linspace(0, 1, numBC_imp)
t0BC6 = np.linspace(0, 1, numBC_imp)
u0BC1 = fixhead_west * np.ones(numBC_fixed)
u0BC2 = fixhead_east * np.ones(numBC_fixed)

numIC = 500
x0IC = np.random.rand(numIC)
z0IC = np.random.rand(numIC)
t0IC = np.zeros(numIC)
u0IC = starting_head * np.ones(numIC)

X0 = np.concatenate([x0IC, x0BC1, x0BC2, Xobs0, Xobs25, Xobs50, Xobs1])
Z0 = np.concatenate([z0IC, z0BC1, z0BC2, Zobs0, Zobs25, Zobs50, Zobs1])
T0 = np.concatenate([t0IC, t0BC1, t0BC2, t0, t25, t50, t1])
U0 = np.concatenate([u0IC, u0BC1, u0BC2, H0, H25, H50, H1])

S0IC = np.vstack([x0IC, t0IC, u0IC])

dlX0 = torch.tensor(X0, dtype=torch.float32, device=device).unsqueeze(0)
dlZ0 = torch.tensor(Z0, dtype=torch.float32, device=device).unsqueeze(0)
dlT0 = torch.tensor(T0, dtype=torch.float32, device=device).unsqueeze(0)
dlH0 = torch.tensor(U0, dtype=torch.float32, device=device).unsqueeze(0)
dlX0BC3 = torch.tensor(x0BC3, dtype=torch.float32, device=device).unsqueeze(0)
dlX0BC4 = torch.tensor(x0BC4, dtype=torch.float32, device=device).unsqueeze(0)
dlX0BC5 = torch.tensor(x0BC5, dtype=torch.float32, device=device).unsqueeze(0)
dlX0BC6 = torch.tensor(x0BC6, dtype=torch.float32, device=device).unsqueeze(0)
dlZ0BC3 = torch.tensor(z0BC3, dtype=torch.float32, device=device).unsqueeze(0)
dlZ0BC4 = torch.tensor(z0BC4, dtype=torch.float32, device=device).unsqueeze(0)
dlZ0BC5 = torch.tensor(z0BC5, dtype=torch.float32, device=device).unsqueeze(0)
dlZ0BC6 = torch.tensor(z0BC6, dtype=torch.float32, device=device).unsqueeze(0)
dlT0BC3 = torch.tensor(t0BC3, dtype=torch.float32, device=device).unsqueeze(0)
dlT0BC4 = torch.tensor(t0BC4, dtype=torch.float32, device=device).unsqueeze(0)
dlT0BC5 = torch.tensor(t0BC5, dtype=torch.float32, device=device).unsqueeze(0)
dlT0BC6 = torch.tensor(t0BC6, dtype=torch.float32, device=device).unsqueeze(0)
S0IC_dl = torch.tensor(S0IC, dtype=torch.float32, device=device)

from initialize_he import initialize_he
from initialize_zeros import initialize_zeros

numLayers = 9
numNeurons = 20

def create_parameters_fc(input_dim, output_dim):
    return {
        'Weights': initialize_he((output_dim, input_dim)).to(device),
        'Bias': initialize_zeros((output_dim, 1)).to(device)
    }

parameters = {}
parameters['fc1'] = create_parameters_fc(3, numNeurons)
for i in range(2, numLayers):
    parameters[f'fc{i}'] = create_parameters_fc(numNeurons, numNeurons)
parameters[f'fc{numLayers}'] = create_parameters_fc(numNeurons, 1)

parameters2 = {}
parameters2['fc1'] = create_parameters_fc(2, numNeurons)
for i in range(2, numLayers):
    parameters2[f'fc{i}'] = create_parameters_fc(numNeurons, numNeurons)
parameters2[f'fc{numLayers}'] = create_parameters_fc(numNeurons, 1)

numEpochs = 200
initialLearnRate = 0.01
decayRate = 0.005

optimizer1 = optim.Adam([p for layer in parameters.values() for p in [layer['Weights'], layer['Bias']]], lr=initialLearnRate)
optimizer2 = optim.Adam([p for layer in parameters2.values() for p in [layer['Weights'], layer['Bias']]], lr=initialLearnRate)

loss_history1 = []
iteration = 0
for epoch in range(numEpochs - 150):
    optimizer1.zero_grad()
    gradients, loss, _ = model_gradients(parameters, parameters2, dlX0, dlZ0, dlT0,
                                           dlX0, dlZ0, dlT0, dlH0,
                                           S0IC_dl, dlX0BC3, dlX0BC4, dlX0BC5, dlX0BC6,
                                           dlZ0BC3, dlZ0BC4, dlZ0BC5, dlZ0BC6,
                                           dlT0BC3, dlT0BC4, dlT0BC5, dlT0BC6,
                                           torch.tensor(T_param, device=device),
                                           torch.tensor(S_param, device=device))
    loss.backward()
    optimizer1.step()
    iteration += 1
    loss_history1.append(loss.item())
    if iteration % 10 == 0:
        print(f"Phase 1 - Epoch {epoch}, Iteration {iteration}, Loss: {loss.item()}")

loss_history2 = []
iteration = 0
for epoch in range(numEpochs - 150):
    optimizer2.zero_grad()
    from model_gradients_parameters2 import model_gradients_parameters2
    gradients2, loss2, _ = model_gradients_parameters2(parameters, parameters2, dlX0, dlZ0, dlT0, S0IC_dl)
    loss2.backward()
    optimizer2.step()
    iteration += 1
    loss_history2.append(loss2.item())
    if iteration % 10 == 0:
        print(f"Phase 2 - Epoch {epoch}, Iteration {iteration}, Loss: {loss2.item()}")

optimizer_joint = optim.Adam([p for layer in parameters.values() for p in [layer['Weights'], layer['Bias']]] +
                             [p for layer in parameters2.values() for p in [layer['Weights'], layer['Bias']]], lr=initialLearnRate)
loss_history_joint = []
for epoch in range(numEpochs):
    optimizer_joint.zero_grad()
    gradients_joint, loss_joint, _ = model_gradients(parameters, parameters2, dlX0, dlZ0, dlT0,
                                                       dlX0, dlZ0, dlT0, dlH0,
                                                       S0IC_dl, dlX0BC3, dlX0BC4, dlX0BC5, dlX0BC6,
                                                       dlZ0BC3, dlZ0BC4, dlZ0BC5, dlZ0BC6,
                                                       dlT0BC3, dlT0BC4, dlT0BC5, dlT0BC6,
                                                       torch.tensor(T_param, device=device),
                                                       torch.tensor(S_param, device=device))
    loss_joint.backward()
    optimizer_joint.step()
    loss_history_joint.append(loss_joint.item())
    if epoch % 10 == 0:
        print(f"Phase Joint - Epoch {epoch}, Loss: {loss_joint.item()}")

plt.figure()
plt.plot(loss_history1, label='Loss parameters')
plt.plot(loss_history2, label='Loss parameters2')
plt.plot(loss_history_joint, label='Loss joint')
plt.xlabel("Iteration/Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss History")
plt.show()
