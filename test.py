from src.lindbladAG import evaluate, lindbladianAG, mesolveAG
import numpy as np
from qutip import Qobj, basis, brmesolve
import matplotlib.pyplot as plt

nLevels = 3

H = Qobj(np.diag([0, 1, 1.01]))
a_ops = [basis(nLevels, 1)*basis(nLevels, 0).dag() + 
         basis(nLevels, 0)*basis(nLevels, 1).dag(), 
         basis(nLevels, 2)*basis(nLevels, 0).dag() + 
         basis(nLevels, 0)*basis(nLevels, 2).dag()]

J = np.asarray([lambda x: 1 * (x>=0), lambda x: 0.02 * (x>=0)])
psi0 = (basis(nLevels, 1) + basis(nLevels, 2)).unit()
tlist = np.linspace(0, 50, 200)
e_ops = [basis(nLevels, i)*basis(nLevels, i).dag() for i in range(nLevels)]
popME = mesolveAG(H=H, psi0=psi0, tlist=tlist, a_ops=a_ops, J=J, 
                  use_secular=False, sec_cutoff=0.1, e_ops=e_ops, progress_bar=True).expect

a_opsBR = [[np.sqrt(2*np.pi)*a_op, j] for a_op, j in zip(a_ops, J)]
popBR = brmesolve(H=H, psi0=psi0, tlist=tlist, 
                  a_ops=a_opsBR,
                  use_secular=False, sec_cutoff=0.1, e_ops=e_ops).expect

fig = plt.figure()
for i in range(nLevels):
    plt.plot(tlist, popME[i])
    plt.plot(tlist, popBR[i], "--")

plt.savefig("test.png")