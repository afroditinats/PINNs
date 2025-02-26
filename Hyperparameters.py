import json
import rich
"""
----------------------------------------------
Hyperparameters for PINN-Burger's eq. problem.
----------------------------------------------
"""

#data.py 

hyp_data={"noise_level": ("In add_noise(data, noise_level=0): "), 
          "training_data_indices": ("In create_training_data(x_test, y_test): "),
          "bc": 80,    #boundary conditions
          "ic": 160,   #initila conditions 
          "pde": 2540} #collocation points     

"""
def create_bc_data():

    x_bc, y_bc = [], []

    for i in range(0, 80):
        x_bc.append(np.array([np.random.choice([-1, 1]), np.random.uniform(0, 1)]))
        y_bc.append(np.array([0]))

    return x_bc, y_bc

def create_ic_data():

    x_ic, y_ic = [], []

    for i in range(0, 160):
        x = np.random.uniform(-1, 1)
        x_ic.append(np.array([x, 0]))
        y_ic.append(np.array([-1 * np.sin(x*np.pi)]))

    return x_ic, y_ic

def create_pde_data():

    pde_x = []

    for i in range(0, 2540):
        pde_x.append(np.array([np.random.uniform(-1, 1), np.random.uniform(0, 1)]))

    return pde_x
"""

#main.py

hyp_main={"noise_levels": [0, 0.5, 1, 2, 3, 5, 7, 10, 25]}

"""
data = get_data()
noise = [0, 0.5, 1, 2, 3, 5, 7, 10, 25]
results = []

results.append(PINN_experiment(data, noise))
"""

#modules.py

hyp_modules={"number of hiden layers":4,
             "number of neurons per hiden layer":20,
             "activation function":"tanh",
             "viscosity":-1.0,  #to be discussed(?) Affects the starting point(?)
             "optimizer":{"LBFGS":{"learning_rate":1, #Limited Memory BFGS
                                   "max_iter": 2000,
                                   "tolerance_grad": 1e-128,
                                   "tolerance_change": 1e-128,
                                   "history_size": 50,
                                   "line_search_fn": "strong_wolfe" #the only one used for LBFGS opt. 
                                   }}}

"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = self.create_network()
        self.bc = None
        self.ic = None
        self.cc = None
        self.ev = None
        self.pde = None
        self.visc = nn.Parameter(data=torch.tensor([-1.0]))
        self.network.register_parameter("visc", self.visc)
        self.adam_optimizer = optim.AdamW(self.network.parameters())
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.network.parameters(),
            lr=1,
            max_iter=2000,
            max_eval=None,
            tolerance_grad=1e-128,
            tolerance_change=1e-128,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.val_loss = None

    def create_network(self):

        network = []

        network.append(nn.Linear(2, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 20).double())
        network.append(nn.Tanh().double())
        network.append(nn.Linear(20, 1).double())

        return nn.Sequential(*network)
"""

#pinn.py

hyp_pinn={"Train model iteractions":2000}

"""
#Train model
        PINN = Model()
        PINN.train_model([x_bc, y_bc], [x_ic, y_ic], [x_train, y_train_noise], [x_val, y_val_noise], pde_x, iterations=2000)
        viscosity = 10**PINN.visc.item()
"""

print("data.py:")
print(json.dumps(hyp_data))
print("\n main.py:")
print(json.dumps(hyp_main))
print("\n modules.py:")
print(json.dumps(hyp_modules))
print("\n pinn.py")
print(json.dumps(hyp_pinn))
