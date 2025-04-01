import os
from modules_esc import EarlyStoppingCriterion, PatienceStoppingModel, GradientNormStoppingModel, HybridStoppingModel
from data import prepare_tensor, add_noise
import numpy as np
from sklearn.metrics import mean_squared_error 

#same as classic PINN | the stopping criterion changes
def PINN_earlystop_experiment(data, noise, verbose=True, rerun=False):
    if os.path.isfile("./results/pinn_results_esc.npy") and not rerun:
        print("Loaded Early Stopping PINN results.")
        return np.load("./results/pinn_results_esc.npy", allow_pickle=True)
    data = prepare_tensor(data)
    x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, _ = data
    rmse = []
    estimated_parameter = []
    parameter_error = []

    for noise_level in noise:
        y_train_noise, y_val_noise = add_noise([y_train, y_val], noise_level=noise_level)
        model = EarlyStoppingCriterion()
        model.train_model([x_bc, y_bc], [x_ic, y_ic], [x_train, y_train_noise],
                          [x_val, y_val_noise], pde_x, iterations=2000)

        pred = model.forward(x_test)
        viscosity = 10 ** model.visc.item()
        rmse_val = mean_squared_error(pred.detach(), y_test.reshape(-1, 1))
        param_error = mean_squared_error([viscosity], [0.01 / np.pi])
        rmse.append(rmse_val)
        estimated_parameter.append(viscosity)
        parameter_error.append(param_error)
        if verbose:
            print("Noise level:", noise_level)
            print("Estimated parameter:", viscosity)
            print("Test set RMSE:", rmse_val)
            print("Parameter error:", param_error)
    results = [rmse, estimated_parameter, parameter_error]
    np.save("./results/pinn_results_esc.npy", results)
    print("PINN early stopping test complete.")
    return results

def PINN_patience_experiment(data, noise, verbose=True, rerun=False):
    if os.path.isfile("./results/pinn_results_patience.npy") and not rerun:
        print("Loaded Patience-Stopping PINN results.")
        return np.load("./results/pinn_results_patience.npy", allow_pickle=True)
    data = prepare_tensor(data)
    x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, _ = data
    rmse = []
    estimated_parameter = []
    parameter_error = []

    for noise_level in noise:
        y_train_noise, y_val_noise = add_noise([y_train, y_val], noise_level=noise_level)
        model = PatienceStoppingModel()
        model.train_model([x_bc, y_bc], [x_ic, y_ic], [x_train, y_train_noise],
                          [x_val, y_val_noise], pde_x, iterations=2000)

        pred = model.forward(x_test)
        viscosity = 10 ** model.visc.item()
        rmse_val = mean_squared_error(pred.detach(), y_test.reshape(-1, 1))
        param_error = mean_squared_error([viscosity], [0.01 / np.pi])
        rmse.append(rmse_val)
        estimated_parameter.append(viscosity)
        parameter_error.append(param_error)

        if verbose:
            print("Noise level:", noise_level)
            print("Estimated parameter:", viscosity)
            print("Test set RMSE:", rmse_val)
            print("Parameter error:", param_error)
    results = [rmse, estimated_parameter, parameter_error]
    np.save("./results/pinn_results_patience.npy", results)
    print("PINN patience-based test complete.")
    return results

def PINN_gradnorm_experiment(data, noise, verbose=True):
    data = prepare_tensor(data)
    x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, _ = data
    y_train_noise, y_val_noise = add_noise([y_train, y_val], noise_level=noise[0])
    model = GradientNormStoppingModel()
    model.train_model([x_bc, y_bc], [x_ic, y_ic], [x_train, y_train_noise], [x_val, y_val_noise], pde_x, iterations=2000)
    pred = model.forward(x_test)
    viscosity = 10 ** model.visc.item()
    rmse_val = mean_squared_error(pred.detach(), y_test.reshape(-1, 1))
    param_error = mean_squared_error([viscosity], [0.01 / np.pi])

    if verbose:
        print("Estimated parameter:", viscosity)
        print("Test set RMSE:", rmse_val)
        print("Parameter error:", param_error)
    return [rmse_val], [viscosity], [param_error]

def PINN_hybrid_experiment(data, noise, verbose=True):
    data = prepare_tensor(data)
    x_test, y_test, x_train, y_train, x_bc, y_bc, x_ic, y_ic, x_val, y_val, pde_x, _ = data
    y_train_noise, y_val_noise = add_noise([y_train, y_val], noise_level=noise[0])
    model = HybridStoppingModel()
    model.train_model([x_bc, y_bc], [x_ic, y_ic], [x_train, y_train_noise], [x_val, y_val_noise], pde_x, iterations=2000)
    pred = model.forward(x_test)
    viscosity = 10 ** model.visc.item()
    rmse_val = mean_squared_error(pred.detach(), y_test.reshape(-1, 1))
    param_error = mean_squared_error([viscosity], [0.01 / np.pi])

    if verbose:
        print("Estimated parameter:", viscosity)
        print("Test set RMSE:", rmse_val)
        print("Parameter error:", param_error)
    return [rmse_val], [viscosity], [param_error]

print("ok pinn_esc")
