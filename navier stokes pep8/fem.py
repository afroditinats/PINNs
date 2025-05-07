import time
import numpy as np
from fenics import *
from tqdm import *
import matplotlib.pyplot as plt


set_log_level(1000)
nx, ny = 64, 64
L = 2 * np.pi
dt = 0.01
max_step = 250
time_scale = 100
single_length = 3969 # data points for each time slice

class PeriodicBoundary(SubDomain):
    """Periodic boundary condition."""


    def inside(self, x, on_boundary):
        return (near(x[0], L) or near(x[1], L)) and on_boundary

    def map(self, x, y):
        if near(x[0], 0) and near(x[1], 0):
            y[0], y[1] = L, L
        elif near(x[0], 0):
            y[0], y[1] = L, x[1]
        elif near(x[1], 0):
            y[0], y[1] = x[0], L 
        else:
            y[0], y[1] = x[0], x[1]
             

def calculate_normalizer(p_next):
    """Normalize pressure field"""

    space_integral = assemble(p_next * dx)
    mean_pressure = space_integral / L**2
    return mean_pressure


def tgv_vortex(visc, slsqp=[], pinn=[]):
    """Inspired by:
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/fenics/lid_driven_cavity.py.
    
    Taylor-Green Vortex using Chorin's projection FEM formulations.
    
    """
    
    # Viscosity and start recording time
    print("Viscosity:", visc[0])
    start_time = time.time()

    visc = Constant(visc[0])

    # Mesh and space
    mesh = RectangleMesh(Point(0, 0), Point(L, L), nx, ny)
    pbc = PeriodicBoundary()

    velocity = VectorFunctionSpace(mesh, "Lagrange", 3, constrained_domain=pbc)
    pressure = FunctionSpace(mesh, "Lagrange", 2, constrained_domain=pbc)

    # Create velocity and pressure functions
    u_trial = TrialFunction(velocity)
    v_test = TestFunction(velocity)

    u_prev = Function(velocity)
    u_init = Expression(("sin(x[0]) * cos(x[1])",
                        "-cos(x[0]) * sin(x[1])"), degree=3)
    u_prev.assign(u_init)
    u_tent = Function(velocity)
    u_next = Function(velocity)

    p_trial = TrialFunction(pressure)
    q_test = TestFunction(pressure)
    p_next = Function(pressure)
    p_init = Expression(("0.25 * (cos(2*x[0]) + cos(2*x[1]))"), degree=3)
    p_next.interpolate(p_init)
    
    momentum = (
        ((1/dt) * inner(u_trial - u_prev, v_test) * dx) 
        + (inner(grad(u_prev) * u_prev, v_test) * dx) 
        + (visc * inner(grad(u_trial), grad(v_test)) * dx)
    )

    pressure = (
        (inner(grad(p_trial), grad(q_test)) * dx) 
        + ((1/dt) * div(u_tent)*q_test*dx)
    )

    velocity = (
        (inner(u_trial, v_test) * dx) 
        - ((inner(u_tent,v_test) * dx) 
        - (dt*inner(grad(p_next), v_test) * dx))
    )

    lhs_momentum = lhs(momentum)
    rhs_momentum = rhs(momentum)

    lhs_pressure = lhs(pressure)
    rhs_pressure = rhs(pressure)

    lhs_velocity = lhs(velocity)
    rhs_velocity = rhs(velocity)

    # For recording predictions
    predictions = []

    solver_parameters = {
    'linear_solver': 'gmres',
    'preconditioner': 'amg',
    'krylov_solver': {
        'absolute_tolerance': 1e-12,
        'relative_tolerance': 1e-10,
        'maximum_iterations': 5000,
    }
    }

    for t in tqdm(range(max_step)):

        solve(
            lhs_momentum == rhs_momentum,
            u_tent,
            solver_parameters=solver_parameters
        )
        
        solve(
            lhs_pressure == rhs_pressure,
            p_next,
            solver_parameters=solver_parameters
        )
        
        solve(
            lhs_velocity == rhs_velocity,
            u_next,
            solver_parameters=solver_parameters
        )

        mean_pressure = calculate_normalizer(p_next)

        # Record relevant time steps 
        if len(slsqp) > 0:
            # For FEM/SLSQP
            for point in slsqp[np.isclose(slsqp[:, 2] * time_scale, t+1), 0:2]:
                x_vel, y_vel = u_next(point[0], point[1])
                pressure = p_next(point[0], point[1]) - mean_pressure
                predictions.append([x_vel, y_vel, pressure])
        elif len(pinn) > 0:
            # For FEM/PINN
            for point in pinn[np.isclose(pinn[:, 2].detach().cpu().numpy() * time_scale, t+1), 0:2]:
                x_vel, y_vel = u_next(point[0], point[1])
                pressure = p_next(point[0], point[1]) - mean_pressure
                predictions.append([x_vel, y_vel, pressure])

        u_prev.assign(u_next)

    del mesh # Memory leak debugging
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return predictions


"""
1. libraries

2.     def map(self, x, y):

        if near(x[0], 0) and near(x[1], 0):
            y[0] = L
            y[1] = L
        elif near(x[0], 0):
            y[0] = L
            y[1] = x[1]
        elif near(x[1], 0):
            y[0] = x[0]
            y[1] = L
        else:
            y[0] = x[0]
            y[1] = x[1]
-->  def map(self, x, y):
        if near(x[0], 0) and near(x[1], 0):
            y[0], y[1] = L, L
        elif near(x[0], 0):
            y[0], y[1] = L, x[1]
        elif near(x[1], 0):
            y[0], y[1] = x[0], L 
        else:
            y[0], y[1] = x[0], x[1]
             
3. mean_pressure = space_integral / (L)**2 --> mean_pressure = space_integral / L**2

4. spaces

5. if len(slsqp) > 0:
            # For FEM/SLSQP
            mean_pressure = calculate_normalizer(p_next)
            for point in slsqp[np.isclose(slsqp[:, 2] * 100, float(t+1)), 0:2]:
--> time_scale =100 then:
    for point in slsqp[np.isclose(slsqp[:, 2] * time_scale, float(t+1)), 0:2]: 

6. if len(slsqp) > 0:
            # For FEM/SLSQP
            mean_pressure = calculate_normalizer(p_next)
            for point in slsqp[np.isclose(slsqp[:, 2] * time_scale, float(t+1)), 0:2]:
                x_vel, y_vel = u_next(point[0], point[1])
                pressure = p_next(point[0], point[1]) - mean_pressure
                predictions.append([x_vel, y_vel, pressure])
        elif len(pinn) > 0:
            # For FEM/PINN
            mean_pressure = calculate_normalizer(p_next)
            for point in pinn[np.isclose(pinn[:, 2].detach().cpu().numpy() * time_scale, float(t+1)), 0:2]:
                x_vel, y_vel = u_next(point[0], point[1])
                pressure = p_next(point[0], point[1]) - mean_pressure
                predictions.append([x_vel, y_vel, pressure])
--> mean_pressure = calculate_normalizer(p_next)

        # Record relevant time steps
        if len(slsqp) > 0:
            # For FEM/SLSQP
            for point in slsqp[np.isclose(slsqp[:, 2] * time_scale, float(t+1)), 0:2]:
                x_vel, y_vel = u_next(point[0], point[1])
                pressure = p_next(point[0], point[1]) - mean_pressure
                predictions.append([x_vel, y_vel, pressure])
        elif len(pinn) > 0:
            # For FEM/PINN
            for point in pinn[np.isclose(pinn[:, 2].detach().cpu().numpy() * time_scale, float(t+1)), 0:2]:
                x_vel, y_vel = u_next(point[0], point[1])
                pressure = p_next(point[0], point[1]) - mean_pressure
                predictions.append([x_vel, y_vel, pressure])

        u_prev.assign(u_next)
!!! renoved the float(t+1) because t is already float because : for t in tqdm(range(max_step)):....
7. 
"""

print('ok')