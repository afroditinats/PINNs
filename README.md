Both files 1d burgers pep8 and navier stokes pep8 have been updated to follow PEP8 standards. Each .py file includes a docstring at the end that states all the changes that have been made. -Some ideas are still up for discussion to decide if they’re worthh adding.-

---

modules_esc / pinn_esc / all_esc_experiments => both used to run the pinn using various stopping criteria.

---

I ran the PINN for different values of each hyperparameter (of the NN architecture) and plotted the results. For now, I'm also playing around with Optuna for hyperparameter tuning. More results on changing the optimizers' learning rates will be added.

---

I gathered all the possible hyperparameters I had in mind, as well as those I could extract from the code. I'm not sure if you'd like to include the viscosity mentioned during the meeting as a hyperparameter -in the Burgers' problem it is fixed, but I assume that experimenting with various values might be interesting or impact the problem's performance(?)-.
