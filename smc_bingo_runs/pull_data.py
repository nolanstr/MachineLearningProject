import numpy as np
import sympy as sy
import glob
import matplotlib.pyplot as plt

from bingo.evolutionary_optimizers.evolutionary_optimizer import \
        load_evolutionary_optimizer_from_file as leoff


files = glob.glob('*.pkl')
pickles = [leoff(FILE) for FILE in files]

gen_ages = np.array([pickle.generational_age for pickle in pickles])

idxs = np.argsort(gen_ages)

gen_ages = gen_ages[idxs]
pickles = [pickles[i] for i in idxs]

for pickle in pickles:
    print(pickle.idx)
    idx = pickle._ea.evaluation.ff_idx
    for ind in pickle.hall_of_fame[pickle.idx]:
        eq_str = ind.get_formatted_string("console")
        X_0=sy.symbols('X_0')
        #ind_str = sy.nsimplify(eq_str.replace(")(",")*("),
        #                                    tolerance=1e-5, rational=True)
        #ind_str = sy.expand(ind_str)#eq_str.replace(")(", ")*("))

        ind_str = sy.expand(eq_str.replace(")(", ")*("))
        ind_str = sy.nsimplify(ind_str,tolerance=1e-5, rational=True)
        print(f"Complexity: {ind.get_complexity()}\nfit: {ind.fitness} \
                \neqn: {ind_str}")
import pdb;pdb.set_trace()
