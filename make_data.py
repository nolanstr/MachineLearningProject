import numpy as np

x = np.random.uniform(0, 2 * np.pi, 50).reshape(-1,1)

y = np.sin(5 * x) + np.random.normal(0,1,x.shape)

data = np.hstack((x,y))

np.save('noisy_data', data)
