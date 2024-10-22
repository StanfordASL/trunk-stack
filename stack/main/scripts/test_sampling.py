import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np


x = np.linspace(0, 1, 1000)
a = b = 0.95
pdf = beta.pdf(x, a, b)

plt.plot(x, pdf)
plt.title('Beta Distribution PDF with α=β=0.5 (U-shaped)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.show()
