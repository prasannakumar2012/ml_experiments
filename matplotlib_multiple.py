import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

# plt.figure(1)
plt.subplot(221)
plt.ylabel('1')
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(222)
plt.ylabel('2')
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

plt.subplot(223)
plt.ylabel('3')
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

plt.subplot(224)
plt.ylabel('4')
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')


plt.show()