import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

# x = jnp.load('x_test_prior.npy')
# samples = jnp.load('prior_samples.npy')

# # print(x_test)

# fig,ax = plt.subplots(1,1)

# for i in range(samples.shape[0]):
#     # x = x_test[i,:]
#     args = x.argsort(axis=0)
#     ax.plot(x.squeeze(-1)[args], samples[i,:].squeeze(-1)[args], '-', markersize=1, lw=1)
# plt.show()

x_test = jnp.load('x_test.npy')
y_test = jnp.load('y_test.npy')
args = x_test.argsort(axis=0)

fig,ax = plt.subplots(1,7, sharex=True, sharey=True)
ax[0].plot(x_test.squeeze(-1)[args], y_test.squeeze(-1)[args])

for i in range(6):
    c_pos = jnp.load(f'context_pos_{i}.npy')
    targ = jnp.delete(x_test, c_pos)
    # t_arg = jnp.delete(args, c_pos)
    t_arg = targ.argsort()
    cont = x_test[c_pos]
    ycont = y_test[c_pos]

    samples = jnp.load(f'cond_samples_{i}.npy')
    for j in range(samples.shape[0]):
        ax[i+1].plot(targ[t_arg], samples[j].squeeze(-1)[t_arg], '-o', markersize=0.5, lw=1, c='black', alpha=0.5)
        ax[i+1].scatter(cont, ycont, marker='x', c='red')
plt.show()