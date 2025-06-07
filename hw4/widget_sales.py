"""
Starter code for the problem "Widget sales".

Autonomous Systems Lab (ASL), Stanford University
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Seed RNG for reproducibility
seed = 0
rng = np.random.default_rng(seed)

# Define the state space, action space, and demand distribution
S = np.array([0, 1, 2, 3, 4, 5])
A = np.array([0, 2, 4])
D = np.array([0, 1, 2, 3, 4])
P = np.array([0.1, 0.3, 0.3, 0.2, 0.1])


def transition(s: int, a: int, d: int) -> int:
    """Compute the next state given the current state, action, and demand."""
    s_next = np.clip(s + a - d, 0, 5)
    return s_next


def reward(s: int, a: int, d: int) -> float:
    """Compute the reward given the current state, action, and demand."""
    price = 1.2
    cost_rent = 1.0
    cost_storage = 0.05 * s
    cost_order = np.sqrt(a)
    r = price * np.minimum(s + a, d) - cost_rent - cost_storage - cost_order
    return r


def simulate(
    rng: np.random.Generator,
    policy: callable,
    T: int,
    s0: int = 5,
    D: np.ndarray = D,
    P: np.ndarray = P,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate widget sales for a given policy."""
    s = np.zeros(T + 1)  # states
    a = np.zeros(T)  # actions
    r = np.zeros(T)  # rewards
    s[0] = s0  # initial state

    for t in tqdm(range(T)):
        # Sample demand
        d = rng.choice(D, p=P)

        # Record action, reward, and next state
        a[t] = policy(s[t])
        r[t] = reward(s[t], a[t], d)
        s[t + 1] = transition(s[t], a[t], d)

    return s, a, r


# Generate historical data with a uniformly random policy
log = {}
T = 3 * 365
log["s"], log["a"], log["r"] = simulate(rng, lambda s, A=A: rng.choice(A), T)


# Do Q-learning
γ = 0.95  # discount factor
α = 1e-2  # learning rate
num_epochs = 5 * int(1 / α)  # number of epochs

Q = np.zeros((S.size, A.size))
Q_epoch = np.zeros((num_epochs + 1, S.size, A.size))

for k in tqdm(range(1, num_epochs + 1)):
    # Shuffle transition tuple indices
    shuffled_indices = rng.permutation(T)

    # ####################### PART (a): YOUR CODE BELOW #######################

    # INSTRUCTIONS: Update `Q` using Q-learning.
    for idx in shuffled_indices:
        s_t = int(log["s"][idx])
        a_t = int(log["a"][idx])
        r_t = log["r"][idx]

        # Sample next state and action
        s_next = int(log["s"][idx + 1] if idx + 1 < T else s_t)
        a_idx = np.where(A == a_t)[0][0]  # Get index of action a_t

        # Update Q-value using the Q-learning update rule
        Q[s_t, a_idx] += α * (r_t + γ * np.max(Q[s_next]) - Q[s_t, a_idx])

    # ############################# END PART (a) ##############################

    # Record Q-values for this epoch
    Q_epoch[k] = Q


# Do value iteration
converged = False
eps = 1e-4
max_iters = 500
Q_vi = np.zeros((S.size, A.size))
Q_vi_prev = np.full(Q_vi.shape, np.inf)

for k in tqdm(range(max_iters)):

    # ####################### PART (b): YOUR CODE BELOW #######################

    # INSTRUCTIONS: Update `Q_vi` using value iteration.
    for s in range(S.size):
        for a in range(A.size):
            # Compute expected value for each action
            expected_value = 0.0
            for d, p in zip(D, P):
                s_next = transition(s, A[a], d)
                expected_value += p * (reward(s, A[a], d) + γ * np.max(Q_vi[s_next]))

            # Update Q-value for state-action pair
            Q_vi[s, a] = expected_value

    # ############################# END PART (b) ##############################

    if np.max(np.abs(Q_vi - Q_vi_prev)) < eps:
        converged = True
        print("Value iteration converged after {} iterations.".format(k))
        break
    else:
        np.copyto(Q_vi_prev, Q_vi)

if not converged:
    raise RuntimeError("Value iteration did not converge!")


# Plot Q-values for each epoch
fig, axes = plt.subplots(
    2, S.size // 2, figsize=(12, 6), sharex=True, sharey=True, dpi=150
)
fig.subplots_adjust(hspace=0.2)
for i, ax in enumerate(axes.ravel()):
    for j in range(A.size):
        plot = ax.plot(Q_epoch[:, i, j], label="$a = {}$".format(A[j]))
        ax.axhline(Q_vi[i, j], linestyle="--", color=plot[0].get_color())
        ax.legend(loc="lower right")
        ax.set_title(r"$s = {}$".format(S[i]))
for ax in axes[-1, :]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("$Q(s,a)$")
fig.savefig("widget_sales_qvalues.png", bbox_inches="tight")
plt.show()


# Report optimal policies from Q-learning and value iteration, simulate them
# for 5 years, and plot the cumulative profits

# ######################### PART (c): YOUR CODE BELOW #########################

# INSTRUCTIONS: Compute the optimal actions `a_opt_ql` and `a_opt_vi` using the
#               Q-values from Q-learning and value iteration, respectively.
#               Both `a_opt_ql` and `a_opt_vi` should be `np.ndarray`s, where
#               each entry is the optimal action for the corresponding state.
#
#               Also, simulate each optimal policy and compute the history of
#               cumulative profits `profit_ql` and `profit_vi` over 5 years
#               (at 365 days per year).

T = 5 * 365

# TODO: replace the next four lines with your code
a_opt_ql = A[np.argmax(Q, axis=1)]  # Optimal actions from Q-learning
s_ql, a_ql, r_ql = simulate(rng, lambda s: a_opt_ql[int(s)], T)  # Simulate Q-learning policy
profit_ql = np.cumsum(r_ql)  # Cumulative profit from Q-learning
a_opt_vi = A[np.argmax(Q_vi, axis=1)]  # Optimal actions from value iteration
s_vi, a_vi, r_vi = simulate(rng, lambda s: a_opt_vi[int(s)], T)  # Simulate value iteration policy
profit_vi = np.cumsum(r_vi)  # Cumulative profit from value iteration

# ############################### END PART (c) ################################

print("Optimal policy (Q-learning):     ", a_opt_ql)
print("Optimal policy (value iteration):", a_opt_vi)

fig, ax = plt.subplots()
ax.plot(profit_ql, label=r"$Q$-learning")
ax.plot(profit_vi, label=r"value iteration")
ax.legend(loc="lower right")
ax.set_xlabel(r"day $t$")
ax.set_ylabel(r"cumulative profit $\sum_{k=0}^t r_k$")
fig.savefig("widget_sales_profits.png", bbox_inches="tight")
plt.show()
