# Modifikasi Example 3.22: Q-learning dengan 8-State Routing Diagram (State 0 hingga 7)
import numpy as np
import pylab as plt

# Fungsi visualisasi grafik yang disesuaikan untuk 8 state (0 sampai 7)
def showgraph_8states(points_list):
    positions = {
        0: (0.1, 0.5),
        1: (0.3, 0.8),
        2: (0.5, 0.8),
        3: (0.7, 0.6),
        4: (0.3, 0.2),
        5: (0.5, 0.2),
        6: (0.7, 0.4),
        7: (0.9, 0.5),
    }
    plt.figure()
    for start, end in points_list:
        start_x, start_y = positions[start]
        end_x, end_y = positions[end]
        plt.plot([start_x, end_x], [start_y, end_y], color="0.25", linewidth=1)
    for node, (x, y) in positions.items():
        plt.scatter(x, y, s=260, color="#1479b8", edgecolors="white", linewidths=1)
        plt.text(x, y, str(node), color="white", ha="center", va="center")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.02, 1.02)
    plt.axis("off")

def createRmat(matrix_size, points_list, goal):
    rewards = np.matrix(np.ones(shape=(matrix_size, matrix_size)))
    rewards *= -1
    for point in points_list:
        rewards[point] = 100 if point[1] == goal else 0
        reverse_point = point[::-1]
        rewards[reverse_point] = 100 if point[0] == goal else 0
    rewards[goal, goal] = 100
    return rewards

def available_actions(rewards, state):
    return np.where(rewards[state,] >= 0)[1]

def sample_next_action(available_act):
    return int(np.random.choice(available_act))

def update(rewards, q_values, current_state, action, gamma):
    max_index = np.where(q_values[action,] == np.max(q_values[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index))
    else:
        max_index = int(max_index.flat[0])
    max_value = q_values[action, max_index]
    q_values[current_state, action] = (
        rewards[current_state, action] + gamma * max_value
    )
    if np.max(q_values) > 0:
        return np.sum(q_values / np.max(q_values) * 100)
    return 0

# Setting up Parameters =========================================
# Daftar koneksi antar node membentuk 8 state dengan tujuan di state 7
points_list = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7), (3, 7)]
goal = 7
showgraph_8states(points_list)

# Ukuran matriks diperbesar menjadi 8x8 untuk menampung state 0 sampai 7
MATRIX_SIZE = 8
R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
gamma = 0.8

# Training ======================================================
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    if available_act.size == 0:
        continue
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# Testing =======================================================
current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(
        Q[current_state,] == np.max(Q[current_state,])
    )[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index))
    else:
        next_step_index = int(next_step_index.flat[0])
    steps.append(next_step_index)
    current_state = next_step_index

# Display Results ===============================================
print("Most efficient path:")
print(steps)
plt.figure()
plt.plot(scores)
plt.show()