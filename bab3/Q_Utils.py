#Example 3.22 - Q_Utils.py
#Modified from:
#https://amunategui.github.io/reinforcement-learning/index.html
#http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarterwith-q-learning.html
#http://mnemstudio.org/path-finding-q-learning-tutorial.htm
import numpy as np
import pylab as plt


def showgraph(points_list):
	positions = {
		0: (0.55, 0.90),
		1: (0.48, 0.52),
		2: (0.72, 0.28),
		3: (0.25, 0.48),
		4: (0.92, 0.08),
		5: (0.08, 0.08),
		6: (0.02, 0.72),
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