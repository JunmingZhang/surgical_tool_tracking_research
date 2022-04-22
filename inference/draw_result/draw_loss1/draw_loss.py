import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

resnet_df = pd.read_csv("./learning_stats_resnet.csv")
mobilenet_df = pd.read_csv("./learning_stats_mobilenet.csv")
efficientnet_df = pd.read_csv("./learning_stats_efficientnet.csv")

resnet_loss_stats = resnet_df[resnet_df.columns[1]]
mobilenet_loss_stats = mobilenet_df[mobilenet_df.columns[1]]
efficientnet_loss_stats = efficientnet_df[efficientnet_df.columns[1]]

x = resnet_df[resnet_df.columns[0]]

plt.figure()
plt.plot(x, resnet_loss_stats, 'b', label='resnet')
plt.plot(x, mobilenet_loss_stats, 'g', label='mobilenet', alpha=0.7)
plt.plot(x, efficientnet_loss_stats, 'r', label='efficientnet', alpha=0.7)

plt.title("loss stats in each step")
plt.xlabel("iteration steps")
plt.ylabel("loss")
plt.xticks(np.arange(0, 110000, 10000), rotation=30)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
# plt.show()

resnet_alpha = resnet_df[resnet_df.columns[2]]
mobilenet_alpha = mobilenet_df[mobilenet_df.columns[2]]
efficientnet_alpha = efficientnet_df[efficientnet_df.columns[2]]

plt.figure()
plt.plot(x, resnet_alpha, 'b', label='resnet')
plt.plot(x, mobilenet_alpha, 'm--', label='mobilenet', alpha=0.7)
plt.plot(x, efficientnet_alpha, 'r', label='efficientnet', alpha=0.8)

plt.title("learning rates in each step")
plt.xlabel("iteration steps")
plt.ylabel("learning rate")
plt.legend()
# plt.show()

resnet_likelihood = pd.read_csv("./lily_1000DLC_resnet50_surgical_tool_trackingMar23shuffle1_100000.csv").iloc[2:]
resnet_likelihood_arm0 = resnet_likelihood[resnet_likelihood.columns[3]].astype(float)
resnet_likelihood_arm1 = resnet_likelihood[resnet_likelihood.columns[6]].astype(float)
resnet_likelihood_joint = resnet_likelihood[resnet_likelihood.columns[9]].astype(float)

resnet_select_indices_arm0 = list(np.where(resnet_likelihood_arm0 < 0.6)[0])
resnet_select_indices_arm1 = list(np.where(resnet_likelihood_arm1 < 0.6)[0])
resnet_select_indices_joint = list(np.where(resnet_likelihood_joint < 0.6)[0])
# print(resnet_select_indices_arm0)
# print(resnet_select_indices_arm1)
# print(resnet_select_indices_joint)

mobilenet_likelihood = pd.read_csv("./lily_1000DLC_mobnet_100_surgical_tool_trackingMar23shuffle1_100000.csv").iloc[2:]
mobilenet_likelihood_arm0 = mobilenet_likelihood[mobilenet_likelihood.columns[3]].astype(float)
mobilenet_likelihood_arm1 = mobilenet_likelihood[mobilenet_likelihood.columns[6]].astype(float)
mobilenet_likelihood_joint = mobilenet_likelihood[mobilenet_likelihood.columns[9]].astype(float)

mobilenet_select_indices_arm0 = list(np.where(mobilenet_likelihood_arm0 < 0.6)[0])
mobilenet_select_indices_arm1 = list(np.where(mobilenet_likelihood_arm1 < 0.6)[0])
mobilenet_select_indices_joint = list(np.where(mobilenet_likelihood_joint < 0.6)[0])
# print(mobilenet_select_indices_arm0)
# print(mobilenet_select_indices_arm1)
# print(mobilenet_select_indices_joint)

mobilenet2_df = pd.read_csv("./learning_stats_mobilenet2.csv")
x = mobilenet2_df[mobilenet2_df.columns[0]]
mobilenet2_loss_stats = mobilenet2_df[mobilenet2_df.columns[1]]

plt.figure()
plt.plot(x, mobilenet2_loss_stats, 'b', label='mobilenet')
plt.title("loss in each step of mobile net")
plt.xlabel("iteration steps")
plt.ylabel("loss")
plt.legend()
#plt.show()

mobilenet2_likelihood = pd.read_csv("./sim_real_1000DLC_mobnet_100_surgical_tool_trackingApr1shuffle1_100000.csv").iloc[2:]
mobilenet2_likelihood_keypoint0 = mobilenet2_likelihood[mobilenet2_likelihood.columns[3]].astype(float)
mobilenet2_likelihood_keypoint1 = mobilenet2_likelihood[mobilenet2_likelihood.columns[6]].astype(float)
mobilenet2_likelihood_keypoint2 = mobilenet2_likelihood[mobilenet2_likelihood.columns[9]].astype(float)
mobilenet2_likelihood_keypoint3 = mobilenet2_likelihood[mobilenet2_likelihood.columns[12]].astype(float)
mobilenet2_likelihood_keypoint4 = mobilenet2_likelihood[mobilenet2_likelihood.columns[15]].astype(float)
mobilenet2_likelihood_keypoint5 = mobilenet2_likelihood[mobilenet2_likelihood.columns[18]].astype(float)
mobilenet2_likelihood_keypoint6 = mobilenet2_likelihood[mobilenet2_likelihood.columns[21]].astype(float)
mobilenet2_likelihood_keypoint7 = mobilenet2_likelihood[mobilenet2_likelihood.columns[24]].astype(float)
mobilenet2_likelihood_keypoint8 = mobilenet2_likelihood[mobilenet2_likelihood.columns[27]].astype(float)
mobilenet2_likelihood_keypoint9 = mobilenet2_likelihood[mobilenet2_likelihood.columns[30]].astype(float)

mobilenet2_select_indices_keypoint0 = list(np.where(mobilenet2_likelihood_keypoint0 < 0.6)[0])
mobilenet2_select_indices_keypoint1 = list(np.where(mobilenet2_likelihood_keypoint1 < 0.6)[0])
mobilenet2_select_indices_keypoint2 = list(np.where(mobilenet2_likelihood_keypoint2 < 0.6)[0])
mobilenet2_select_indices_keypoint3 = list(np.where(mobilenet2_likelihood_keypoint3 < 0.6)[0])
mobilenet2_select_indices_keypoint4 = list(np.where(mobilenet2_likelihood_keypoint4 < 0.6)[0])
mobilenet2_select_indices_keypoint5 = list(np.where(mobilenet2_likelihood_keypoint5 < 0.6)[0])
mobilenet2_select_indices_keypoint6 = list(np.where(mobilenet2_likelihood_keypoint6 < 0.6)[0])
mobilenet2_select_indices_keypoint7 = list(np.where(mobilenet2_likelihood_keypoint7 < 0.6)[0])
mobilenet2_select_indices_keypoint8 = list(np.where(mobilenet2_likelihood_keypoint8 < 0.6)[0])
mobilenet2_select_indices_keypoint9 = list(np.where(mobilenet2_likelihood_keypoint9 < 0.6)[0])

# print(mobilenet2_select_indices_keypoint0, len(mobilenet2_select_indices_keypoint0))
# print(mobilenet2_select_indices_keypoint1, len(mobilenet2_select_indices_keypoint1))
# print(mobilenet2_select_indices_keypoint2, len(mobilenet2_select_indices_keypoint2))
# print(mobilenet2_select_indices_keypoint3, len(mobilenet2_select_indices_keypoint3))
# print(mobilenet2_select_indices_keypoint4, len(mobilenet2_select_indices_keypoint4))
# print(mobilenet2_select_indices_keypoint5, len(mobilenet2_select_indices_keypoint5))
# print(mobilenet2_select_indices_keypoint6, len(mobilenet2_select_indices_keypoint6))
# print(mobilenet2_select_indices_keypoint7, len(mobilenet2_select_indices_keypoint7))
# print(mobilenet2_select_indices_keypoint8, len(mobilenet2_select_indices_keypoint8))
# print(mobilenet2_select_indices_keypoint9, len(mobilenet2_select_indices_keypoint9))

mobilenet2_select_indices_keypoints = [mobilenet2_select_indices_keypoint0, mobilenet2_select_indices_keypoint1, \
    mobilenet2_select_indices_keypoint2, mobilenet2_select_indices_keypoint3, \
    mobilenet2_select_indices_keypoint4, mobilenet2_select_indices_keypoint5, \
    mobilenet2_select_indices_keypoint6, mobilenet2_select_indices_keypoint7, \
    mobilenet2_select_indices_keypoint8, mobilenet2_select_indices_keypoint9]

mobilenet_keypoint_counts = {}
for keypoint in mobilenet2_select_indices_keypoints:
    print(len(keypoint))
    for i in range(len(keypoint)):
        if keypoint[i] not in mobilenet_keypoint_counts:
            mobilenet_keypoint_counts[keypoint[i]] = 1
        else:
            mobilenet_keypoint_counts[keypoint[i]] += 1

mobilenet_keypoint_counts = dict(sorted(mobilenet_keypoint_counts.items(), key=lambda x: x[1], reverse=True))
print(mobilenet_keypoint_counts)

print(900 in mobilenet_keypoint_counts)
print(896 in mobilenet_keypoint_counts)
print(898 in mobilenet_keypoint_counts)
print(894 in mobilenet_keypoint_counts)
print(0 in mobilenet_keypoint_counts)
print(2 in mobilenet_keypoint_counts)
print(3 in mobilenet_keypoint_counts)
print(4 in mobilenet_keypoint_counts)
