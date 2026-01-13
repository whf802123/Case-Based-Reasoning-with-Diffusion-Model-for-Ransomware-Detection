import os
import cv2


output_dir = "Grey_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rgb_img = cv2.cvtColor(cv2.imread("D://Data/Benign_images_16/FTP_0.png"), cv2.COLOR_BGR2RGB)

R_channel = rgb_img[:, :, 0]
G_channel = rgb_img[:, :, 1]
B_channel = rgb_img[:, :, 2]

cv2.imwrite(os.path.join(output_dir, "R_channel_gray.png"), R_channel)
cv2.imwrite(os.path.join(output_dir, "G_channel_gray.png"), G_channel)
cv2.imwrite(os.path.join(output_dir, "B_channel_gray.png"), B_channel)
