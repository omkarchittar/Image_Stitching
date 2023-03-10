import cv2
import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from matplotlib import rc
import matplotlib.pyplot as plt

images = []

def ransac(src, dst, goodMatches):
    threshold = 1
    iterations = 50
    best_count = 0 
    H_final = np.zeros((3,3))
    best_dist = np.inf

    normalize_src_x , normalize_src_y , T_src_normalize = keypointNorm(src , goodMatches)
    normalize_src = np.hstack((normalize_src_x.reshape(-1 ,1) , normalize_src_y.reshape(-1 ,1)))

    normalize_dst_x , normalize_dst_y , T_dst_normalize = keypointNorm(dst , goodMatches)
    normalize_dst = np.hstack((normalize_dst_x.reshape(-1 ,1) , normalize_dst_y.reshape(-1 ,1)))

    for i in range(iterations): 
        rand_4 = np.random.randint(0, goodMatches, size = 4)
        src_4 = np.vstack((normalize_src[rand_4[0],:], normalize_src[rand_4[1],:], normalize_src[rand_4[2],:], normalize_src[rand_4[3],:]))
        dst_4 = np.vstack((normalize_dst[rand_4[0],:], normalize_dst[rand_4[1],:], normalize_dst[rand_4[2],:], normalize_dst[rand_4[3],:]))
    
        H = homography(src_4, dst_4)
        post_H_dst = applyH(H, normalize_src)

        # Computing L2 norm between newly computed and existing points
        dist = np.linalg.norm(post_H_dst - normalize_dst, axis = 1)
        dis_sum = np.sum(dist)

        inliers = np.where(dist<threshold, 1, 0)
        count = np.sum(inliers)

        # Check condition to store the H matrix with max. inliers
        if count>best_count or (count==best_count and dis_sum<best_dist):
            best_count = count
            best_dist = dis_sum
            H_final = H

    # Denormalizing H matrix
    Homography = np.linalg.inv(T_dst_normalize) @ H_final @ T_src_normalize
    return Homography 

def homography(src, dst):
    src_h = np.hstack((src, np.ones((src.shape[0], 1))))
    A = np.array([np.block([[src_h[n], np.zeros(3), -dst[n, 0] * src_h[n]], [np.zeros(3), src_h[n], -dst[n, 1] * src_h[n]]]) for n in range(src.shape[0])]).reshape(2 * src.shape[0], 9)
    [_, _, V] = np.linalg.svd(A)
    h = V[-1,:]/V[-1,-1]
    h = np.array(h)
    return np.reshape(h, (3, 3))

def keypointNorm(points, goodMatches):
    x_mean = np.mean(points[:, 0, 0])
    y_mean = np.mean(points[:, 0, 1])

    xy_og = np.hstack(( points[:, 0, 0].reshape(-1,1), points[:, 0, 1].reshape(-1,1)))
    std_check = np.std(xy_og)

    trans_mat = np.array([[(np.sqrt(2)/std_check), 0, -(x_mean*np.sqrt(2)/std_check)], [0, (np.sqrt(2)/std_check), -(y_mean*np.sqrt(2)/std_check)], [0, 0, 1]])

    points = np.vstack((points[: , 0 , 0].reshape(1,-1), points[:, 0, 1].reshape(1,-1), np.ones((1 , goodMatches))))

    norm = np.matmul(trans_mat, points)
    norm_x = norm[0, :]
    norm_y = norm[1, :] 

    return norm_x, norm_y, trans_mat

def goodMatches(image1, image2, dst_image):   
    sift = cv2.SIFT_create()
    keypoint1, dst1 = sift.detectAndCompute(image1,None)
    keypoint2, dst2 = sift.detectAndCompute(image2,None)
    best = cv2.BFMatcher()
    if dst_image:
        matches = best.knnMatch(dst1 ,dst2, k=2)
        # Lowe's ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.4*n.distance:
                good.append(m)
        if len(good)>10:
            src_points = np.float32([keypoint1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_points = np.float32([keypoint2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)  
    else :
        matches = best.knnMatch(dst2 ,dst1, k=2)
        # Lowe's ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.4*n.distance:
                good.append(m)
        if len(good)>10:
            src_points = np.float32([keypoint2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_points = np.float32([keypoint1[m.trainIdx].pt for m in good]).reshape(-1,1,2)        
    return src_points, dst_points, len(good)

def applyH(H, src):
    src_h = np.hstack((src, np.ones((src.shape[0], 1))))
    dst_h = src_h @ H.T
    return (dst_h / dst_h[:, [2]])[:, 0:2]
    
sift = cv2.SIFT_create()

image1 = cv2.imread('image_1.jpg')
image1 = cv2.resize(image1, (int(4032/5), int(3024/5)))
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) 
images.append(image1)
kp_1, des_1 = sift.detectAndCompute(gray1,None)

image2 = cv2.imread('image_2.jpg')
image2 = cv2.resize(image2, (int(4032/5), int(3024/5)))
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
images.append(image2)
kp_2, des_2 = sift.detectAndCompute(gray2,None)

image3 = cv2.imread('image_3.jpg')
image3 = cv2.resize(image3, (int(4032/5), int(3024/5)))
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
gray3 = cv2.cvtColor(image3, cv2.COLOR_RGB2GRAY)
images.append(image3)
kp_3, des_3 = sift.detectAndCompute(gray3,None)

image4 = cv2.imread('image_4.jpg')
image4 = cv2.resize(image4, (int(4032/5), int(3024/5)))
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
gray4 = cv2.cvtColor(image4, cv2.COLOR_RGB2GRAY)
images.append(image4)
kp_4, des_4 = sift.detectAndCompute(gray4,None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

match12 = bf.match(des_1, des_2)
match12 = sorted(match12, key=lambda x: x.distance)
matched_image12 = cv2.drawMatches(image1, kp_1, image2, kp_2, match12[:50], None, flags=2)
cv2.imshow("1 and 2", matched_image12)
cv2.waitKey(0)

match23 = bf.match(des_2, des_3)
match23 = sorted(match23, key=lambda x: x.distance)
matched_image23 = cv2.drawMatches(image2, kp_2, image3, kp_3, match23[:50], None, flags=2)
cv2.imshow("2 and 3", matched_image23)
cv2.waitKey(0)

matches34 = bf.match(des_3, des_4)
matches34 = sorted(matches34, key=lambda x: x.distance)
matched_image34 = cv2.drawMatches(image3, kp_3, image4, kp_4, matches34[:50], None, flags=2)
cv2.imshow("3 and 4", matched_image34)
cv2.waitKey(0)

# padding middle image(2nd image from left)
center_image = np.zeros((gray1.shape[0] , gray1.shape[1]+gray2.shape[1]) , dtype = np.uint8)
center_image[: , gray1.shape[1]:] = gray2

# For Image 1 and 2
src , dst , match_count = goodMatches(gray1, center_image, True)
Homo1 = ransac(src , dst , match_count)
print(f"H12\n{Homo1}\n")

dst = cv2.warpPerspective(image1 , Homo1, (image1.shape[1] + image2.shape[1], image2.shape[0]))
dst[:, image2.shape[1]:] = image2

src_2 , dst_2 , match_count_2 = goodMatches(dst, gray3, False)
Homo2 = ransac(src_2 , dst_2 , match_count_2)
print(f"H23\n{Homo2}\n")

stitched = cv2.warpPerspective(image3 , Homo2,(dst.shape[1] + gray3.shape[1], gray3.shape[0]))
stitched[:, :dst.shape[1]] = dst

# For stitched (image1, image2, image3) and image 4
src_3 , dst_3 , match_count_3 = goodMatches(dst, gray4, False)
Homo3 = ransac(src_3 , dst_3 , match_count_3)
print(f"H34\n{Homo3}\n")

output = cv2.warpPerspective(image4, Homo3,(dst.shape[1] + gray4.shape[1], gray4.shape[0]))
output[:, : stitched.shape[1]] = stitched

cv2.imshow("Final", output)
cv2.waitKey(0)

fig, ax = plt.subplots()
im = ax.imshow(output)
plt.show()