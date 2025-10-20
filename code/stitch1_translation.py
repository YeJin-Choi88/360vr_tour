
import numpy as np
import cv2
from superpoint import SuperPointFrontend
import torch
import os
import time
import logging

logger = logging.getLogger("stitching")
logger.setLevel(logging.DEBUG)


# 카메라 내부 파라미터
K = np.array([[567.62351, 0., 352.19171],
              [0., 569.73487, 262.00278],
              [0., 0., 1.]], dtype=np.float32)

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

def scale_K(K, from_size, to_size):
    scale_x = to_size[0] / from_size[0]
    scale_y = to_size[1] / from_size[1]
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    return K_scaled

def cylindrical_warp(img, K):
    h_, w_ = img.shape[:2]
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_*w_, 3)
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(h_*w_, 3)
    B = K.dot(A.T).T
    B = B[:, :-1] / B[:, [-1]]
    B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)
    cylinder = cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cylinder

def estimate_translation(pts1, pts2):
    shifts = pts1 - pts2
    x_shift = int(np.median(shifts[:, 0]))
    y_shift = int(np.median(shifts[:, 1]))
    return x_shift, y_shift

def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        print("Warning: No non-black area found.")
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def alpha_blend(canvas_left, canvas_right):
    h, w, _ = canvas_left.shape
    maskL = (canvas_left.sum(2) > 0)
    maskR = (canvas_right.sum(2) > 0)

    blended = canvas_left.copy().astype(np.float32)

    for y in range(h):
        cols = np.where(maskL[y] & maskR[y])[0]
        if cols.size == 0:
            # 한쪽만 존재 → 빈 곳에 다른 쪽 색 삽입
            blended[y][maskR[y]] = canvas_right[y][maskR[y]]
            continue

        x0, x1 = cols[0], cols[-1]
        wov = x1 - x0 + 1
        alpha = np.linspace(0, 1, wov).astype(np.float32)

        # 좌·우 모두 있는 부분만 선형 혼합
        Lrow = canvas_left[y, x0:x1+1].astype(np.float32)
        Rrow = canvas_right[y, x0:x1+1].astype(np.float32)
        blended[y, x0:x1+1] = (1-alpha)[:,None]*Lrow + alpha[:,None]*Rrow

        # 오버랩 바깥에서 한쪽만 존재하는 픽셀 복사
        left_only  = maskL[y] & ~maskR[y]
        right_only = maskR[y] & ~maskL[y]
        blended[y, left_only]  = canvas_left [y, left_only ]
        blended[y, right_only] = canvas_right[y, right_only]

    return blended.astype(np.uint8)

def stitch_images(img1, img2, region_width=100):
    model_path = "/home/yejin/360camera/test/pythonProject/superpoint_v1.pth"
    superpoint = SuperPointFrontend(
        weights_path=model_path,
        nms_dist=2, conf_thresh=0.015, nn_thresh=0.7,
        cuda=torch.cuda.is_available()
    )

    # 그레이 변환 후 특징점 & 디스크립터 추출
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pts1, desc1, _ = superpoint.run(gray1)
    pts2, desc2, _ = superpoint.run(gray2)
    desc1 = desc1.T.astype(np.float32)
    desc2 = desc2.T.astype(np.float32)

    # 초기 매칭
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # 거리 필터링 & 지역 검사
    h1, w1 = img1.shape[:2]
    filtered = []
    for m in matches:
        if m.distance >= 0.7:
            continue
        x1 = pts1[0, m.queryIdx]
        x2 = pts2[0, m.trainIdx]
        if x1 >= w1 - region_width and x2 <= region_width:
            filtered.append(m)

    if len(filtered) < 3:
        print("Error: Not enough region matches.")
        return None

    # 매칭 시각화
    max_h = max(img1.shape[0], img2.shape[0])
    vis1 = cv2.resize(img1, (img1.shape[1], max_h))
    vis2 = cv2.resize(img2, (img2.shape[1], max_h))
    vis_img = np.hstack((vis1, vis2))
    for m in filtered:
        pt1 = (int(pts1[0, m.queryIdx]), int(pts1[1, m.queryIdx]))
        pt2 = (int(pts2[0, m.trainIdx]) + img1.shape[1], int(pts2[1, m.trainIdx]))
        cv2.circle(vis_img, pt1, 3, (0, 255, 0), -1)
        cv2.circle(vis_img, pt2, 3, (0, 0, 255), -1)
        cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)
    cv2.imshow("Matched Points", vis_img)

    # translation 계산
    pts1_m = np.float32([pts1[:2, m.queryIdx] for m in filtered])
    pts2_m = np.float32([pts2[:2, m.trainIdx] for m in filtered])
    x_shift = int(round((w1 + pts2_m[:, 0].mean()) - pts1_m[:, 0].mean()))
    y_shift = int(np.median(pts1_m[:, 1] - pts2_m[:, 1]))
    print(f"Estimated Translation: x = {x_shift}, y = {y_shift}")

    # 캔버스 생성
    h2, w2 = img2.shape[:2]
    canvas_w = w1 + w2 - abs(x_shift)
    # 세로 오프셋 계산
    y_off_left  = max(0, -y_shift)   # y_shift<0일 때 img1을 아래로 내림
    y_off_right = max(0,  y_shift)   # y_shift>0일 때 img2를 아래로 내림
    # 캔버스 높이는 두 이미지가 완전히 들어갈 만큼
    canvas_h    = max(h1 + y_off_left, h2 + y_off_right)

    canvas_left  = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_right = np.zeros_like(canvas_left)

    # img1 배치
    canvas_left[y_off_left : y_off_left + h1, 0 : w1] = img1

    # img2 배치
    x_off_right = w1 - x_shift
    canvas_right[y_off_right : y_off_right + h2,
                x_off_right : x_off_right + w2] = img2

    # 알파 블렌딩
    stitched = alpha_blend(canvas_left, canvas_right)
    return stitched

def process_and_show():
    img1 = cv2.imread('/home/yejin/360camera/python/s1_3/001100.png')
    img2 = cv2.imread('/home/yejin/360camera/python/s1_4/001100.png')
    if img1 is None or img2 is None:
        print("Error: Unable to read image(s). Check the file path or integrity.")
        return
    # 12 x = 171, y = 33
    # 34 x = 192, y = 2
    # 56 x = 169, y = -7
    # 1234  x = 184, y = 6
    # 3456  x = 145, y = 0
    #img1 = cv2.resize(img1, (640,480))
    #img2 = cv2.resize(img2, (640,480))

    cyl1 = cylindrical_warp(img1, K)
    cyl2 = cylindrical_warp(img2, K)


    result = stitch_images(cyl1, cyl2)
    if result is None:
        print("Stitching failed.")
        return
    result_cropped = auto_crop(result)
    cv2.imshow("Stitched Image", result_cropped)

    print("Press ESC to exit.")
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_and_show()
