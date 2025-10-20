import cv2, yaml, os, numpy as np

def read_yaml(path: str):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    K = np.array(d['camera_matrix']['data'],  np.float32).reshape(3, 3)
    D = np.array(d['distortion_coefficients']['data'], np.float32).ravel()
    w0 = int(d['image_width']);  h0 = int(d['image_height'])
    return K, D, w0, h0

def undistort_video_keep_fov(out_path, alpha=1.0, idx=1):
    yml  = f"camera_{idx}.yaml"
    vin  = f"camera{idx}_output.mp4"
    
    cap = cv2.VideoCapture(vin)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    K, D, w0, h0 = read_yaml(yml)

    # alpha=1.0 으로 설정하면 크롭 없이 최대한 블랙 영역을 줄입니다.
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha, (w,h))
    # remap 용 맵 생성
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None,
                                             newK, (w,h), cv2.CV_32FC1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # remap 으로 보정
        und = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        # **절대 ROI 크롭 금지** 하여 수평·전체 영역 유지
        out.write(und)

    cap.release()
    out.release()
    print(f"Saved undistorted video to {out_path}")

if __name__ == "__main__":
    undistort_video_keep_fov('6_cali.mp4', alpha=1.0, idx = 6)
