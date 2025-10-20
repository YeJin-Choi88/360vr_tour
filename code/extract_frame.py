import cv2
import os

def extract_frames(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: '{video_path}' 열기 실패")
        return

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"{idx:06d}.png")
        cv2.imwrite(filename, frame)
        idx += 1

    cap.release()
    print(f"'{video_path}' → {idx} frames saved to '{output_dir}'")

def main():
    extract_frames('.mp4', 's1_1')
    # extract_frames('2_cali.mp4', 's1_2')
    # extract_frames('3_cali.mp4', 's1_3')
    # extract_frames('4_cali.mp4', 's1_4')
    # extract_frames('5_cali.mp4', 's1_5')
    # extract_frames('6_cali.mp4', 's1_6')


if __name__ == '__main__':
    main()
