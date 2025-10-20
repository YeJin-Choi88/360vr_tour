import cv2, numpy as np, math, sys

def align_and_crop(frame, strip_w=100, max_jump=3, clamp=1):
    h, w = frame.shape[:2]
    if strip_w <= 0 or strip_w*2 >= w:
        return frame.copy(), 0

    L = frame[:, :strip_w]
    R = frame[:, -strip_w:]

    if L.size == 0 or R.size == 0:
        return frame.copy(), 0

    L32 = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32).copy()
    R32 = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32).copy()

    try:
        (dx, dy), _ = cv2.phaseCorrelate(L32, R32)
        dy = int(round(dy))
        if math.isnan(dy):
            dy = 0
    except cv2.error:
        dy = 0

    if abs(dy) > max_jump:
        dy = clamp if dy > 0 else -clamp

    aligned = np.roll(frame, -dy, axis=0)
    cropped = aligned[:, strip_w:-strip_w]
    return cropped, dy

def process_video(src, dst, strip_w=100):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, fr = cap.read()
    if not ret:
        sys.exit("Empty video")

    fr0, dy0 = align_and_crop(fr, strip_w)
    H, W = fr0.shape[:2]
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    out.write(fr0);   print(f"[0] dy={dy0}")

    idx = 1
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        fr_aligned, dy = align_and_crop(fr, strip_w)
        if fr_aligned.shape[:2] != (H, W):            # 아주 드물게 크기 mismatch
            fr_aligned = cv2.resize(fr_aligned, (W, H), cv2.INTER_LINEAR)
        out.write(fr_aligned)
        if idx % 100 == 0 or dy != 0:
            print(f"[{idx}] dy={dy}")
        idx += 1

    cap.release(); out.release()
    print(f"Saved {idx} frames → {dst}")

if __name__ == "__main__":
    process_video("e8_7_f1_2.mp4", "e8_7_f1_2_aligned.mp4", strip_w=85)
