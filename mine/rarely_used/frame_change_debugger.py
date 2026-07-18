import cv2
import numpy as np
import os

# === 基本設定 ===
videoname = r"C:\Users\roger\IDEA\Boom\data\砲擊彈著點-1.mp4"
display_scale_factor = 0.35
process_scale_factor = 0.25
info_width = 320

# RGB / HSV 全圖變化偵測參數
rgb_diff_threshold = 28
hue_diff_threshold = 12
sat_diff_threshold = 30
val_diff_threshold = 30
min_area = 1800
max_area = 160000
blur_size = 5

open_kernel = np.ones((3, 3), dtype=np.uint8)
close_kernel = np.ones((7, 7), dtype=np.uint8)


def build_change_mask(prev_frame_small, frame_small):
    prev_blur = cv2.GaussianBlur(prev_frame_small, (blur_size, blur_size), 0)
    curr_blur = cv2.GaussianBlur(frame_small, (blur_size, blur_size), 0)

    rgb_diff = cv2.absdiff(prev_blur, curr_blur)
    b_diff = rgb_diff[..., 0]
    g_diff = rgb_diff[..., 1]
    r_diff = rgb_diff[..., 2]
    rgb_score = np.mean(rgb_diff, axis=2).astype(np.uint8)
    rgb_mask = (rgb_score >= rgb_diff_threshold).astype(np.uint8) * 255

    prev_hsv = cv2.cvtColor(prev_blur, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_blur, cv2.COLOR_BGR2HSV)
    hsv_diff = cv2.absdiff(prev_hsv, curr_hsv)

    hue_diff = hsv_diff[..., 0]
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    sat_diff = hsv_diff[..., 1]
    val_diff = hsv_diff[..., 2]

    hsv_mask = (
        (hue_diff >= hue_diff_threshold)
        | (sat_diff >= sat_diff_threshold)
        | (val_diff >= val_diff_threshold)
    ).astype(np.uint8) * 255

    combined_mask = cv2.bitwise_or(rgb_mask, hsv_mask)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, open_kernel)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, close_kernel)
    combined_mask = cv2.dilate(combined_mask, open_kernel, iterations=1)

    channel_views = {
        "R Diff": r_diff,
        "G Diff": g_diff,
        "B Diff": b_diff,
        "H Diff": np.uint8(hue_diff.astype(np.float32) * (255.0 / 90.0)),
        "S Diff": sat_diff,
        "V Diff": val_diff,
    }

    return combined_mask, channel_views


def draw_change_boxes(frame, change_mask):
    contours, _ = cv2.findContours(
        change_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    result_frame = frame.copy()
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt) / (process_scale_factor ** 2)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x = int(x / process_scale_factor)
        y = int(y / process_scale_factor)
        w = int(w / process_scale_factor)
        h = int(h / process_scale_factor)

        detections.append((x, y, w, h, area))

    detections.sort(key=lambda item: item[4], reverse=True)
    return result_frame, detections


def build_dashboard_view(result_small, channel_views):
    tile_width = result_small.shape[1] // 2
    tile_height = result_small.shape[0] // 3

    channel_tiles = {}
    for window_name in ["R Diff", "G Diff", "B Diff", "H Diff", "S Diff", "V Diff"]:
        channel_view = channel_views[window_name]
        channel_bgr = cv2.cvtColor(channel_view, cv2.COLOR_GRAY2BGR)
        channel_small = cv2.resize(channel_bgr, (tile_width, tile_height))
        cv2.putText(
            channel_small,
            window_name,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        channel_tiles[window_name] = channel_small

    row_1 = np.hstack((channel_tiles["R Diff"], channel_tiles["H Diff"]))
    row_2 = np.hstack((channel_tiles["G Diff"], channel_tiles["S Diff"]))
    row_3 = np.hstack((channel_tiles["B Diff"], channel_tiles["V Diff"]))
    right_panel = np.vstack((row_1, row_2, row_3))

    return np.hstack((result_small, right_panel))


def main():
    os.makedirs("./output", exist_ok=True)

    video_path = videoname if os.path.splitext(
        videoname)[1] else f"{videoname}.mp4"
    if not os.path.exists(video_path):
        print(f"找不到影片: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, prev_frame = cap.read()
    if not ret:
        print("無法讀取第一幀")
        cap.release()
        return

    prev_frame_small = cv2.resize(
        prev_frame,
        (0, 0),
        fx=process_scale_factor,
        fy=process_scale_factor,
    )

    result_small = cv2.resize(
        prev_frame,
        (0, 0),
        fx=display_scale_factor,
        fy=display_scale_factor,
    )
    out_height, out_width = result_small.shape[:2]

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"./output/{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"{output_dir}/output_{base_name}.mp4",
        fourcc,
        fps,
        (out_width, out_height),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(
            frame,
            (0, 0),
            fx=process_scale_factor,
            fy=process_scale_factor,
        )

        change_mask, channel_views = build_change_mask(
            prev_frame_small, frame_small)
        result_frame, detections = draw_change_boxes(frame, change_mask)
        result_small = cv2.resize(
            result_frame,
            (0, 0),
            fx=display_scale_factor,
            fy=display_scale_factor,
        )

        dashboard_view = build_dashboard_view(result_small, channel_views)
        cv2.imshow("Detection + RGB HSV", dashboard_view)
        out.write(result_small)

        prev_frame_small = frame_small.copy()

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
