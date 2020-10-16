from pathlib import Path
import shutil
import torchvision
import matplotlib.pyplot as plt
import util
import numpy as np


def main(args):
    device = util.get_device()

    # Process args
    predict_from_sec = util.hms_to_seconds(args.predict_from)
    predict_to_sec = util.hms_to_seconds(args.predict_to)

    # Load video
    vframes, _, info = torchvision.io.read_video(
        args.raw_video_path, start_pts=predict_from_sec, end_pts=predict_to_sec + 1, pts_unit="sec",
    )
    fps = info["video_fps"]

    # Process video
    _, height, width, num_channels = vframes.shape
    vframes = (vframes / 255.0).to(device)

    # Load model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # Predict and save pngs
    frames_dir = "frames"
    every_nth_frame = args.every_nth_frame
    num_seconds_predict = predict_to_sec - predict_from_sec
    num_frames_predict = int(fps * num_seconds_predict / every_nth_frame)
    fps_predict = fps / every_nth_frame
    frame_ids = np.arange(num_frames_predict) * every_nth_frame
    util.logging.info(
        f"Predicting {num_seconds_predict} seconds from {args.predict_from} to {args.predict_to} "
        f"({num_frames_predict} frames, {fps_predict} fps)"
    )
    for frame_id in frame_ids:
        # Predict
        prediction = model(
            [vframe.permute(2, 0, 1) for vframe in vframes[frame_id : frame_id + 1]]
        )[0]

        # Extract bounding boxes
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        person_label = util.COCO_INSTANCE_CATEGORY_NAMES.index("person")
        person_boxes = boxes[labels == person_label]

        # Plot
        fig, ax = plt.subplots(1, 1)
        ax.imshow(vframes[frame_id].cpu())
        for person_box in person_boxes:
            util.plot_box(ax, person_box)
        ax.set_axis_off()
        util.save_fig(fig, f"{frames_dir}/{frame_id}.png", tight_layout_kwargs={"pad": 0})

    # Make video
    util.make_video(
        [f"{frames_dir}/{frame_id}.png" for frame_id in frame_ids],
        args.processed_video_path,
        fps_predict,
    )

    # Remove png frames
    if Path(frames_dir).exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
        util.logging.info(f"Removed {frames_dir}")


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--predict-from", default="0:0:0", help=" ")
    parser.add_argument("--predict-to", default="0:0:10", help=" ")
    parser.add_argument("--every-nth-frame", default=5, type=int, help=" ")
    parser.add_argument("--raw-video-path", default="video.mp4", help=" ")
    parser.add_argument("--processed-video-path", default="video_processed.mp4", help=" ")

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
