import os, sys, platform, argparse, torch, numpy, cv2
from pathlib import Path
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.utils import VID_FORMATS
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.utils import LOGGER, colorstr
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
MODELS = ROOT / 'weights'

# Add directories to PATH.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort')) # Necessary.

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

@torch.no_grad() # Don't update model weights.

# Run tracking.
def run(detector=MODELS / 'yolov8n.pt', # Default settings.
        re_id=MODELS / 'lmbn_n_cuhk03_d.pt',
        tracker='strongsort',
        project=ROOT / 'results' / 'trackings',
        name='run',
        track_config=None,  # Config for chosen tracker.
        source='0',         # File, stream or webcam (0).
        imgsz=(640, 640),   # Frame size.
        conf_thres=0.5,     # Confidence threshold.
        iou_thres=0.5,      # NMS IOU threshold.
        max_det=1000,       # Maximum detections per frame.
        device='',          # Cuda device, i.e. 0, 1, 2 or cpu.
        classes=None,       # Filter by class, e.g. 0 is background.
        agnostic_nms=False, # Should NMS be same for all classes.
        exist_ok=False,     # Existing project/name ok, do not increment.
        vid_stride=1,       # Amount of frames per step.
        half=False,         # Half-precision inference?
        hd_segs=False,      # Should segmentation masks be full resolution?
        show=False,         # Display results?
        save_conf=False,    # Save confidences?
        save_crop=False,    # Save cropped prediction boxes?
        save_paths=False,   # Save each track path?
        save_txt=False):    # Save results?

    # Checks and registers if source is video file or webcam stream.
    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('http://', 'https://', 'rtsp://', 'rtmp://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # Download file.

    # Setup directories for this run with appropriate naming.
    if not isinstance(detector, list):
        run = detector.stem
    elif type(detector) is list and len(detector) == 1:
        run = Path(detector[0]).stem
    else:
        run = 'ensemble' # Multiple models.

    run = name if name else run + "_" + re_id.stem
    save_dir = increment_path(Path(project) / run, exist_ok=exist_ok)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model into GPU if available.
    device = select_device(device)
    is_seg = '-seg' in str(detector) # Is segmentation version of yolo?
    model = AutoBackend(detector, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)

    # Create dataloader appropriate for the given source.
    if webcam:
        show = check_imshow(warn=True)
        dataset = LoadStreams(
            source, imgsz=imgsz, stride=stride, auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        batch_size = len(dataset) # Supports multiple webcam streams.
    else:
        dataset = LoadImages(
            source, imgsz=imgsz, stride=stride, auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        batch_size = 1

    vid_path, vid_writer, txt_path = [None] * batch_size, [None] * batch_size, [None] * batch_size
    model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3, *imgsz))

    # Create tracker instance for each video source.
    tracker_list = []
    for i in range(batch_size):
        tracker = create_tracker(tracker, track_config, re_id, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()

    outputs = [None] * batch_size

    # Prepare for detection and tracking.
    processed = 0; windows = []
    dt = (Profile(), Profile(), Profile(), Profile())
    curr_frames = [None] * batch_size
    prev_frames = [None] * batch_size

    # Run detection and tracking.
    for frame_index, batch in enumerate(dataset):
        path, img, im0s, vid_cap, s = batch
        with dt[0]:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0  # Normalize.
            if len(img.shape) == 3:
                img = img[None]  # Expand for batch dimension.

        # Infer.
        with dt[1]:
            preds = model(img)

        # Non-maximum suppression.
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections.
        for i, detection in enumerate(p):  # Per image.
            processed += 1
            if webcam:
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)

                # For video input.
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)
                # For series-of-images input.
                else:
                    txt_file_name = p.parent.name
                    save_path = str(save_dir / p.parent.name)

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)
            s += '%gx%g ' % img.shape[2:]
            imc = im0.copy() if save_crop else im0

            annotator = Annotator(im0, line_width=2, example=str(names))

            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):

                # Camera repositioning compensation.
                if prev_frames[i] is not None and curr_frames[i] is not None:
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if detection is not None and len(detection):
                if is_seg:
                    shape = im0.shape
                    if hd_segs:
                        detection[:, :4] = scale_boxes(img.shape[2:], detection[:, :4], shape).round()  # Rescale boxes to im0 size.
                        masks.append(process_mask_native(proto[i], detection[:, 6:], detection[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], detection[:, 6:], detection[:, :4], img.shape[2:], upsample=True))  # HWC
                        detection[:, :4] = scale_boxes(img.shape[2:], detection[:, :4], shape).round()  # Rescale boxes to im0 size.
                else:
                    detection[:, :4] = scale_boxes(img.shape[2:], detection[:, :4], im0.shape).round()  # Rescale boxes to im0 size.

                # Print results on terminal.
                for c in detection[:, 5].unique():
                    n = (detection[:, 5] == c).sum()  # Detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Pass detections to tracker.
                with dt[3]:
                    outputs[i] = tracker_list[i].update(detection.cpu(), im0)

                # Draw bounding boxes and segmentation.
                if len(outputs[i]) > 0:

                    if is_seg:
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in detection[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if hd_segs else img[i]
                        )

                    for j, (output) in enumerate(outputs[i]):
                        bbox = output[0:4]; id = output[4]
                        cls = output[5]; conf = output[6]

                        if save_txt: # In MOT format.
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]

                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_index + 1, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_conf or save_crop or show:
                            id = int(id); c = int(cls)
                            label = f'{id} {names[c]} {conf:.2f}'
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)

                            if save_paths and tracker == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(numpy.array(bbox, dtype=numpy.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

            else:
                pass

            # Display video output, i.e. tracking results, until 'q' is pressed.
            im0 = annotator.result()
            if show:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    exit()

            # Save video results.
            if save_conf:
                if vid_path[i] != save_path:
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

        # Print stats.
        LOGGER.info(f"{s}{'' if len(detection) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results.
    t = tuple(x.t / processed * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracker} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_conf:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', nargs='+', type=Path, default=MODELS / 'yolov8n.pt')
    parser.add_argument('--re-id', type=Path, default=MODELS / 'lmbn_n_cuhk03_d.pt')
    parser.add_argument('--tracker', type=str, default='botsort')
    parser.add_argument('--track-config', type=Path, default=None)
    parser.add_argument('--name', default='run')
    parser.add_argument('--project', default=ROOT / 'results' / 'trackings')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.5)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)
    parser.add_argument('--hd-segs', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-crop', action='store_true')
    parser.add_argument('--save-paths', action='store_true')
    parser.add_argument('--save-txt', action='store_true')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    opt.track_config = ROOT / 'trackers' / opt.tracker / 'configs' / (opt.tracker + '.yaml')

    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt')
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
