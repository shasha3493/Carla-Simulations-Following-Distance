import torch
import time
import torchvision

import numpy as np
import pkg_resources as pkg

# import contextlib

def check_version(current: str = '0.0.0',
                  minimum: str = '0.0.0',
                  name: str = 'version ',
                  pinned: bool = False,
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.

    Returns:
        (bool): True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f'WARNING ⚠️ {name}{minimum} is required, but {name}{current} is currently installed'
    if hard:
        assert result, (warning_message)  # assert min requirements met
    if verbose and not result:
        print(warning_message)
    return result


# class Profile(contextlib.ContextDecorator):
#     """
#     YOLOv8 Profile class.
#     Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'
#     """

#     def __init__(self, t=0.0):
#         """
#         Initialize the Profile class.

#         Args:
#             t (float): Initial time. Defaults to 0.0.
#         """
#         self.t = t
#         self.cuda = torch.cuda.is_available()

#     def __enter__(self):
#         """
#         Start timing.
#         """
#         self.start = self.time()
#         return self

#     def __exit__(self, type, value, traceback):
#         """
#         Stop timing.
#         """
#         self.dt = self.time() - self.start  # delta-time
#         self.t += self.dt  # accumulate dt

#     def time(self):
#         """
#         Get current time.
#         """
#         if self.cuda:
#             torch.cuda.synchronize()
#         return time.time()


def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        TORCH_1_9 = check_version(torch.__version__, '1.9.0')
        """Applies appropriate torch decorator for inference mode based on torch version."""
        return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate

# class TryExcept(contextlib.ContextDecorator):
#     """YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager."""

#     def __init__(self, msg='', verbose=True):
#         """Initialize TryExcept class with optional message and verbosity settings."""
#         self.msg = msg
#         self.verbose = verbose

#     def __enter__(self):
#         """Executes when entering TryExcept context, initializes instance."""
#         pass

#     def __exit__(self, exc_type, value, traceback):
#         """Defines behavior when exiting a 'with' block, prints error message if necessary."""
#         if self.verbose and value:
#             print(f"{self.msg}{': ' if self.msg else ''}{value}")
        # return True


def non_max_suppression(
        prediction,
        conf_thres=0.5,
        iou_thres=0.45,
        agnostic=False,
        max_det=300,
        max_nms=30000,
        max_wh=7680,
):

    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 5 # number of classes
    xc = prediction[:, 4] > conf_thres  # candidates

    time_limit = 0.5
    prediction = prediction.transpose(-2, -1)  
    # prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, conf, cls = x.split((4, 1, nc), 1)
        j = cls.max(1, keepdim=True)[1]
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    return output

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def cls_to_color(mask, task):
    '''
    Convert segmentation class to color code RGB.
    Args:
        mask (numpy.ndarray): segmentation bit map
        task (string): flag for task
    '''
    if task == 'drive': 
        color_dict = {
            0: [0, 255, 0],     # Green: Direct Area
            1: [255, 0, 0],     # Blue: Alternative Area
            2: [0, 0, 0]        # Black
        }
    elif task == 'lane':        # Update v0.1.6
        color_dict = {
            0: [0, 0, 0],       # Black = background
            1: [255, 255, 0],   # Aqua : Vertical Dash White Line
            2: [42, 42, 165],   # Brown: Vertical Solid White Line
            3: [255, 245, 152],  # Cadet Blue: Crosswalk and Horizontal Line 
            4: [87, 207, 227],   # Banana: Yellow Line 
            5: [147, 20, 255],   # Deeppink: Road Curb
        }
    else:
        raise ValueError("Undefined task for cls_to_color")
        
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, v in color_dict.items():
        color_mask[mask == k] = v
    return color_mask
