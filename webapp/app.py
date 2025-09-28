
import os
import time
import logging
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, render_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Try several possible model locations (project root and webapp folder)
MODEL_CANDIDATES = [
    os.path.join(os.getcwd(), 'webapp', 'best.onnx'),
    os.path.join(os.getcwd(), 'best.onnx'),
    'best.onnx',
]

session = None
input_name = None
output_name = None
model_input_size = (640, 640)

for p in MODEL_CANDIDATES:
    if os.path.isfile(p):
        try:
            logger.info(f'Loading ONNX model from: {p}')
            session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            input_meta = session.get_inputs()[0]
            input_name = input_meta.name
            # try to infer input size from model shape
            shape = input_meta.shape  # e.g. [1,3,640,640]
            logger.info(f'Model input shape: {shape}')
            if shape and len(shape) >= 3:
                # detect channel-first vs channel-last
                try:
                    s = [int(x) if isinstance(x, (int,)) or (isinstance(x, str) and x.isdigit()) else None for x in shape]
                except Exception:
                    s = [None] * len(shape)
                # assume [N,C,H,W] or [N,H,W,C]
                if len(s) >= 4 and s[1] == 3 and s[2] is not None and s[3] is not None:
                    model_input_size = (s[3], s[2])
                elif len(s) >= 4 and s[3] == 3 and s[1] is not None and s[2] is not None:
                    model_input_size = (s[2], s[1])
                else:
                    # fall back
                    model_input_size = (640, 640)
            output_name = session.get_outputs()[0].name
            break
        except Exception as e:
            logger.exception(f'Failed to load model at {p}: {e}')

if session is None:
    logger.warning('No ONNX model found. The server will run but inference will be disabled.')
else:
    logger.info(f'Loaded model. Input name: {input_name}, output name: {output_name}, size: {model_input_size}')


def find_working_capture(max_index=4):
    # Try opening camera indices 0..max_index and return the first working VideoCapture
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue
        # try to read one frame
        ret, _ = cap.read()
        if ret:
            logger.info(f'Using camera index {idx}')
            return cap
        else:
            cap.release()
    return None


def preprocess(frame, size=None):
    if size is None:
        size = model_input_size
    img = cv2.resize(frame, size)
    # model expects RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # to CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def infer(frame):
    if session is None:
        return None
    blob = preprocess(frame, size=model_input_size)
    try:
        preds = session.run([output_name], {input_name: blob})[0]
        return preds
    except Exception as e:
        logger.exception(f'Inference failed: {e}')
        return None


def make_placeholder_frame(msg, width=640, height=480):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, msg, (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def gen_frames():
    # Prefer an actual camera; if not available, look for VIDEO_FILE env var, else use placeholder
    cap = find_working_capture()
    video_file = os.environ.get('VIDEO_FILE')
    if cap is None and video_file:
        if os.path.isfile(video_file):
            logger.info(f'Opening video file {video_file} as fallback')
            cap = cv2.VideoCapture(video_file)
        else:
            logger.warning(f'VIDEO_FILE set but not found: {video_file}')

    if cap is None:
        logger.warning('No camera or video source available; streaming placeholder frames.')
        while True:
            frame = make_placeholder_frame('No camera available', width=640, height=480)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    try:
        while True:
            success, frame = cap.read()
            if not success:
                logger.info('Frame read failed (end of video or camera error)')
                break

            preds = infer(frame)
            if preds is not None:
                # Minimal overlay showing model produced something
                cv2.putText(frame, 'Inference OK', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # show a tiny summary of preds shape
                try:
                    s = np.array(preds).shape
                    cv2.putText(frame, f'Out shape: {s}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                except Exception:
                    pass
            else:
                cv2.putText(frame, 'No model / inference disabled', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # throttle a bit
            time.sleep(0.03)
    finally:
        try:
            cap.release()
        except Exception:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logger.info(f'Starting server on port {port}')
    app.run(host="0.0.0.0", port=port, debug=True)
