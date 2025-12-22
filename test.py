import os
import time
import argparse
import collections
import numpy as np
import cv2

DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

def load_torch_model(path: str, device: str):
    import torch
    from common.models import GestureCNNTemporal
    obj = torch.load(path, map_location=device, weights_only=False)
    classes = obj.get("classes") or obj.get("meta", {}).get("classes")
    meta = obj.get("meta", {})
    num_classes = int(meta.get("dataset", {}).get("num_classes", len(classes) if classes else 3))
    hidden = int(meta.get("hidden_size", 64))
    model = GestureCNNTemporal(num_classes=num_classes, hidden_size=hidden)
    model.load_state_dict(obj["model_state"])  # type: ignore
    model.to(device)
    model.eval()
    return model, classes or ["paper", "rock", "scissors"], device

def load_tf_model(path: str):
    import tensorflow as tf
    m = tf.keras.models.load_model(path)
    return m

def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x -= np.max(x)
    e = np.exp(x)
    s = e.sum()
    return (e / s) if s > 0 else np.zeros_like(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.path.join("models", "gesture_recognition_cnn_temporal_v1.pth"))
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError("mediapipe not installed") from e

    ext = os.path.splitext(args.model_path)[1].lower()
    if ext == ".pth":
        import torch
        model, classes, device = load_torch_model(args.model_path, DEVICE)
        use_torch = True
    elif ext == ".h5":
        tf_model = load_tf_model(args.model_path)
        classes = ["paper", "rock", "scissors"]
        use_torch = False
    else:
        raise ValueError("unsupported model format")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    seq = collections.deque(maxlen=args.seq_len)
    last = time.perf_counter()
    fps_show = 0.0
    label_text = ""
    prob_val = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
            xs = []
            for lm in lms.landmark:
                xs.extend([lm.x, lm.y, lm.z])
            feat = np.array(xs, dtype=np.float32)
            seq.append(feat)
            if len(seq) < args.seq_len:
                pad = np.repeat(seq[-1][None, :], args.seq_len - len(seq), axis=0)
                arr = np.concatenate([np.stack(list(seq)), pad], axis=0)
            else:
                arr = np.stack(list(seq))
            if ext == ".pth" and use_torch:
                import torch
                inp = torch.tensor(arr[None, ...], dtype=torch.float32).to(DEVICE)
                out = model(inp)
                probs = torch.softmax(out[0], dim=0).detach().cpu().numpy()
            else:
                inp = arr[None, ...].astype(np.float32)
                y = tf_model.predict(inp, verbose=0)
                probs = softmax(y[0])
            idx = int(np.argmax(probs))
            prob_val = float(probs[idx])
            label_text = f"{classes[idx] if idx < len(classes) else idx}: {prob_val:.2f}"
        else:
            label_text = "no hand"
            prob_val = 0.0

        now = time.perf_counter()
        dt = now - last
        last = now
        fps_show = 0.9 * fps_show + 0.1 * (1.0 / dt)

        cv2.putText(frame, f"FPS: {fps_show:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        color = (0, 0, 255) if prob_val < args.threshold else (0, 255, 0)
        cv2.putText(frame, label_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    try:
        hands.close()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
