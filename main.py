import cv2
import socket
import numpy as np
from fer import FER
from flask import Flask, Response, render_template
import threading
import time

app = Flask(__name__)

# 共享变量
latest_frame = None
frame_lock = threading.Lock()
emotion_info = {"emotion": "等待检测", "confidence": 0}

# 中文字典
emotion_dict = {
    "angry": "生气", "happy": "高兴", "fear": "害怕",
    "sad": "悲伤", "surprise": "惊讶", "neutral": "常态",
    "disgust": "厌恶"
}

def udp_receiver():
    global latest_frame, emotion_info

    # 配置UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', 9090))
    print("UDP服务器已启动，等待ESP32-CAM连接...")

    # 初始化FER
    detector = FER()

    while True:
        try:
            # 接收数据（增大缓冲区）
            data, addr = sock.recvfrom(65535)
            print(f"收到来自 {addr} 的 {len(data)} 字节数据")

            # 转换为OpenCV格式
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("警告: 图像解码失败")
                continue

            # 情绪检测
            try:
                results = detector.detect_emotions(frame)
                if results:
                    emotion = max(results[0]['emotions'], key=results[0]['emotions'].get)
                    confidence = results[0]['emotions'][emotion]

                    # 绘制结果
                    (x, y, w, h) = results[0]['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion_dict.get(emotion, emotion)}: {confidence:.2f}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 更新共享数据
                    with frame_lock:
                        latest_frame = frame
                        emotion_info = {
                            "emotion": emotion_dict.get(emotion, emotion),
                            "confidence": confidence * 100
                        }
                else:
                    with frame_lock:
                        latest_frame = frame
                        emotion_info = {"emotion": "未检测到人脸", "confidence": 0}

            except Exception as e:
                print(f"情绪检测出错: {e}")
                with frame_lock:
                    latest_frame = frame

        except Exception as e:
            print(f"UDP接收错误: {e}")
            time.sleep(1)


def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_data')
def get_emotion():
    with frame_lock:
        return emotion_info


if __name__ == '__main__':
    # 启动UDP接收线程
    threading.Thread(target=udp_receiver, daemon=True).start()

    # 启动Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)