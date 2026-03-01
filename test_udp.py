import cv2
import socket
import numpy as np
from fer import FER
from flask import Flask, Response, render_template
import threading
import time
import atexit
import logging
from PIL import Image, ImageDraw, ImageFont  # 添加Pillow库支持中文

app = Flask(__name__)

# 配置日志
logging.basicConfig(filename='app.log', level=logging.INFO)
app.logger.info('Application started')

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

# 全局socket对象
sock = None
# 字体文件路径
font_path = 'SimHei.ttf'  # 或者使用其他中文字体


def cleanup():
    """退出时释放资源"""
    global sock
    app.logger.info('Cleaning up resources...')
    if sock:
        sock.close()
    cv2.destroyAllWindows()


atexit.register(cleanup)


def draw_chinese_text(image, text, position, font_size=30, color=(0, 0, 255)):
    """使用Pillow在图像上绘制中文"""
    # 转换OpenCV图像为Pillow格式
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        # 尝试加载字体
        font = ImageFont.truetype(font_path, font_size)
    except:
        # 如果字体加载失败，使用默认字体
        font = ImageFont.load_default()
        app.logger.warning("中文字体加载失败，使用默认字体")

    # 绘制文本
    draw.text(position, text, font=font, fill=color)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def udp_receiver():
    global latest_frame, emotion_info, sock

    # 配置UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB缓冲区
    sock.bind(('0.0.0.0', 9090))
    sock.settimeout(0.1)  # 100ms超时
    app.logger.info("UDP服务器已启动在 0.0.0.0:9090")

    # 初始化FER
    detector = FER()
    frame_counter = 0
    skip_frames = 2  # 每3帧处理1帧

    while True:
        try:
            # 接收数据
            data, addr = sock.recvfrom(65535)
            app.logger.debug(f"收到来自 {addr} 的 {len(data)} 字节数据")

            # 转换为OpenCV格式
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                app.logger.warning("图像解码失败")
                continue

            frame_counter += 1
            if frame_counter % (skip_frames + 1) != 0:
                with frame_lock:
                    latest_frame = frame
                continue

            # 情绪检测
            try:
                results = detector.detect_emotions(frame)
                if results:
                    emotion = max(results[0]['emotions'], key=results[0]['emotions'].get)
                    confidence = results[0]['emotions'][emotion]
                    chinese_emotion = emotion_dict.get(emotion, emotion)
                    display_text = f"{chinese_emotion}: {confidence:.2f}"

                    # 绘制人脸框
                    (x, y, w, h) = results[0]['box']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # 使用Pillow绘制中文文本
                    frame = draw_chinese_text(frame, display_text, (x, y - 30))

                    # 更新共享数据
                    with frame_lock:
                        latest_frame = frame
                        emotion_info = {
                            "emotion": chinese_emotion,
                            "confidence": confidence * 100
                        }
                else:
                    with frame_lock:
                        latest_frame = frame
                        emotion_info = {"emotion": "未检测到人脸", "confidence": 0}

            except Exception as e:
                app.logger.error(f"情绪检测出错: {str(e)}", exc_info=True)
                with frame_lock:
                    latest_frame = frame

        except socket.timeout:
            continue
        except Exception as e:
            app.logger.error(f"UDP接收错误: {str(e)}", exc_info=True)
            time.sleep(0.1)


def generate_frames():
    last_frame_time = time.time()
    min_frame_interval = 1 / 30  # 30FPS

    while True:
        with frame_lock:
            if latest_frame is not None:
                current_time = time.time()
                if current_time - last_frame_time >= min_frame_interval:
                    ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        last_frame_time = current_time
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)  # 减少CPU占用


@app.route('/')
def index():
    app.logger.info('Accessed homepage')
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    app.logger.debug('Video feed accessed')
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_data')
def get_emotion():
    with frame_lock:
        app.logger.debug(f'Emotion data: {emotion_info}')
        return emotion_info


if __name__ == '__main__':
    # 防止端口占用
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 启动UDP接收线程
    threading.Thread(target=udp_receiver, daemon=True).start()

    # 启动Flask
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)