from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import imutils
import easyocr

app = Flask(__name__)
socketio = SocketIO(app)

# Инициализация easyocr
reader = easyocr.Reader(['en', 'tr'])
recognized_plates = []  # Список для хранения истории распознанных номеров

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Преобразование в оттенки серого и шумоподавление
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        # Поиск контуров
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is not None:
            # Создание маски и обрезка изображения для номерного знака
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(frame, frame, mask=mask)

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

            # Распознавание текста
            result = reader.readtext(cropped_image)
            if result:
                text = result[0][-2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, text=text, org=(approx[0][0][0], approx[1][0][1] + 60),
                                    fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                frame = cv2.rectangle(frame, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

                # Добавление записи в историю
                recognized_plates.insert(0, {
                    'index': len(recognized_plates) + 1,
                    'plate': text,
                    'time': cv2.getTickCount()
                })

                # Отправка данных на клиент
                socketio.emit('plate_data', {'plate': text, 'cropped_image': cropped_image.tolist(), 'index': len(recognized_plates)})

        # Кодирование кадра
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
