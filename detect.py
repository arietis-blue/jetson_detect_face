import cv2

# カスケード分類器の読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# カメラ映像の取得
cap = cv2.VideoCapture(0)  # /dev/video0 を使う場合

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔を検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 検出した顔にフレームを描画
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # フレームを表示
    cv2.imshow('Face Detection', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラとウィンドウのリソースを解放
cap.release()
cv2.destroyAllWindows()


# sudo apt update
# sudo apt install python3-opencv
# python3 detect.py
