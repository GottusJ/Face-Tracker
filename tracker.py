import PySimpleGUI as sg
import cv2

layout = [
    [sg.Image(key="-IMAGE-")],
    [sg.Text(".", key="-TEXT-", expand_x=True, justification="c")],
]
window = sg.Window("Face Detector", layout)

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    "randomPy/tracer/haarcascade_frontalface_default.xml"
)

while True:
    event, values = window.read(timeout=0)
    if event == sg.WIN_CLOSED:
        break

    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    imagebytes = cv2.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(data=imagebytes)

    window["-TEXT-"].update(f".")

window.close()
