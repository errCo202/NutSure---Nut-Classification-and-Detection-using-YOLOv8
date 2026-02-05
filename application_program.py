import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
from threading import Thread

model = YOLO('runs/detect/train2/weights/best.pt')  # model path

# Global variables
camera_on = False
cap = None


def load_image():
    global img, img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((550, 550))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        status_label.config(text="Image loaded. Ready for prediction.")


def predict_image():
    global img_path
    if img_path:
        results = model.predict(source=img_path, conf=0.7, save=False)
        annotated_img = results[0].plot()
        num_detections = len(results[0].boxes)

        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_img = Image.fromarray(annotated_img)
        annotated_img.thumbnail((550, 550))
        img_tk = ImageTk.PhotoImage(annotated_img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        if num_detections > 0:
            status_label.config(text=f"Prediction completed. {num_detections} detections found.")
        else:
            status_label.config(text="Prediction completed. No detections found.")


def start_camera():
    global camera_on, cap
    camera_on = True
    cap = cv2.VideoCapture(0)

    def process_camera():
        while camera_on:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=0.5, save=False)
            annotated_frame = results[0].plot()
            num_detections = len(results[0].boxes)

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(annotated_frame)
            frame_pil.thumbnail((550, 550))
            frame_tk = ImageTk.PhotoImage(frame_pil)

            image_label.config(image=frame_tk)
            image_label.image = frame_tk

            if num_detections > 0:
                status_label.config(text=f"Live: {num_detections} detections found.")
            else:
                status_label.config(text="Live: No detections found.")

        cap.release()

    Thread(target=process_camera).start()


def stop_camera():
    global camera_on, cap
    camera_on = False
    if cap is not None and cap.isOpened():
        cap.release()
    status_label.config(text="Camera stopped.")


root = tk.Tk()
root.title("YOLOv8 Image Prediction")
root.geometry("1600x900")
root.resizable(False, False)

bg_color = "#F5F5DC"
btn_color = "#D4B886"
font_color = "#5A4234"

root.configure(bg=bg_color)

logo_path = "logo.png"
logo_image = Image.open(logo_path)
logo_image = logo_image.resize((200, 200), Image.Resampling.LANCZOS)
logo_tk = ImageTk.PhotoImage(logo_image)
logo_label = Label(root, image=logo_tk, bg=bg_color)
logo_label.pack(pady=0)

header_frame = Frame(root, bg=bg_color)
header_frame.pack(pady=0)
header_label = Label(
    header_frame,
    text="Nut Classification Application",
    font=("Helvetica", 18, "bold"),
    bg=bg_color,
    fg=font_color,
)
header_label.pack()

image_label = Label(root, bg=bg_color)
image_label.pack(pady=20)

button_frame = Frame(root, bg=bg_color)
button_frame.pack(pady=10)

load_button = Button(
    button_frame,
    text="Load Image",
    font=("Helvetica", 12),
    bg=btn_color,
    fg=font_color,
    activebackground="#C8A772",
    activeforeground=font_color,
    command=load_image,
)
load_button.grid(row=0, column=0, padx=10)

predict_button = Button(
    button_frame,
    text="Predict",
    font=("Helvetica", 12),
    bg=btn_color,
    fg=font_color,
    activebackground="#C8A772",
    activeforeground=font_color,
    command=predict_image,
)
predict_button.grid(row=0, column=1, padx=10)

camera_button = Button(
    button_frame,
    text="Start Camera",
    font=("Helvetica", 12),
    bg=btn_color,
    fg=font_color,
    activebackground="#C8A772",
    activeforeground=font_color,
    command=start_camera,
)
camera_button.grid(row=0, column=2, padx=10)

stop_button = Button(
    button_frame,
    text="Stop Camera",
    font=("Helvetica", 12),
    bg=btn_color,
    fg=font_color,
    activebackground="#C8A772",
    activeforeground=font_color,
    command=stop_camera,
)
stop_button.grid(row=0, column=3, padx=10)

# Status Label
status_label = Label(
    root,
    text="Welcome! Please load an image or start the camera.",
    font=("Helvetica", 12),
    wraplength=500,
    bg=bg_color,
    fg=font_color,
)
status_label.pack(pady=20)

# Footer Frame
footer_frame = Frame(root, bg=bg_color)
footer_frame.pack(side=tk.BOTTOM, pady=10)
footer_label = Label(
    footer_frame,
    text="© 2025 Nut Detection App",
    font=("Helvetica", 10),
    bg=bg_color,
    fg=font_color,
)
footer_label.pack()

root.mainloop()
