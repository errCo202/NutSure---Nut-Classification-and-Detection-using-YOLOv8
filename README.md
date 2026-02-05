# 🥜 NutSure: Nut Classification and Detection using YOLOv8

**NutSure** is a computer vision application designed to identify and localize five different types of nuts in real-time. This project involved a full machine learning pipeline—from manual data curation and annotation to model training and GUI deployment.

---

### 🚀 Key Features
*   **Custom Object Detection:** Fine-tuned a **YOLOv8** model to detect and classify: *Almonds, Cashews, Walnuts, Peanuts, and Pistachios.*
*   **Data Engineering:** Curated a custom dataset from web-sourced images, manually annotated using **LabelIMG** to ensure high-quality bounding box accuracy.
*   **Desktop Application:** Developed a user-friendly interface using **Tkinter**, allowing users to perform inference on images easily.

### 🛠️ Tech Stack
*   **Language:** Python
*   **Computer Vision:** YOLOv8 (Ultralytics)
*   **Data Labeling:** LabelIMG
*   **GUI:** Tkinter

---

### 📸 Screenshots

<!-- You need to upload images to your repo or a folder named 'screenshots' to make these work -->
<!-- Format is: ![Alt Text](relative/path/to/image.png) -->

**1. Labeling Process (LabelIMG)**
![LabelIMG Interface](path/to/your/labelimg_screenshot.png)

**2. The NutSure App in Action**
![App Interface](path/to/your/app_screenshot.png)

**3. Model Training Results**
![Training Curves](path/to/your/results_chart.png)

---

### 💿 Installation & Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/errCo202/NutSure---Nut-Classification-and-Detection-using-YOLOv8.git
    ```
2.  Install dependencies:
    ```bash
    pip install ultralytics tk
    ```
3.  Run the application:
    ```bash
    python app.py
    ```
    *(Note: Replace `app.py` with the actual name of your main python file)*
