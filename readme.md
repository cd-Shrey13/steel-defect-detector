## 🧠 Surface Defect Detection on Hot-Rolled Steel Strips

An AI4MFG project that uses a deep learning model to detect and classify surface defects in steel strips with real-time Grad-CAM visualization and webcam support.

---

### 📌 Project Features

* ✅ **Image Upload** & prediction
* ✅ **Deep Learning-based classification** (6 defect types)
* ✅ **Grad-CAM heatmap** to explain model's decision
* ✅ **Downloadable PDF reports**
* ✅ **Live Camera mode** with real-time prediction
* ✅ **Deployable via Streamlit Cloud**

---

### 🏷️ Defect Classes Detected

* CRAZING
* INCLUSION
* PATCHES
* PITTED\_SURFACE
* ROLLED-IN-SCALE
* SCRATCHES

---

### 📸 Demo Screenshot

> Prediction: **PITTED\_SURFACE**
> Confidence: 94.73%
> With Grad-CAM visualizing the defect location:

![Demo Screenshot](/screenshots/pitted_surface.png)

---

### 🗂️ Folder Structure

```
├── app.py                  # Main Streamlit app
├── model/
│   └── steel_defect_classifier.h5
├── dataset/NEU-DET/        # Training + validation data
├── requirements.txt
└── README.md
```

---

### 🚀 Getting Started

#### 1. Clone the Repo

```bash
git clone https://github.com/cd-Shrey13/steel-defect-detector.git
cd steel-defect-detector
```

#### 2. Install Requirements

```bash
pip install -r requirements.txt
```

#### 3. Run the App

```bash
streamlit run app.py
```

---

### 📤 Deployment

Deploy easily using [Streamlit Cloud](https://share.streamlit.io):

* Connect your GitHub repo
* Select `app.py` as entry point
* Done ✅

---

### 📁 Dataset

We used the **NEU Surface Defect Dataset**. You can download it from <a href='https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database'>Here</a>
```

---

### 🎓 Technologies Used

* TensorFlow / Keras
* Streamlit
* Grad-CAM (XAI)
* OpenCV
* Python (FPDF, NumPy, PIL)


