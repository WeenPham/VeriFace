# 🧠 VeriFace — Deepfake Detection & Generation Web Application

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Flask](https://img.shields.io/badge/Framework-Flask-orange)
![PyTorch](https://img.shields.io/badge/DeepLearning-PyTorch-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

<p align="center">
  <img src="https://raw.githubusercontent.com/WeenPham/VeriFace/main/assets/banner.png" alt="VeriFace Banner" width="80%">
</p>

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Important Notes](#-important-notes)
- [Contributing](#-contributing)
- [Credits](#-credits)
- [Acknowledgments](#-acknowledgments)
- [Ethical Considerations](#-ethical-considerations)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 📖 Overview

**VeriFace** is a deep learning–powered web application built with **Flask** that allows users to **detect** and **generate deepfakes**.  
It provides an intuitive interface for analyzing videos or webcam feeds to identify manipulated content and for creating realistic face swaps.

---

## ⚙️ Features

- 🎭 **Deepfake Detection:** Upload a video or use your webcam to determine if faces in the content are real or manipulated.  
- 🧬 **Deepfake Generation:** Swap faces between a source and target image to create realistic deepfakes.  
- ⚡ **Real-time Processing:** Detect deepfakes live using your webcam feed.  
- 📹 **Video Analysis:** Analyze uploaded videos and automatically save detection results.

---

## 🧩 Prerequisites

- **Python 3.10**
- **CUDA (optional)** — Recommended for GPU acceleration.

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/WeenPham/VeriFace.git
cd VeriFace
```

### 2️⃣ Create Virtual Environment and Install Dependencies

This project requires Python 3.10.

```bash
# --- For Linux/macOS ---
python3.10 -m venv venv
source venv/bin/activate

# --- For Windows (PowerShell) ---
python -m venv venv
venv\Scripts\activate

# --- Install the required dependencies ---
pip install -r requirements.txt
```

### 3️⃣ Download Pre-trained Models

Download the pre-trained model weights (`meso_net_epoch_40-50.pth`) and place them in the `app/models/checkpoints/` directory.

**Note:** For optimal performance, especially with real-time processing, run the application on a machine with a GPU and CUDA installed.

---

## 🚀 Usage

### 1️⃣ Start the Flask Application

```bash
python -m flask run -p <port-num>
```

This will start the application on `http://localhost:<port-num>`.

### 2️⃣ Access the Web Interface

Open your web browser and navigate to `http://localhost:<port-num>`.

### 3️⃣ Explore Features

* **Home Page:** View general information about the project.
* **Detect Page:** Upload an MP4 video or use your webcam to detect deepfakes in real-time.
* **Generate Page:** Upload source and target images (JPEG or PNG) to generate a deepfake image.

### 4️⃣ View Results

The application processes your inputs and displays the results on the respective pages.

---

## ⚠️ Important Notes

* **Webcam Access:** Ensure your browser has permission to access your camera for real-time detection.
* **Supported Formats:**
  * Videos: MP4
  * Images: JPEG, PNG

---

## 🤝 Contributing

Found a bug or have a suggestion?  
* Open an issue on the GitHub repository  
* Pull requests are welcome!

---

## 👨‍💻 Credits

* **Project Owner:** [ThangPNH](https://github.com/WeenPham) (ThangPNH.work@gmail.com)

---

## 🙏 Acknowledgments

This project leverages the following open-source libraries and models:
* InsightFace
* MediaPipe
* Flask
* PyTorch
* OpenCV

---

## ⚖️ Ethical Considerations

This project is designed for **educational and research purposes only**. Deepfake technology carries significant ethical implications, and we urge users to employ it responsibly and ethically.

---

## ⚠️ Disclaimer

The creators of this project are not liable for any misuse of this technology. Users are responsible for ensuring their use complies with all applicable laws and regulations.

---

## 🪪 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Note:** Exact versions may vary. Always check for the latest compatible versions.
