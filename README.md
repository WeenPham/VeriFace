# VeriFace: Deepfake Detection and Generation Web Application

## Overview

This project is a web application built with Flask that enables users to detect and generate deepfakes using Deep learning models. It provides an intuitive interface for analyzing videos or webcam feeds to identify manipulated content and creating deepfake images through face swapping.


## Features

* **Deepfake Detection:** Upload a video or use your webcam to determine if faces in the content are real or manipulated.
* **Deepfake Generation:** Swap faces between a source and target image to create deepfake images.
* **Real-time Processing:** Detect deepfakes in real-time using a webcam feed.
* **Video Processing:** Analyze uploaded videos and save the detection results.

## Prerequisites

* Python 3.10
* CUDA (optional, for GPU acceleration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/VeriFace.git
cd VeriFace
```

### 2. Create Virtual Environment and Install Dependencies

This project requires Python 3.10.

```bash
# Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

Download the pre-trained model weights (`meso_net_epoch_40-50.pth`) and place them in the `app/models/checkpoints/` directory.

**Note:** For optimal performance, especially with real-time processing, run the application on a machine with a GPU and CUDA installed.

## Usage

### 1. Start the Flask Application

```bash
python -m flask run -p <port-num>
```

This will start the application on `http://localhost:<port-num>`.

### 2. Access the Web Interface

Open your web browser and navigate to `http://localhost:<port-num>`.

### 3. Explore Features

* **Home Page:** View general information about the project.
* **Detect Page:** Upload an MP4 video or use your webcam to detect deepfakes in real-time.
* **Generate Page:** Upload source and target images (JPEG or PNG) to generate a deepfake image.

### 4. View Results

The application processes your inputs and displays the results on the respective pages.

## Important Notes

* **Webcam Access:** Ensure your browser has permission to access your camera for real-time detection.
* **Supported Formats:**
  * Videos: MP4
  * Images: JPEG, PNG

## Contributing

Found a bug or have a suggestion? 
* Open an issue on the GitHub repository
* Pull requests are welcome!

## Credits

* **Project Owner:** [ThangPNH](https://github.com/ThangPNH) (ThangPNH.work@gmail.com)

## Acknowledgments

This project leverages the following open-source libraries and models:
* InsightFace
* MediaPipe
* Flask
* PyTorch
* OpenCV

## Ethical Considerations

This project is designed for **educational and research purposes only**. Deepfake technology carries significant ethical implications, and we urge users to employ it responsibly and ethically.

## Disclaimer

The creators of this project are not liable for any misuse of this technology. Users are responsible for ensuring their use complies with all applicable laws and regulations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Note:** Exact versions may vary. Always check for the latest compatible versions.