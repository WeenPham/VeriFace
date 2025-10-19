document.addEventListener('DOMContentLoaded', function () {
    const cameraButton = document.getElementById('cameraButton');
    const detectButton = document.getElementById('detectButton');
    const uploadButton = document.getElementById('uploadDetectVideoBtn');
    const videoUpload = document.getElementById('detectVideoUpload');
    const cameraFeed = document.getElementById('cameraFeed');
    const deepfakeImage = document.getElementById('deepfakeImage');
    const statusInfo = document.getElementById('statusInfo');
    const parametersInfo = document.getElementById('parametersInfo');

    let cameraOn = false;
    let detectionOn = false;
    let detectionInterval = null;

    cameraButton.addEventListener('click', () => {
        cameraOn = !cameraOn;
        if (cameraOn) {
            // Start camera
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    cameraFeed.srcObject = stream;
                    cameraFeed.style.display = 'block';
                    deepfakeImage.style.display = 'none';
                })
                .catch(err => {
                    console.error("Error accessing camera: ", err);
                });
            cameraButton.textContent = 'Toggle Camera (On)';
        } else {
            // Stop camera
            let stream = cameraFeed.srcObject;
            if (stream) {
                let tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
                cameraFeed.srcObject = null;
            }
            cameraButton.textContent = 'Toggle Camera (Off)';
        }
    });

    detectButton.addEventListener('click', () => {
        const spinner = detectButton.querySelector('.spinner-border');
        const buttonText = detectButton.querySelector('.button-text');

        detectionOn = !detectionOn;
        if (detectionOn) {
            if (!cameraOn) {
                alert('Please turn on the camera first.');
                detectionOn = false;
                return;
            }
            
            spinner.style.display = 'inline-block';
            buttonText.textContent = 'Stop Detection';
            statusInfo.textContent = 'DeepFake Detection Turned On';
            
            detectionInterval = setInterval(() => {
                const canvas = document.createElement('canvas');
                canvas.width = cameraFeed.videoWidth;
                canvas.height = cameraFeed.videoHeight;
                const context = canvas.getContext('2d');
                context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
                const frame = canvas.toDataURL('image/jpeg');

                fetch('/predict_deepfake', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: frame,
                        source: 'webcam',
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.annotated_frame) {
                        deepfakeImage.src = data.annotated_frame;
                        deepfakeImage.style.display = 'block';
                        cameraFeed.style.display = 'none';
                    } else {
                        deepfakeImage.style.display = 'none';
                        cameraFeed.style.display = 'block';
                    }
                    if (data.results) {
                        parametersInfo.textContent = JSON.stringify(data.results, null, 2);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    // In case of error, stop the detection
                    spinner.style.display = 'none';
                    buttonText.textContent = 'Detect Deepfake';
                    detectionOn = false;
                    clearInterval(detectionInterval);
                });
            }, 200); // Send a request every 200ms

        } else {
            spinner.style.display = 'none';
            buttonText.textContent = 'Detect Deepfake';
            statusInfo.textContent = 'DeepFake Detection Turned Off';
            clearInterval(detectionInterval);
            deepfakeImage.style.display = 'none';
            cameraFeed.style.display = 'block';
        }
    });

    uploadButton.addEventListener('click', () => {
        videoUpload.click();
    });

    videoUpload.addEventListener('change', (event) => {
        const files = event.target.files;
        // Handle video upload here
    });
});
