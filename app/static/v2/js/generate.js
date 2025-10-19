document.addEventListener('DOMContentLoaded', function () {
    const sourceImageUpload = document.getElementById('sourceImageUpload');
    const sourceImageGallery = document.getElementById('sourceImageGallery');
    const generateButton = document.getElementById('generateButton');
    const cameraButton = document.getElementById('cameraButton');
    const cameraFeed = document.getElementById('cameraFeed');
    const deepfakeOverlay = document.getElementById('deepfakeOverlay');

    let cameraOn = false;
    let selectedSourceImage = null;
    let isGenerating = false;

    sourceImageUpload.addEventListener('change', (event) => {
        const files = event.target.files;
        sourceImageGallery.innerHTML = '';
        for (const file of files) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const col = document.createElement('div');
                col.classList.add('col');
                const card = document.createElement('div');
                card.classList.add('card');
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('card-img-top');
                img.addEventListener('click', () => {
                    selectedSourceImage = e.target.result;
                    // Highlight the selected image
                    document.querySelectorAll('#sourceImageGallery img').forEach(i => i.classList.remove('border', 'border-primary'));
                    img.classList.add('border', 'border-primary');
                });
                card.appendChild(img);
                col.appendChild(card);
                sourceImageGallery.appendChild(col);
            };
            reader.readAsDataURL(file);
        }
    });

    cameraButton.addEventListener('click', () => {
        cameraOn = !cameraOn;
        if (cameraOn) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    cameraFeed.srcObject = stream;
                })
                .catch(err => {
                    console.error("Error accessing camera: ", err);
                });
            cameraButton.textContent = 'Toggle Camera (On)';
        } else {
            let stream = cameraFeed.srcObject;
            if (stream) {
                let tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
                cameraFeed.srcObject = null;
            }
            cameraButton.textContent = 'Toggle Camera (Off)';
        }
    });

    async function generateDeepfakeLoop() {
        while (isGenerating) {
            if (!selectedSourceImage) {
                console.warn("Please select a source image.");
                isGenerating = false;
                return;
            }
            
            const canvas = document.createElement('canvas');
            canvas.width = cameraFeed.videoWidth;
            canvas.height = cameraFeed.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
            const targetImage = canvas.toDataURL('image/jpeg');

            try {
                const response = await fetch('/generate_deepfake', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: selectedSourceImage, target: targetImage }),
                });
    
                const data = await response.json();
    
                if (data.deepfake_image) {
                    deepfakeOverlay.src = data.deepfake_image;
                    deepfakeOverlay.style.display = 'block';
                } else {
                    deepfakeOverlay.style.display = 'none';
                }
            } catch (error) {
                console.error("Error generating deepfake:", error);
                deepfakeOverlay.style.display = 'none';
            }
    
            await new Promise(resolve => setTimeout(resolve, 500)); // Wait 500ms before next frame

            if (!isGenerating) {
                deepfakeOverlay.style.display = 'none';
                deepfakeOverlay.src = '';
            }
        }
    }

    generateButton.addEventListener('click', function () {
        const spinner = generateButton.querySelector('.spinner-border');
        const buttonText = generateButton.querySelector('.button-text');

        isGenerating = !isGenerating; // Toggle generating state
        if (isGenerating) {
            if (!selectedSourceImage) {
                alert('Please select a source image.');
                isGenerating = false;
                return;
            }
            if (!cameraOn) {
                alert('Please turn on the camera.');
                isGenerating = false;
                return;
            }

            spinner.style.display = 'inline-block';
            buttonText.textContent = 'Stop Generation';
            
            generateDeepfakeLoop();

        } else {
            console.log("Stopping generation...");
            spinner.style.display = 'none';
            buttonText.textContent = 'Generate Deepfake';
        }
    });
});
