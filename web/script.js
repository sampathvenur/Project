async function handleUpload(file, resultElement, expectedType) {
    if (!file) {
        resultElement.textContent = 'Please select a file.';
        return;
    }

    // Reset previous detection results if they exist
    const processedImage = document.getElementById('processed-image');
    const measurementList = document.getElementById('measurement-list');
    if (processedImage) {
        processedImage.style.display = 'none';
        processedImage.src = '';
    }
    if (measurementList) {
        measurementList.innerHTML = '';
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('expected_type', expectedType);

    resultElement.textContent = 'Analyzing image...';

    try {
        const response = await fetch('/predict_image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || `Request failed: ${response.statusText}`);
        }

        let resultText = '';
        if (data.prediction_type === 'stone_detection') {
            const confidenceScore = (data.confidence * 100).toFixed(2);
            resultText = `${data.result} (Confidence: ${confidenceScore}%)`;

            // --- NEW: Handle Image and Measurements ---
            if (data.processed_image) {
                processedImage.src = `data:image/jpeg;base64,${data.processed_image}`;
                processedImage.style.display = 'block';
            }
            
            if (data.measurements && data.measurements.length > 0) {
                data.measurements.forEach(measurement => {
                    const li = document.createElement('li');
                    li.textContent = measurement;
                    measurementList.appendChild(li);
                });
            }
            // ------------------------------------------

        } else if (data.prediction_type === 'stone_type_classification') {
            resultText = `Prediction: ${data.result}`;
        } else {
            resultText = 'Received an unexpected response from the server.';
        }
        resultElement.textContent = resultText;

    } catch (error) {
        resultElement.textContent = 'Error: ' + error.message;
    }
}

document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file-input');
    const resultElement = document.getElementById('result');
    handleUpload(fileInput.files[0], resultElement, 'ct_scan');
});

document.getElementById('upload-form-stone-type').addEventListener('submit', function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file-input-stone-type');
    const resultElement = document.getElementById('result-stone-type');
    handleUpload(fileInput.files[0], resultElement, 'stone_image');
});