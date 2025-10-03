
document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('file-input');
    const resultElement = document.getElementById('result');
    const file = fileInput.files[0];

    if (!file) {
        resultElement.textContent = 'Please select a file.';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    resultElement.textContent = 'Predicting...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction request failed');
        }

        const data = await response.json();
        resultElement.textContent = data.prediction;

    } catch (error) {
        resultElement.textContent = 'Error: ' + error.message;
    }
});

document.getElementById('upload-form-stone-type').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('file-input-stone-type');
    const resultElement = document.getElementById('result-stone-type');
    const file = fileInput.files[0];

    if (!file) {
        resultElement.textContent = 'Please select a file.';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    resultElement.textContent = 'Classifying...';

    try {
        const response = await fetch('/predict_stone_type', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Classification request failed');
        }

        const data = await response.json();
        resultElement.textContent = data.prediction;

    } catch (error) {
        resultElement.textContent = 'Error: ' + error.message;
    }
});
