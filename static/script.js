const revealElements = document.querySelectorAll('.reveal');

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.12,
  }
);

revealElements.forEach((element) => revealObserver.observe(element));

const yearElement = document.getElementById('year');
if (yearElement) {
  yearElement.textContent = new Date().getFullYear();
}

const uploadInput = document.getElementById('mri-upload');
const previewImage = document.getElementById('preview-image');
const placeholderText = document.getElementById('upload-placeholder');
const predictionLabel = document.getElementById('prediction-label');
const predictionScore = document.getElementById('prediction-score');

// Handle image upload and prediction
if (uploadInput && previewImage && placeholderText && predictionLabel && predictionScore) {
  uploadInput.addEventListener('change', async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Show image preview
    const imageUrl = URL.createObjectURL(file);
    previewImage.src = imageUrl;
    previewImage.style.display = 'block';
    placeholderText.style.display = 'none';

    // Show loading state
    predictionLabel.textContent = "Analyzing...";
    predictionScore.textContent = "—";

    // Read image as Base64 and send to FastAPI /predict endpoint
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        // Strip data:image/...;base64, prefix for decodeImage()
        const base64Image = reader.result.split(',')[1];

        const response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: base64Image }),
        });

        if (!response.ok) {
          throw new Error(`Inference failed with status ${response.status}`);
        }

        const result = await response.json();
        const prediction = result.data[0];

        predictionLabel.textContent = prediction.prediction;
        predictionScore.textContent = `${prediction.confidence}%`;
      } catch (err) {
        console.error(err);
        predictionLabel.textContent = "Error";
        predictionScore.textContent = "Failed";
      }
    };
    reader.readAsDataURL(file);
  });
}