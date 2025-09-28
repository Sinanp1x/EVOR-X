const video = document.getElementById('video');
const resultDiv = document.getElementById('result');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    video.play();
    setInterval(captureAndSendFrame, 1000); // 1 frame per second
  })
  .catch(err => {
    resultDiv.textContent = 'Camera access denied.';
  });

function captureAndSendFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');
    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      resultDiv.textContent = data.class ? `Prediction: ${data.class}` : 'Error: ' + (data.error || 'Unknown error');
    });
  }, 'image/jpeg');
}
