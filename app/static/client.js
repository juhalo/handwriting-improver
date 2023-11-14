const canvas = document.getElementById('drawing-board');
const clearBtn = document.querySelector('#clear');
const sendBtn = document.querySelector('#send');
const downloadBtn = document.querySelector('#download');
const loadBtn = document.querySelector('#load');
const form = document.querySelector('form');
const input = document.querySelector('#file-input');
const ctx = canvas.getContext('2d');
const prediction = document.getElementById('predictions');

let isDrawing = false;

function initializeCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'black';
}

initializeCanvas();

const draw = (e) => {
  if (isDrawing) {
    // We have to declare this here (instead of outside) since otherwise resizing does not work properly
    var rect = canvas.getBoundingClientRect();
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    // Shout out to https://stackoverflow.com/questions/17130395/real-mouse-position-in-canvas
    // for coming up how to make the lineTo algorithm work with resizing and different screen sizes
    ctx.lineTo(
      ((e.clientX - rect.left) / (rect.right - rect.left)) * canvas.width,
      ((e.clientY - rect.top) / (rect.bottom - rect.top)) * canvas.height
    );
    ctx.stroke();
  }
};

canvas.addEventListener('mousedown', (_) => {
  isDrawing = true;
});

canvas.addEventListener('mouseup', (_) => {
  isDrawing = false;
  ctx.stroke();
  ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

clearBtn.addEventListener('click', (_) => {
  const predictions = document.getElementById('predictions');
  predictions.innerHTML = '';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  initializeCanvas();
});

sendBtn.addEventListener('click', async (_) => {
  sendBtn.disabled = true;
  console.log('HiHi');
  // window.alert('HiHouuuU');
  var canvasData = canvas.toDataURL('image/png');
  const formData = new FormData();
  formData.append('file', canvasData);
  const response = await fetch('/predict/', {
    method: 'POST',
    body: formData,
  });

  const json_response = await response.json();
  prediction.innerHTML = json_response.prediction;
  sendBtn.disabled = false;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  console.log('HereHere');
  // window.alert('HereHo');
  loadBtn.disabled = true;
  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/predict/', {
    method: 'POST',
    body: formData,
  });

  const json_response = await response.json();
  prediction.innerHTML = json_response.prediction;
  loadBtn.disabled = false;
});

downloadBtn.addEventListener('click', (_) => {
  var link = document.createElement('a');
  link.setAttribute('download', 'your_image.png');
  link.setAttribute(
    'href',
    canvas.toDataURL('image/png').replace('image/png', 'image/octet-stream')
  );
  link.click();
});
