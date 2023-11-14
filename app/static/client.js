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
  // https://stackoverflow.com/questions/72300422/how-can-i-post-canvas-data-in-javascript
  const dataURItoBlob = function (dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
      byteString = atob(dataURI.split(',')[1]);
    else byteString = unescape(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], { type: mimeString });
  };
  const url = canvas.toDataURL();
  const file = dataURItoBlob(url);
  const formData = new FormData();
  formData.append('file', file);
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
