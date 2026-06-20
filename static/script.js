// Poll metrics every 500ms
async function fetchMetrics() {
    try {
        const data = await fetch('/metrics').then(r => r.json());

        const statusTextEl = document.getElementById('model-status-text');

        if (!data.model_ready) {
            // Show loading status from /status endpoint
            fetch('/status').then(r => r.json()).then(s => {
                statusTextEl.textContent = s.status;
            });
            document.getElementById('val-detected').textContent = "0";
            document.getElementById('val-ripe').textContent = "0";
            document.getElementById('val-unripe').textContent = "0";
        } else {
            statusTextEl.textContent = "";
            document.getElementById('val-detected').textContent = data.detected;
            document.getElementById('val-ripe').textContent = data.ripe;
            document.getElementById('val-unripe').textContent = data.unripe;
        }

        const badge = document.getElementById('status-badge');
        badge.style.color = data.camera_running ? 'var(--accent-green)' : 'var(--accent-red)';
        badge.textContent = data.camera_running ? '● Live' : '● Offline';
        badge.style.animation = data.camera_running ? '' : 'none';

    } catch (e) {
        console.error('Metrics fetch error:', e);
    }
}

setInterval(fetchMetrics, 500);

// Image upload predict
document.getElementById('img-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const resultEl = document.getElementById('upload-result');
    resultEl.textContent = 'Analyzing...';
    resultEl.style.color = 'var(--text-muted)';

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();

        if (data.error) {
            resultEl.textContent = data.error;
            resultEl.style.color = 'var(--accent-red)';
        } else {
            resultEl.innerHTML = `
                Found ${data.detected} arecanuts:<br>
                <span class="text-green">${data.ripe} Ripe</span> | <span class="text-red">${data.unripe} Unripe</span><br>
                <img src="data:image/jpeg;base64,${data.image_b64}" style="max-width: 100%; max-height: 300px; margin-top: 10px; border-radius: 8px;">
            `;
            resultEl.style.color = 'var(--text-main)';
        }
    } catch (err) {
        resultEl.textContent = 'Request failed';
        resultEl.style.color = 'var(--accent-red)';
    }
});
