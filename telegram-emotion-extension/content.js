// content.js

// Function to extract message context (text, video, audio) from Telegram's web interface
function getContextFromTelegram() {
    const messages = document.querySelectorAll('.message'); // Adjust selector to Telegram's DOM
    const context = { text: [], videoFrames: [], audio: [] };

    Array.from(messages).slice(-5).forEach(msg => {
        // Extract text
        const textElement = msg.querySelector('.text');
        if (textElement) {
            context.text.push(textElement.textContent.trim());
        }

        // Extract video and convert to frames
        const videoElement = msg.querySelector('video');
        if (videoElement && videoElement.src) {
            extractVideoFrames(videoElement).then(frames => {
                context.videoFrames.push(...frames);
            });
        }

        // Extract audio message
        const audioElement = msg.querySelector('audio');
        if (audioElement && audioElement.src) {
            context.audio.push(audioElement.src);
        }
    });

    return context;
}

// Function to extract frames from a video
async function extractVideoFrames(videoElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const frames = [];

    videoElement.currentTime = 0;
    const duration = videoElement.duration;
    const frameInterval = 1; // Extract one frame per second

    while (videoElement.currentTime < duration) {
        await new Promise(resolve => {
            videoElement.onseeked = () => {
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                frames.push(canvas.toDataURL('image/jpeg')); // Save frame as base64
                resolve();
            };
            videoElement.currentTime = Math.min(videoElement.currentTime + frameInterval, duration);
        });
    }

    return frames;
}

// Function to send context to background.js for emotion analysis
function analyzeContext() {
    const context = getContextFromTelegram();

    chrome.runtime.sendMessage({ type: 'analyzeContext', context }, response => {
        if (response && response.analysis) {
            console.log('Analysis results:', response.analysis);

            // Optionally, display results in the Telegram interface
            displayEmotionOverlay(response.analysis);
        }
    });
}

// Function to display emotion analysis overlay
function displayEmotionOverlay(analysis) {
    const overlay = document.createElement('div');
    overlay.className = 'emotion-overlay';
    overlay.style.position = 'absolute';
    overlay.style.background = 'rgba(0, 0, 0, 0.8)';
    overlay.style.color = 'white';
    overlay.style.padding = '10px';
    overlay.style.borderRadius = '5px';
    overlay.style.zIndex = '1000';

    // Create analysis content
    const emotionContent = Object.entries(analysis)
        .map(([modality, results]) => {
            const resultText = results
                .map(({ emotion, score }) => `${emotion}: ${(score * 100).toFixed(2)}%`)
                .join('<br>');
            return `<strong>${modality.toUpperCase()}:</strong><br>${resultText}`;
        })
        .join('<br><br>');

    overlay.innerHTML = `<strong>Emotion Analysis:</strong><br>${emotionContent}`;

    // Append to body
    document.body.appendChild(overlay);

    // Position overlay near the message container
    const messageContainer = document.querySelector('.im_history_messages');
    if (messageContainer) {
        const rect = messageContainer.getBoundingClientRect();
        overlay.style.top = `${rect.bottom + window.scrollY + 10}px`;
        overlay.style.left = `${rect.left}px`;
    }

    // Auto-remove overlay after 5 seconds
    setTimeout(() => overlay.remove(), 5000);
}

// Observe Telegram DOM for new messages
const observer = new MutationObserver(() => {
    analyzeContext();
});

// Start observing Telegram's message container
const messageContainer = document.querySelector('.im_history_messages'); // Adjust selector
if (messageContainer) {
    observer.observe(messageContainer, { childList: true, subtree: true });
    console.log('Observation started for Telegram messages.');
} else {
    console.error('Failed to find Telegram message container.');
}
