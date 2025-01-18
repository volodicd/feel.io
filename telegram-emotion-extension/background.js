// background.js

let model;

// Load the TensorFlow.js model
async function loadModel() {
    const modelUrl = chrome.runtime.getURL('model_web/model.json'); // Path to the exported model
    try {
        model = await tf.loadGraphModel(modelUrl);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Failed to load model:', error);
    }
}

// Analyze multiple frames from a video
async function analyzeVideoFrames(frames) {
    if (!model) {
        console.error('Model not loaded');
        return;
    }

    try {
        const results = [];

        // Process each frame and predict emotions
        for (const frame of frames) {
            const imageTensor = preprocessImage(frame);
            const prediction = model.predict(imageTensor);
            const emotionScores = prediction.arraySync()[0];

            // Map emotions to scores
            const emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
            const frameResult = emotions.map((emotion, index) => ({ emotion, score: emotionScores[index] }));
            results.push(frameResult);
        }

        // Aggregate results using voting mechanism
        const aggregatedResults = aggregateEmotionVotes(results);

        return aggregatedResults;
    } catch (error) {
        console.error('Error during video frame analysis:', error);
    }
}

// Helper function to preprocess an image
function preprocessImage(frame) {
    const image = new Image();
    image.src = frame;

    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(image)
            .resizeBilinear([224, 224]) // Resize to match model input
            .toFloat()
            .div(255.0) // Normalize to [0, 1]
            .expandDims(); // Add batch dimension
        return tensor;
    });
}

// Aggregate results from multiple frames using voting
function aggregateEmotionVotes(results) {
    const emotionVotes = {};

    results.forEach(frameResult => {
        frameResult.forEach(({ emotion, score }) => {
            if (!emotionVotes[emotion]) {
                emotionVotes[emotion] = 0;
            }
            emotionVotes[emotion] += score; // Sum up scores for each emotion
        });
    });

    // Normalize scores to percentages
    const totalScore = Object.values(emotionVotes).reduce((sum, value) => sum + value, 0);
    const aggregatedResults = Object.entries(emotionVotes).map(([emotion, score]) => ({
        emotion,
        score: (score / totalScore) * 100
    }));

    return aggregatedResults;
}

// Message listener for communication with content scripts
chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
    if (request.type === 'analyzeContext') {
        const { text, videoFrames, audio } = request.context;
        const analysis = {};

        if (text && text.length > 0) {
            analysis.text = await analyzeText(text);
        }

        if (videoFrames && videoFrames.length > 0) {
            analysis.video = await analyzeVideoFrames(videoFrames);
        }

        // Handle audio analysis if needed (not implemented here)

        sendResponse({ analysis });
    }

    return true; // Keep the message channel open for async responses
});

// Load the model when the extension is installed
chrome.runtime.onInstalled.addListener(() => {
    console.log('Extension installed');
    loadModel();
});
