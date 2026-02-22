const ELEVENLABS_API_KEY = "sk_63cac0138c6cdafad99a25e4885de77b9d0126bbc8961285";
const ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // Default voice (Rachel)

async function readText(text) {
    if (!text || text.trim() === '') {
        alert('No text to read!');
        return;
    }

    const readButton = document.getElementById('readItBtn');
    if (readButton) {
        readButton.disabled = true;
        readButton.textContent = 'Reading...';
    }

    try {
        const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`, {
            method: 'POST',
            headers: {
                'Accept': 'audio/mpeg',
                'Content-Type': 'application/json',
                'xi-api-key': ELEVENLABS_API_KEY
            },
            body: JSON.stringify({
                text: text,
                model_id: 'eleven_monolingual_v1',
                voice_settings: {
                    stability: 0.5,
                    similarity_boost: 0.5
                }
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            if (readButton) {
                readButton.disabled = false;
                readButton.textContent = 'Read It';
            }
        };

        audio.onerror = () => {
            if (readButton) {
                readButton.disabled = false;
                readButton.textContent = 'Read It';
            }
        };

        await audio.play();

    } catch (error) {
        console.error('Text-to-speech error:', error);
        alert('Failed to read text. Please try again.');
        if (readButton) {
            readButton.disabled = false;
            readButton.textContent = 'Read It';
        }
    }
}

// Function to get text from your input/textarea
function handleReadIt() {
    // Adjust the selector based on your actual input element
    const textElement = document.getElementById('textInput') || 
                        document.querySelector('textarea') || 
                        document.querySelector('input[type="text"]');
    
    if (textElement) {
        readText(textElement.value);
    } else {
        alert('No text input found!');
    }
}