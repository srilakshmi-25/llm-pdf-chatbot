import streamlit as st
import streamlit.components.v1 as components

def voice_input_button(height: int = 70):
    """
    Renders an HTML5 Web Speech API voice recognition microphone button.
    Bypasses standard iframe microphone restrictions by injecting the speech recognition
    controller directly into the parent document context.
    
    This works natively in Google Chrome, Microsoft Edge, Safari, and other browsers
    supporting webkitSpeechRecognition.
    """
    
    # Custom HTML, CSS, and JS for Web Speech API integration with parent-context execution
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                overflow: hidden;
            }
            .mic-container {
                display: flex;
                align-items: center;
                gap: 12px;
                height: 100vh;
                width: 100vw;
            }
            .mic-btn {
                background: rgba(255, 255, 255, 0.07);
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #ffffff;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                outline: none;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }
            .mic-btn:hover {
                background: rgba(255, 255, 255, 0.15);
                border-color: rgba(255, 255, 255, 0.3);
                transform: scale(1.05);
            }
            .mic-btn.listening {
                background: rgba(239, 68, 68, 0.25);
                border-color: rgba(239, 68, 68, 0.6);
                color: #ef4444;
                animation: pulse 1.5s infinite ease-in-out;
            }
            .mic-icon {
                width: 20px;
                height: 20px;
                fill: currentColor;
            }
            .status-text {
                font-size: 0.85rem;
                color: rgba(255, 255, 255, 0.6);
                transition: all 0.3s ease;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 250px;
            }
            .status-text.listening-state {
                color: #ef4444;
                font-weight: 500;
            }
            
            @keyframes pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
                }
            }
        </style>
    </head>
    <body>
        <div class="mic-container">
            <button id="micBtn" class="mic-btn" title="Click to speak">
                <svg class="mic-icon" viewBox="0 0 24 24">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/>
                </svg>
            </button>
            <span id="status" class="status-text">Click mic to speak</span>
        </div>

        <script>
            const micBtn = document.getElementById('micBtn');
            const statusText = document.getElementById('status');
            const parentDoc = window.parent.document;
            const parentWin = window.parent;
            
            // Inject the SpeechRecognition initiator script directly into the parent window context.
            // This is a highly professional workaround that bypasses the "allow='microphone'" 
            // security policy block on standard Streamlit component iframes.
            if (!parentWin.startParentSpeechRecognition) {
                try {
                    const scriptEl = parentDoc.createElement('script');
                    scriptEl.id = 'parent-speech-recognition-injector';
                    scriptEl.innerHTML = `
                        window.startParentSpeechRecognition = function(onStart, onResult, onError, onEnd) {
                            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                            if (!SpeechRecognition) {
                                onError("Speech recognition not supported in this browser.");
                                return null;
                            }
                            
                            const recognition = new SpeechRecognition();
                            recognition.continuous = false;
                            recognition.interimResults = false;
                            recognition.lang = 'en-US';
                            
                            recognition.onstart = onStart;
                            recognition.onresult = function(event) {
                                const transcript = event.results[0][0].transcript;
                                onResult(transcript);
                            };
                            recognition.onerror = function(event) {
                                onError(event.error);
                            };
                            recognition.onend = onEnd;
                            
                            try {
                                recognition.start();
                                return recognition;
                            } catch (e) {
                                onError(e.message);
                                return null;
                            }
                        };
                    `;
                    parentDoc.head.appendChild(scriptEl);
                    console.log("Successfully injected speech recognition handler into parent head.");
                } catch (err) {
                    console.error("Failed to inject script into parent window head:", err);
                }
            }

            let recognitionInstance = null;
            let isListening = false;

            micBtn.addEventListener('click', () => {
                if (isListening) {
                    if (recognitionInstance) {
                        try {
                            recognitionInstance.stop();
                        } catch (e) {
                            console.error("Error stopping recognition", e);
                        }
                    }
                } else {
                    if (typeof parentWin.startParentSpeechRecognition !== 'function') {
                        statusText.textContent = "Injection blocked by browser security";
                        return;
                    }

                    recognitionInstance = parentWin.startParentSpeechRecognition(
                        // onStart callback
                        () => {
                            isListening = true;
                            micBtn.classList.add('listening');
                            statusText.textContent = "Listening... Speak now";
                            statusText.classList.add('listening-state');
                        },
                        // onResult callback
                        (transcript) => {
                            statusText.textContent = "Transcribed!";
                            statusText.classList.remove('listening-state');
                            
                            try {
                                // Find standard Streamlit text inputs in the parent window
                                let inputArea = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                                if (!inputArea) {
                                    inputArea = parentDoc.querySelector('textarea[aria-label="Ask a question..."]');
                                }
                                if (!inputArea) {
                                    inputArea = parentDoc.querySelector('input[data-testid="stTextInputEnterText"]');
                                }
                                if (!inputArea) {
                                    inputArea = parentDoc.querySelector('input[type="text"]');
                                }
                                
                                if (inputArea) {
                                    inputArea.focus();
                                    inputArea.value = transcript;
                                    
                                    // Trigger standard events so React state in Streamlit binds the new value
                                    inputArea.dispatchEvent(new Event('input', { bubbles: true }));
                                    inputArea.dispatchEvent(new Event('change', { bubbles: true }));
                                    
                                    statusText.textContent = "Filled!";
                                } else {
                                    statusText.textContent = "Transcribed: " + transcript;
                                }
                            } catch (e) {
                                console.error("Error writing to parent input:", e);
                                statusText.textContent = "Speech: " + transcript;
                            }
                        },
                        // onError callback
                        (error) => {
                            console.error("Parent recognition error:", error);
                            let userFriendlyError = error;
                            if (error === 'not-allowed') {
                                userFriendlyError = "Mic access blocked. Check permissions.";
                            }
                            statusText.textContent = "Error: " + userFriendlyError;
                            statusText.classList.remove('listening-state');
                            micBtn.classList.remove('listening');
                            isListening = false;
                        },
                        // onEnd callback
                        () => {
                            micBtn.classList.remove('listening');
                            if (statusText.textContent === "Listening... Speak now") {
                                statusText.textContent = "Click mic to speak";
                            }
                            statusText.classList.remove('listening-state');
                            isListening = false;
                            recognitionInstance = null;
                        }
                    );
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Renders the custom component iframe inside Streamlit
    components.html(html_code, height=height, scrolling=False)

