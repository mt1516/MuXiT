import React from 'react';
import { Message as MessageType } from '@/type';

interface MessageProps {
  message: MessageType;
}

const Message: React.FC<MessageProps> = ({ message }) => {
  // Create audio URL for user-uploaded audio
  const userAudioUrl = message.audioFile ? URL.createObjectURL(message.audioFile) : null;

  return (
    <div className={`message ${message.sender}`}>
      <div className="message-content">
        {/* user message with audio */}
        {message.sender === 'user' && message.audioFile && (
          <div>
            {message.text && <div className="message-text">{message.text}</div>}
            <div>
              <audio controls src={userAudioUrl || undefined}>
                Your browser does not support audio.
              </audio>
              <span className="audio-label">Your audio</span>
            </div>
          </div>
        )}

        {/* user text only */}
        {message.sender === 'user' && !message.audioFile && (
          <div className="message-text">{message.text}</div>
        )}

        {/* AI response generated audio */}
        {message.sender === 'ai' && message.audioUrl && (
          <div>
            <div className="message-text">{message.text}</div>
            <div>
              <audio controls src={message.audioUrl}>
                Your browser does not support audio.
              </audio>
              <span className="audio-label">Generated music</span>
            </div>
          </div>
        )}

        {/* AI text only */}
        {message.sender === 'ai' && !message.audioUrl && (
          <div className="message-text">{message.text}</div>
        )}

        {message.error && (
          <div className="error-message">⚠️ Error generating music</div>
        )}
      </div>
    </div>
  );
};

export default Message;