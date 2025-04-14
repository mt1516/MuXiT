import React from 'react';
import { Message as MessageType } from '@/type';

interface MessageProps {
  message: MessageType;
}

const Message: React.FC<MessageProps> = ({ message }) => {
  return (
    <div className={`message ${message.sender}`}>
      <div className="message-content">
        {message.isAudio && message.sender === 'user' && (
          <div className="audio-indicator">🎤 Audio Message</div>
        )}
        {message.audioUrl && (
          <div className="generated-audio">
            <audio controls src={message.audioUrl}>
              Your browser does not support the audio.
            </audio>
          </div>
        )}
        {message.error && (
          <div className="error-message">Error generating music</div>
        )}
        <div className="message-text">{message.text}</div>
      </div>
    </div>
  );
};

export default Message;