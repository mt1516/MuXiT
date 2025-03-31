import React from 'react';
import { Message as MessageType } from '@/type';

interface MessageProps {
  message: MessageType;
}

const Message: React.FC<MessageProps> = ({ message }) => {
  return (
    <div className={`message ${message.sender}`}>
      <div className="message-content">
        {message.isAudio && (
          <div className="audio-indicator">🎤 Audio Message</div>
        )}
        <div className="message-text">{message.text}</div>
      </div>
    </div>
  );
};

export default Message;