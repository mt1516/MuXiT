import React from 'react';
import Message from './message';
import { Message as MessageType } from '@/type';

interface ChatMainProps {
  messages: MessageType[];
}

// message showcase compoment
const ChatMain: React.FC<ChatMainProps> = ({ messages }) => {
  return (
    <div className="chat-main">
      <div className="messages-container">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
      </div>
    </div>
  );
};

export default ChatMain;