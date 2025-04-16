import React, { useState } from 'react';
import Message from './message';
import { Message as MessageType } from '@/type';
import axios from 'axios';

interface ChatMainProps {
  messages: MessageType[];
}

const ChatMain: React.FC<ChatMainProps> = ({ messages }) => {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:3000/generate', { input });
      setResponse(res.data.output);
    } catch (error) {
      console.error('Error generating response:', error);
    }
  };

  return (
    <div className="chat-main">
      <div className="messages-container">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        {response && <Message key="response" message={{ id: 'response', text: response }} />}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message here..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatMain;
