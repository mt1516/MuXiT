import React, { useState } from 'react';

interface InputProps {
  onSendMessage: (text: string, isAudio?: boolean) => void;
}

const Input: React.FC<InputProps> = ({ onSendMessage }) => {
  const [inputText, setInputText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim()) {
      onSendMessage(inputText);
      setInputText('');
    }
  };

  // dummy text to check out the response
  const handleAudioInput = () => {
    const dummyAudioText = "dummy audio";
    onSendMessage(dummyAudioText, true);
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        type="text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Provide your idea on music style changing..."
      />
      <button type="button" className="send-button" onClick={handleAudioInput}>
        Audio
      </button>
      <button type="submit" className="send-button">
        Send
      </button>
    </form>
  );
};

export default Input;