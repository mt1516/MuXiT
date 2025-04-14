import React, { useRef, useState } from 'react';

interface InputProps {
  onSendMessage: (text: string, audioFile?: File) => void;
}

const Input: React.FC<InputProps> = ({ onSendMessage }) => {
  const [inputText, setInputText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim()) {
      onSendMessage(inputText);
      setInputText('');
    }
  };

  const handleAudioClick = () => {
    fileInputRef.current?.click();
  };


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file: any = e.target.files?.[0];
    if (file) {
      onSendMessage(`Audio input: ${file.name}`, file);
      //reset
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        type="text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Provide your idea on music style changing..."
      />
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="audio/*"
        style={{ display: 'none' }}
      />
      <button 
        type="button" 
        className="send-button" 
        onClick={handleAudioClick}
      >
        Audio
      </button>
      <button type="submit" className="send-button">
        Send
      </button>
    </form>
  );
};

export default Input;