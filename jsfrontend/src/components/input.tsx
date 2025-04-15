import React, { useState, useRef } from 'react';

interface InputProps {
  onSendMessage: (text: string, audioFile?: File) => void;
  pendingAudio: File | null;
  setPendingAudio: (file: File | null) => void;
}

const Input: React.FC<InputProps> = ({ 
  onSendMessage, 
  pendingAudio, 
  setPendingAudio 
}) => {
  const [inputText, setInputText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() || pendingAudio) {
      onSendMessage(inputText, pendingAudio || undefined);
      setInputText('');
    }
  };

  const handleAudioUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPendingAudio(file);
    }
  };

  const removePendingAudio = () => {
    setPendingAudio(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        type="text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Describe your music..."
      />
      
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleAudioUpload}
        accept="audio/*"
        style={{ display: 'none' }}
      />
      
      <div className="audio-controls">
        <button 
          type="button" 
          className={`audio-button ${pendingAudio ? 'active' : ''}`}
          onClick={() => fileInputRef.current?.click()}
        >
          {pendingAudio ? 'Uploaded' : 'Add Audio'}
        </button>
        
        {pendingAudio && (
          <div className="audio-preview">
            <span>{pendingAudio.name}</span>
            <button 
              type="button" 
              className="remove-audio"
              onClick={removePendingAudio}
            >
              ×
            </button>
          </div>
        )}
      </div>
      
      <button 
        type="submit" 
        className="send-button"
        disabled={!inputText && !pendingAudio}
      >
        Send
      </button>
    </form>
  );
};

export default Input;