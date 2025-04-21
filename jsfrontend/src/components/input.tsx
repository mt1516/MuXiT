import React, { useState, useRef } from "react";

interface InputProps {
  onSendMessage: (text: string, duration: number, audioFile?: File) => void;
  pendingAudio: File | null;
  setPendingAudio: (file: File | null) => void;
}

const Input: React.FC<InputProps> = ({
  onSendMessage,
  pendingAudio,

  setPendingAudio,
}) => {
  const [inputText, setInputText] = useState("");
  const [duration, setDuration] = useState(30);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() || pendingAudio) {
      onSendMessage(inputText, duration, pendingAudio || undefined);
      setInputText("");
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
      fileInputRef.current.value = "";
    }
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        className="chat-input-text"
        type="text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Describe your music..."
      />

      <div className="duration">
        <span className="duration-label">Duration (seconds)</span>
        <input
          className="chat-input-duration"
          type="number"
          min="1"
          max="300"
          value={duration}
          onChange={(e) =>
            setDuration(Math.min(300, Math.max(1, Number(e.target.value))))
          }
        />
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleAudioUpload}
        accept="audio/*"
        style={{ display: "none" }}
      />

      <div className="audio-controls">
        <button
          type="button"
          className={`audio-button ${pendingAudio ? "active" : ""}`}
          onClick={() => fileInputRef.current?.click()}
        >
          {pendingAudio ? "Uploaded" : "Add Audio"}
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
