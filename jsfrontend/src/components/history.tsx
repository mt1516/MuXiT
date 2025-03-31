import React from 'react';
import { ChatHistoryItem } from '@/type';

interface ChatHistoryProps {
  history: ChatHistoryItem[];
  activeChat: string;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ 
  history, 
  activeChat, 
  onSelectChat, 
  onNewChat,
  onDeleteChat
}) => {
  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    onDeleteChat(id);
  };
  return (
    <div className="chat-history">
      <button className="new-chat-setting-button" onClick={onNewChat}>
        Add New Chat
      </button>

      {/* not implemented for the setting part */}
      <button className="new-chat-setting-button">
        Setting
      </button>
      <div className="history-list">
        {history.map((item) => (
          <div
            key={item.id}
            className={`history-item ${activeChat === item.id ? 'active' : ''}`}
            onClick={() => onSelectChat(item.id)}
          >
            <div className="history-item-title">{item.title}</div>
            <button 
              className="delete-button"
              onClick={(e) => handleDelete(e, item.id)}
            >
              🗑️
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ChatHistory;