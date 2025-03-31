"use client";

import React, { useState, useEffect } from 'react';
import '../styles/Main.css';
import ChatHistory from '@/components/history';
import ChatMain from '@/components/chatbot';
import { Message, ChatHistoryItem } from '@/type';
import Input from '@/components/input';

const App: React.FC = () => {
  // send message with default prompt in real case
  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<ChatHistoryItem[]>(() => {
    // Initialize with one empty chat if no history exists
    return [
      {
        id: '1',
        title: 'New Chat',
        timestamp: new Date(),
        messages: [
          {
            id: '1',
            text: 'Hello! Input your thoughts or an audio to get a great idea!',
            sender: 'ai',
            timestamp: new Date(),
          },
        ],
      },
    ];
  });
  const [activeChat, setActiveChat] = useState<string>('1');

  // load message for finding history
  useEffect(() => {
    const currentChat = history.find(chat => chat.id === activeChat);
    if (currentChat) {
      setMessages(currentChat.messages);
    } else {
      setMessages([]);
    }
  }, [activeChat, history]);

  const handleSendMessage = (text: string, isAudio: boolean = false) => {
    const newUserMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: 'user',
      isAudio,
    };

    setMessages((prev) => [...prev, newUserMessage]);
    updateHistory(newUserMessage);

    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `dummy AI response to: "${text}"`,
        sender: 'ai',
      };
      setMessages(prev => [...prev, aiResponse]);
      updateHistory(aiResponse);
    }, 1000);
  };

  const updateHistory = (message: Message) => {
    setHistory(prev =>
      prev.map(chat =>
        chat.id === activeChat
          ? {
              ...chat,
              messages: [...chat.messages, message],
              // Update title
              title:
                chat.messages.length === 1 && message.sender === 'user'
                  ? message.text.slice(0, 30) + (message.text.length > 30 ? '...' : '')
                  : chat.title,
            }
          : chat
      )
    );
  };

  const startNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat: ChatHistoryItem = {
      id: newChatId,
      title: 'New Chat',
      messages: [
        {
          id: '1',
          text: 'Hello! Input your thoughts or an audio to get a great idea!',
          sender: 'ai',
        },
      ],
    };

    setHistory(prev => [newChat, ...prev]);
    setActiveChat(newChatId);
  };

  const deleteChat = (chatId: string) => {
    setHistory(prev => prev.filter(chat => chat.id !== chatId));
    // If we're deleting the active chat, switch to the first available chat
    if (chatId === activeChat) {
      setActiveChat(prev => {
        const remainingChats = history.filter(chat => chat.id !== chatId);
        return remainingChats.length > 0 ? remainingChats[0].id : '';
      });
    }
  };

  return (
    <div className="app">
      <ChatHistory 
        history={history} 
        activeChat={activeChat} 
        onSelectChat={setActiveChat} 
        onNewChat={startNewChat}
        onDeleteChat={deleteChat}
      />
      <div className="chat-container">
        <ChatMain messages={messages} />
        <Input onSendMessage={handleSendMessage} />
      </div>
    </div>
  );
};

export default App;
