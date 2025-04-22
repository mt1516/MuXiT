"use client";

import React, { useState, useEffect } from 'react';
import '../styles/Main.css';
import ChatHistory from '@/components/history';
import ChatMain from '@/components/chatbot';
import { Message, ChatHistoryItem } from '@/type';
import Input from '@/components/input';
//import axios from 'axios';

const LOCAL_STORAGE_KEY = 'musicChatHistory';

const App: React.FC = () => {
  //load from localStorage on init render
  const loadFromLocalStorage = (): ChatHistoryItem[] => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch (e) {
          console.error('LocalStorage data parse error', e);
        }
      }
    }
    //init if no localStorage data
    return [
      {
        id: '1',
        title: 'New Chat',
        messages: [
          {
            id: '1',
            text: 'Hello! Describe your music or upload audio to generate!',
            sender: 'ai',
          },
        ],
      },
    ];
  };

  const [messages, setMessages] = useState<Message[]>([]);
  const [history, setHistory] = useState<ChatHistoryItem[]>(loadFromLocalStorage);
  const [activeChat, setActiveChat] = useState<string>('1');
  const [pendingAudio, setPendingAudio] = useState<File | null>(null);

  //save to localStorage when history change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(history));
    }
  }, [history]);

  //load messages when activeChat change
  useEffect(() => {
    const currentChat = history.find(chat => chat.id === activeChat);
    if (currentChat) {
      setMessages(currentChat.messages);
    } else {
      setMessages([]);
    }
  }, [activeChat, history]);

  // update history with new message and title
  const updateHistory = (message: Message) => {
    setHistory(prev => {
      const updatedHistory = prev.map(chat =>
        chat.id === activeChat
          ? {
              ...chat,
              messages: [...chat.messages, message],
              title:
                chat.messages.length === 1 && message.sender === 'user'
                  ? message.text.slice(0, 30) + (message.text.length > 30 ? '...' : '')
                  : chat.title,
            }
          : chat
      );
      return updatedHistory;
    });
  };

  // main commucation
  const handleSendMessage = async (text: string, duration: number, audioFile?: File) => {
    // Create combined message
    const newUserMessage: Message = {
      id: Date.now().toString(),
      text: text || (audioFile ? "Audio input" : ""),
      sender: 'user',
      isAudio: !!audioFile,
      audioFile: audioFile,
    };

    setMessages(prev => [...prev, newUserMessage]);
    updateHistory(newUserMessage);
    setPendingAudio(null); // Clear pending audio after send

    try {
      const formData = new FormData();
      formData.append('prompt', text || "Generate music");
      formData.append('duration', String(30));
      
      if (audioFile) {
        formData.append('audio_input', audioFile);
      }

      const response = await fetch('http://localhost:8000/generate-music/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error details:', errorData);
        throw new Error(errorData.detail || 'Failed to generate music');
      }
      
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // mini LLM (disabled for now)
      try {
        //const aiMessage = await axios.post('http://localhost:8001/generate', { text });
        const aiResponse: Message = {
          id: Date.now().toString(),
          text: 'Generated music',
          //text: text ? aiMessage.data.text : 'Generated music',
          sender: 'ai',
          audioUrl: audioUrl,
        };

        setMessages(prev => [...prev, aiResponse]);
        updateHistory(aiResponse);
      } catch (error) {
        console.error('Error generating AI response:', error);
        const errorMessage: Message = {
          id: Date.now().toString(),
          text: 'Error generating AI response',
          sender: 'ai',
          audioUrl: audioUrl,
        };
        setMessages(prev => [...prev, errorMessage]);
        updateHistory(errorMessage);
      }

    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: 'Error generating music',
        sender: 'ai',
        error: true,
      };
      setMessages(prev => [...prev, errorMessage]);
      updateHistory(errorMessage);
    }
  };

  // getting new chat history
  const startNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat: ChatHistoryItem = {
      id: newChatId,
      title: 'New Chat',
      messages: [
        {
          id: '1',
          text: 'Hello! Describe your music or upload audio to generate!',
          sender: 'ai',
        },
      ],
    };

    setHistory(prev => [newChat, ...prev]);
    setActiveChat(newChatId);
    setPendingAudio(null);
  };

  // delete chat history
  const deleteChat = (chatId: string) => {
    setHistory(prev => {
      const updatedHistory = prev.filter(chat => chat.id !== chatId);
      return updatedHistory;
    });
    
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
        <Input 
          onSendMessage={handleSendMessage}
          pendingAudio={pendingAudio}
          setPendingAudio={setPendingAudio}
        />
      </div>
    </div>
  );
};

export default App;