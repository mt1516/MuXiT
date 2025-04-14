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

  const generateMusic = async (prompt: string, audioFile?: File) => {
    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('duration', '30'); //default duration?
    
    if (audioFile) {
      formData.append('audio_input', audioFile);
    }
    try {
      const response = await fetch('http://localhost:8000/generate-music/', {
        method: 'POST',
        body: formData,
      });

      // catch error handler
      if (!response.ok) {
        throw new Error('Failed to generate music');
      }
      const audioBlob = await response.blob();
      return URL.createObjectURL(audioBlob);
    } catch (error) {
      console.error('Error generating music:', error);
      throw error;
    }
  };

  // handle message detect audio
  const handleSendMessage = async (text: string, audioFile?: File) => {
    const newUserMessage: Message = {
      id: Date.now().toString(),
      text: audioFile ? `Audio input: ${audioFile.name}` : text,
      sender: 'user',
      isAudio: !!audioFile,
    };

    // update history and message
    setMessages((prev) => [...prev, newUserMessage]);
    updateHistory(newUserMessage);

    // loading if the above tested correct
    /* const loadingMessage: Message = {
      id: `loading-${Date.now()}`,
      text: 'Generating music...',
      sender: 'ai',
      isLoading: true,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, loadingMessage]);
    updateHistory(loadingMessage); */

    try{
      const audioUrl = await generateMusic(text, audioFile);

      const aiResponse: Message = {
        id: Date.now().toString(),
        text: 'Here is your generated music:',
        sender: 'ai',
        audioUrl: audioUrl,
      };

      setMessages(prev => [...prev, aiResponse]);
      updateHistory(aiResponse);

    } catch (error) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: 'Sorry, there was an error generating the music. Please try again.',
        sender: 'ai',
        error: true,
      };

      setMessages(prev => [...prev, errorMessage]);
      updateHistory(errorMessage);
    }
    

    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `System timeout.`,
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
