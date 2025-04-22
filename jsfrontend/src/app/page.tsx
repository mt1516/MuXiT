"use client";

import React, { useState, useEffect } from "react";
import "../styles/Main.css";
import ChatHistory from "@/components/history";
import ChatMain from "@/components/chatbot";
import { Message, ChatHistoryItem } from "@/type";
import Input from "@/components/input";
import { useLocalStorage } from "@/hooks/useLocalStorage";
//import axios from 'axios';

//local storage safe access
const getLocalStorage = (key: string) => {
  if (typeof window !== "undefined") {
    return localStorage.getItem(key);
  }
  return null;
};

const setLocalStorage = (key: string, value: string) => {
  if (typeof window !== "undefined") {
    localStorage.setItem(key, value);
  }
};

const App: React.FC = () => {
  const [isClient, setIsClient] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeChat, setActiveChat] = useState<string>("1");
  const [pendingAudio, setPendingAudio] = useState<File | null>(null);
  const [history, setHistory] = useLocalStorage<ChatHistoryItem[]>(
    "musicChatHistory",
    [
      {
        id: "1",
        title: "New Chat",
        messages: [
          {
            id: "1",
            text: "Hello! Describe your music or upload audio to generate!",
            sender: "ai",
          },
        ],
        createdAt: Date.now(),
      },
    ]
  );

  //init history
  useEffect(() => {
    setIsClient(true);
    const savedHistory = getLocalStorage("musicChatHistory");
    const initialHistory = savedHistory
      ? JSON.parse(savedHistory)
      : [
          {
            id: "1",
            title: "New Chat",
            messages: [
              {
                id: "1",
                text: "Hello! Describe your music or upload audio to generate!",
                sender: "ai",
              },
            ],
            createdAt: Date.now(),
          },
        ];

    setHistory(initialHistory);
    setActiveChat(initialHistory[0].id);
  }, []);

  //load messages
  useEffect(() => {
    const currentChat = history.find((chat) => chat.id === activeChat);
    setMessages(currentChat?.messages || []);
  }, [activeChat, history]);

  //save message to local when change
  useEffect(() => {
    const cleanHistory = history.map((chat) => ({
      ...chat,
      messages: chat.messages.map((msg) => ({
        ...msg,
        audioUrl: undefined,
      })),
    }));
    setLocalStorage("musicChatHistory", JSON.stringify(cleanHistory));

    // console log checkers
    console.log("Current history state:", history);
    console.log(
      "Current localStorage:",
      JSON.parse(localStorage.getItem("musicChatHistory") || "null")
    );
  }, [history]);

  // recreate audio url
  useEffect(() => {
    const messagesWithUrls = messages.map((msg) => {
      if (msg.audioData && !msg.audioUrl) {
        try {
          const byteString = atob(msg.audioData.split(",")[1]);
          const ab = new ArrayBuffer(byteString.length);
          const ia = new Uint8Array(ab);
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
          }
          const blob = new Blob([ab], { type: "audio/wav" });
          return {
            ...msg,
            audioUrl: URL.createObjectURL(blob),
          };
        } catch (e) {
          console.error("Error recreating audio:", e);
          return msg;
        }
      }
      return msg;
    });

    if (JSON.stringify(messagesWithUrls) !== JSON.stringify(messages)) {
      setMessages(messagesWithUrls);
    }

    return () => {
      messagesWithUrls.forEach((msg) => {
        if (msg.audioUrl) URL.revokeObjectURL(msg.audioUrl);
      });
    };
  }, [messages]);

  //update one commucation box -> prev + id + chat
  const updateHistory = (message: Message) => {
    setHistory((prev) => {
      const updatedHistory = prev.map((chat) =>
        chat.id === activeChat // matching id
          ? {
              ...chat,
              messages: [...chat.messages, message], // chat messages + new message
              title:
                chat.messages.length === 1 && message.sender === "user"
                  ? message.text.slice(0, 30) +
                    (message.text.length > 30 ? "..." : "")
                  : chat.title, //update chat title
            }
          : chat
      );

      window.localStorage.setItem(
        "musicChatHistory",
        JSON.stringify(updatedHistory)
      );
      return updatedHistory;
    });
    console.log("new message being added:", message);
    // this console show a corret localstorage updated to musicchathistory
    //why??? the localstorage cannot keep the user sent message
  };

  const handleSendMessage = async (
    text: string,
    duration: number,
    audioFile?: File
  ) => {
    // Create combined message
    const newUserMessage: Message = {
      id: Date.now().toString(),
      text: text || (audioFile ? "Audio input" : ""),
      sender: "user",
      isAudio: !!audioFile,
      audioFile: audioFile,
    };

    console.log("Before updateHistory - user message:", newUserMessage);
    updateHistory(newUserMessage);

    setPendingAudio(null); // Clear pending audio after send

    // feedback from systems, it is correct for now.
    try {
      const formData = new FormData();
      formData.append("prompt", text || "Generate music");
      formData.append("duration", String(duration));

      if (audioFile) {
        formData.append("audio_input", audioFile);
      }

      //get current history before API call to prevent race conditions
      const currentHistory = JSON.parse(
        localStorage.getItem("musicChatHistory") || "[]"
      );
      const currentChat = currentHistory.find(
        (c: ChatHistoryItem) => c.id === activeChat
      );

      const response = await fetch("http://localhost:8000/generate-music/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error details:", errorData);
        throw new Error(errorData.detail || "Failed to generate music");
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      const aiResponse: Message = {
        id: Date.now().toString(),
        text: "Generated music",
        sender: "ai",
        audioUrl: audioUrl,
      };

      updateHistory(aiResponse);
      console.log("AI response added:", aiResponse);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: "Error generating music",
        sender: "ai",
        error: true,
      };
      updateHistory(errorMessage);
    }
  };

  //manage history chat list use set history, if manage messages in the history, use updatehistory
  const startNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat: ChatHistoryItem = {
      id: newChatId,
      title: "New Chat",
      messages: [
        {
          id: "1",
          text: "Hello! Describe your music or upload audio to generate!",
          sender: "ai",
        },
      ],
      createdAt: Date.now(),
    };

    setHistory((prev) => [newChat, ...prev]);
    setActiveChat(newChatId);
    setPendingAudio(null);
  };

  // deleting a histroy chat box
  const deleteChat = (chatId: string) => {
    setHistory((prev) => prev.filter((chat) => chat.id !== chatId));
    if (chatId === activeChat) {
      setActiveChat(
        (prev) => history.find((chat) => chat.id !== prev)?.id || ""
      );
    }
  };

  return (
    <div className="app">
      {isClient ? (
        <>
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
        </>
      ) : (
        <div>Loading...</div>
      )}
    </div>
  );
};

export default App;
