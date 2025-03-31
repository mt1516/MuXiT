export interface Message {
    id: string;
    text: string;
    sender: 'user' | 'ai';
    isAudio?: boolean;
  }
  
  export interface ChatHistoryItem {
    id: string;
    title: string;
    messages: Message[];
  }