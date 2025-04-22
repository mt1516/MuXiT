export interface Message {
    audioFile?: any;
    id: string;
    text: string;
    sender: 'user' | 'ai';
    isAudio?: boolean;
    audioUrl?: string; 
    audioData?: string; //really need this?
    error?: boolean; 
  }
  
  export interface ChatHistoryItem {
    id: string;
    title: string;
    messages: Message[];
    createdAt?: number;
  }