export interface Message {
    audioFile?: any;
    id: string;
    text: string;
    sender: 'user' | 'ai';
    isAudio?: boolean;
    audioUrl?: string; 
    //isLoading?: boolean;
    error?: boolean; 
  }
  
  export interface ChatHistoryItem {
    id: string;
    title: string;
    messages: Message[];
  }