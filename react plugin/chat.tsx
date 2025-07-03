import React, { ReactNode, useEffect, useRef, useState } from "react";

import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import CloseIcon from "@mui/icons-material/Close";
import PersonOutlineOutlinedIcon from "@mui/icons-material/PersonOutlineOutlined";
import SendOutlinedIcon from "@mui/icons-material/SendOutlined";
import SmartToyOutlinedIcon from "@mui/icons-material/SmartToyOutlined";
import "../styles/style3.css";

const ChatBot_URL = "http://127.0.0.1:5000/rag";
interface Message {
  sender: "user" | "bot";
  text: string;
}

// Typing indicator component
const Typing: React.FC = () => (
  <div className="message bot">
    <div className="icon">
      <SmartToyOutlinedIcon />
    </div>
    <div className="typing-indicator">
      <div></div>
      <div></div>
      <div></div>
    </div>
  </div>
);

const ChatbotWidget: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { sender: "bot", text: "Hey there 👋 How can I help you today?" },
  ]);
  const [input, setInput] = useState("");
  const [showTyping, setShowTyping] = useState(false);
  const scrollBottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (scrollBottom.current) {
      scrollBottom.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const formatMessage = (msg: Message, idx: number) => {
    return (
      <div className={`message ${msg.sender}`} key={idx}>
        <div className="icon">
          {msg.sender === "bot" ? (
            <SmartToyOutlinedIcon />
          ) : (
            <PersonOutlineOutlinedIcon />
          )}
        </div>
        <div className="text">{msg.text}</div>
      </div>
    );
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage: Message = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setShowTyping(true);
    try {
      const response = await fetch(ChatBot_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });

      const data = await response.json();

      // Simulate typing delay
      setTimeout(() => {
        const botMessage: Message = {
          sender: "bot",
          text: data.response || "No reply.",
        };
        setMessages((prev) => [...prev, botMessage]);
        //playNotificationSound();
        setShowTyping(false);
        // setLoading(false);
      }, 150);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error: Unable to reach server." },
      ]);
      setShowTyping(false);
      // setLoading(false);
    }
  };
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <>
      <div className="chat-toggle-btn" onClick={() => setOpen((open) => !open)}>
        {open ? <CloseIcon /> : <ChatBubbleOutlineIcon />}
      </div>

      {open && (
        <div className="chatbot-container">
          <div className="chatbot-header">
            Chatbot
            <button onClick={() => setOpen(false)}>
              <CloseIcon />
            </button>
          </div>
          <div className="chatbot-messages">
            {messages.map((msg, idx) => formatMessage(msg, idx))}
            {showTyping && <Typing />}
            <div ref={scrollBottom} />
          </div>
          <div className="chatbot-input">
            <input
              type="text"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button onClick={sendMessage}>
              <SendOutlinedIcon />
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatbotWidget;
