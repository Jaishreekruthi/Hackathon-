import React, { useState } from "react";

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const newMessages = [...messages, { sender: "user", text: input }];
    setMessages(newMessages);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();
      setIsTyping(false);

      // Add bot reply
      setMessages([...newMessages, { sender: "bot", text: data.response }]);
    } catch (error) {
      setIsTyping(false);
      setMessages([
        ...newMessages,
        { sender: "bot", text: "⚠️ Error connecting to server" },
      ]);
    }
  };

  return (
    <div
      style={{
        maxWidth: "800px",
        margin: "20px auto",
        padding: "20px",
        background: "#fdf6e3", // cream background
        borderRadius: "12px",
        boxShadow: "0px 4px 12px rgba(0,0,0,0.1)",
      }}
    >
      <h2 style={{ marginBottom: "15px", color: "#1d3557" }}>
        💬 Healthcare Chatbot
      </h2>

      {/* Chat window */}
      <div
        style={{
          border: "1px solid #ddd",
          borderRadius: "8px",
          padding: "15px",
          height: "400px",
          overflowY: "auto",
          background: "white",
          marginBottom: "10px",
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              textAlign: msg.sender === "user" ? "right" : "left",
              margin: "8px 0",
            }}
          >
            <span
              style={{
                background:
                  msg.sender === "user" ? "#e0e0e0" : "#a8dadc", // user = gray, bot = teal
                color: "#000",
                padding: "10px 14px",
                borderRadius: "18px",
                display: "inline-block",
                maxWidth: "70%",
                wordWrap: "break-word",
                fontSize: "15px",
                fontWeight: msg.sender === "bot" ? "500" : "normal",
              }}
            >
              {msg.text}
            </span>
          </div>
        ))}

        {/* Typing indicator */}
        {isTyping && (
          <div style={{ textAlign: "left", margin: "6px 0", color: "#1d3557" }}>
            <span
              style={{
                background: "#a8dadc",
                padding: "8px 12px",
                borderRadius: "18px",
                display: "inline-block",
                fontSize: "14px",
              }}
            >
              Bot is typing<span className="dot">.</span>
              <span className="dot">.</span>
              <span className="dot">.</span>
            </span>
          </div>
        )}
      </div>

      {/* Input area */}
      <div style={{ display: "flex", gap: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "8px",
            border: "1px solid #ccc",
          }}
        />
        <button
          onClick={sendMessage}
          style={{
            padding: "12px 24px",
            background: "#1d3557",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "600",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default Chat;
