import React from "react";
import { Link } from "react-router-dom";
import "./Sidebar.css";

function Sidebar({ isOpen, toggleSidebar }) {
  return (
    <div className={`sidebar ${isOpen ? "expanded" : "collapsed"}`}>
      {/* Toggle button inside sidebar */}
      <button className="sidebar-toggle" onClick={toggleSidebar}>
        {isOpen ? "«" : "»"}
      </button>

      <h2 className={`sidebar-title ${isOpen ? "" : "hidden"}`}>
        Healthcare
      </h2>

      <nav className="sidebar-nav">
        <Link to="/dashboard" className="sidebar-link">
          <span className="icon">🏠</span>
          {isOpen && <span className="text">Home</span>}
        </Link>
        <Link to="/dashboard/chat" className="sidebar-link">
          <span className="icon">💬</span>
          {isOpen && <span className="text">Chat</span>}
        </Link>
        <Link to="/" className="sidebar-link">
          <span className="icon">🚪</span>
          {isOpen && <span className="text">Logout</span>}
        </Link>
      </nav>
    </div>
  );
}

export default Sidebar;
