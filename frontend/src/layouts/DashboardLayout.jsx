import React, { useState } from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "../components/Sidebar";

function DashboardLayout() {
  const [isOpen, setIsOpen] = useState(true);

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div style={{ display: "flex" }}>
      <Sidebar isOpen={isOpen} toggleSidebar={toggleSidebar} />
      <div
        style={{
          flex: 1,
          marginLeft: isOpen ? "220px" : "70px", // auto adjust
          padding: "20px",
          transition: "margin-left 0.3s ease",
        }}
      >
        <Outlet /> {/* Home or Chat loads here */}
      </div>
    </div>
  );
}

export default DashboardLayout;
