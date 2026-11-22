# ❄️ AI-Powered Cold-Chain Logistics — AI Agent & Monitoring Dashboard

An intelligent cold-chain incident management assistant that combines real-time shipment telemetry with Retrieval-Augmented Generation (RAG) to detect temperature breaches, retrieve Standard Operating Procedures (SOPs), and recommend Tier 1 / Tier 2 escalations.

![Dashboard Preview](assets/dashboard_preview.png)
*(Tip: Replace this line with a screenshot or GIF of your Streamlit dashboard)*

---

## 🚀 Key Features

* **Autonomous AI Agent Workflows:** Built with **LangChain** to dynamically decide when to query live shipment databases versus searching unstructured SOP documentation.
* **Temperature Breach Detection & Escalation:** Automatically flags cargo temperature deviations, evaluates delay probabilities, and generates structured **Tier 1 / Tier 2 escalation recommendations**.
* **RAG-Powered SOP Retrieval:** Uses **Hugging Face** embeddings and **Pinecone** vector search to surface exact handling protocols for specific pharmaceuticals, perishables, or transit routes.
* **Live Shipment Telemetry (SQL Server):** Connects to **Microsoft SQL Server** to inspect cargo temperature, GPS location, risk classification, and shipment status via natural-language queries.
* **Interactive Streamlit Dashboard:** Provides an intuitive chat and monitoring interface for logistics operators to investigate incidents without writing SQL.

---

## 🏗️ System Architecture

```text
[ Streamlit UI ] 
       │
       ▼
[ LangChain AI Agent ]
       ├──► [ SQL Server Tool ] ──► Queries live shipment telemetry & risk scores
       │
       └──► [ RAG Retrieval Tool ] ──► Hugging Face Embeddings + Pinecone (SOPs)

## 🛠️ Tech Stack

* Language: Python 3.10+
* AI & Agent Framework: LangChain
* Embeddings & Vector Store: Hugging Face, Pinecone
* Relational Database: Microsoft SQL Server
* Frontend / Dashboard: Streamlit
* Containerization: Docker

## ⚙️ Detailed Setup
For step-by-step SQL Server schema setup and Pinecone indexing guides, see [instructions.md](instructions.md).

