# 📰 FakeNewsAI — Installation & Setup Guide

Fake News Detection System using **TF-IDF + Stylometric Machine Learning** and **Hugging Face Transformers**.

---

## ⚡ Prerequisites

Make sure the following are installed on your desktop:
- **Python 3.9+**: [Download Python](https://www.python.org/downloads/)
- **Node.js 18+ & npm**: [Download Node.js](https://nodejs.org/)
- **Git**: [Download Git](https://git-scm.com/downloads)

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/FakeNewsAI.git
cd FakeNewsAI
```

### 2. Configure Environment (.env)
Copy `.env.example` to `.env` inside the `server/` folder:

**Windows (PowerShell):**
```powershell
cd server
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cd server
cp .env.example .env
```

*(Optional: Set your `HF_TOKEN` or `LOCAL_HF_MODEL=gpt2` in `server/.env` if using gated Hugging Face models)*

---

### 3. Setup Python Backend (Server)

From inside the `server` directory:

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Run Flask backend server
python app.py
```

**macOS / Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run Flask backend server
python app.py
```
> Server runs at `http://localhost:5000`

---

### 4. Setup React Frontend (Client)

Open a **new terminal window** at the project root:

```bash
cd FakeNewsAI/client

# Install frontend dependencies
npm install

# Run dev server
npm run dev
```
> Client runs at `http://localhost:5173`

---

## 🧪 Verification

Open `http://localhost:5173` in your browser, enter any claim or headline, and view real-time ML classification & Hugging Face analysis!
