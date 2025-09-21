# 🚗 EV Sales Predictor – India

An interactive **EV Sales Prediction Web App** built with **Flask**, **Scikit-learn**, and **Pandas**, deployed on **Render**, and version-controlled with **GitHub**. This project predicts EV sales based on state, date, and vehicle details using a trained **Random Forest model**.

---

## 📊 About the Data
- Dataset: **EV_Dataset.csv**  
- Contains historical EV sales data across Indian states.  
- Used to:
  - Train an ML model for EV sales prediction  
  - Demonstrate data-driven applications in the automotive sector  

---

## ⚙️ Why These Technologies?
- **Flask** → Lightweight backend framework for Python-based web apps.  
- **Pandas** → For data handling and preprocessing.  
- **Scikit-learn** → Trained Random Forest prediction model (`rf_model.pkl`).  
- **Joblib** → Save/load trained ML models.  
- **Render** → Cloud platform for deployment with GitHub integration.  

---

## 🚀 Complete Process (From Start to Finish)
1. **Create Project Folder** Example: `D:\ev_sales_app`  
2. **Create & Activate Virtual Environment**  
   - Windows: `python -m venv venv` → `venv\Scripts\activate`  
   - macOS/Linux: `python3 -m venv venv` → `source venv/bin/activate`  
3. **Install Dependencies**  
   ```bash
   pip install flask pandas numpy scikit-learn joblib
   pip freeze > requirements.txt
   ```  
4. **Build Flask Application**  
   - `app.py` contains routes: `/` → Home page with prediction form, `/result` → Display predicted EV sales  
   - Used: Flask, Scikit-learn, Pandas  
5. **Add Project Files**  
   - `app.py`, `templates/`, `static/`, `requirements.txt`, `Procfile`, `rf_model.pkl`, `data/EV_Dataset.csv`  
6. **Initialize Git & Push to GitHub**  
   ```bash
   git init
   git remote add origin https://github.com/afridmd12/ev_sales_app.git
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```  
7. **Deploy on Render**  
   - Go to [Render](https://render.com), create a Web Service, connect GitHub repository  
   - Build Command: `pip install -r requirements.txt`  
   - Start Command: `gunicorn app:app`  
   - Deploy 🚀  

---

## ✅ Features
- Predict EV sales based on state, date, and vehicle type  
- Trained Random Forest model for accurate predictions  
- Simple and interactive web interface  
- Hosted on Render  

---

## 💻 Run Locally
1. Clone repo: `git clone https://github.com/afridmd12/ev_sales_app.git`  
2. Navigate: `cd ev_sales_app`  
3. Create virtual environment & activate  
4. Install dependencies: `pip install -r requirements.txt`  
5. Run app: `python app.py`  
6. Open in browser: `http://127.0.0.1:5000`  

---

## 🌐 Deployment
- Hosted on **Render**  
- Live URL: [https://ev-sales-app.onrender.com](https://ev-sales-app.onrender.com)  

---

## 📝 Author
👤 Mohammed Afrid  
📌 GitHub: [afridmd12](https://github.com/afridmd12/ev_sales_app.git)
