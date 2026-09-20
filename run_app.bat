@echo off
if not exist .env copy .env.example .env
streamlit run app.py
pause
