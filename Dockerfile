FROM python:3.11-slim
WORKDIR /app
COPY requirements_streamlit_bonus.txt ./requirements_streamlit_bonus.txt
RUN pip install --no-cache-dir -r requirements_streamlit_bonus.txt
COPY . .
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "streamlit_app_bonus_complete.py", "--server.address=0.0.0.0", "--server.port=8501"]
