# 1️⃣ Base Image
FROM python:3.11-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Install dependencies first (for Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy the rest of your app
COPY . .

# 5️⃣ Streamlit runs best on port 8501 by default, and Lightsail expects that.
EXPOSE 8501

# 6️⃣ Optional but recommended: add a simple health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 7️⃣ Launch Streamlit
# Lightsail doesn’t need proxy workarounds — direct port exposure works fine.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]

