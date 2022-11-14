## base image
FROM python:3.10-slim-buster

ENV PYTHONBUFFERED True

ENV APP_HOME /app

#move into the app working directory
WORKDIR /app

#Expose port 8080 on the container
EXPOSE 8080

#upgrade pip to install any issues with version mismatch
RUN python3 -m pip install --upgrade pip
#copy requirements.txt into requirements.txt in the /app directory
COPY requirements.txt requirements.txt 
#install all the required libraries from requirements.txt
RUN pip install -r requirements.txt

#copy soft-skill summaries data and soft_skills_app.py
COPY . ./

CMD streamlit run --server.port 8080 --server.enableCORS false app.py
