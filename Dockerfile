FROM nikolaik/python-nodejs:latest

WORKDIR /FASTAPI

COPY ./la-pret-fastApi/loan_data.csv .
COPY ./la-pret-fastApi/main.py .
COPY ./la-pret-fastApi/requirements.txt .

RUN pip install -r ./requirements.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8080"]