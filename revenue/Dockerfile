FROM python:3.7-slim


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt


COPY . .
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

CMD ["gunicorn", "-k", "gevent", "-w", "1", "-b", "0.0.0.0:5000", "flask_app:app"]

