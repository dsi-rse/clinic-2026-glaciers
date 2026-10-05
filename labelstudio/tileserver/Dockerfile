# Renders NISAR GCOV HDF5 frames as PNG tiles for Label Studio. See tileserver.py.
FROM python:3.12-slim

RUN pip install --no-cache-dir "h5py~=3.12" "numpy~=2.2" "pillow~=12.0"

COPY tileserver.py /app/tileserver.py

# Unbuffered, so `docker compose logs tileserver` shows progress as it happens.
ENV PYTHONUNBUFFERED=1
EXPOSE 8081
CMD ["python", "/app/tileserver.py"]
