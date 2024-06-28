port="5002"

docker run --rm --shm-size=8G -p ${port}:${port} -it frontend \
streamlit run streamlit_frontend/serving.py --server.port=${port}