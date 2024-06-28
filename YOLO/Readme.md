
## Command

### Model Train
~~~
# mlflow_train Image
docker build -t mlflow_train -f mlflow_model/Dockerfile .

# model train
mlflow run mlflow_model -A gpus=all
~~~

### Model Serving
~~~
1. Model Choice
mlflow ui

2. serving image build
sudo docker build -t mlflow_serving --build-arg LOCAL_PATH={mlrun_path} -f mlflow_serving/Dockerfile .

3. docker run
sh shell/model_serving.sh
~~~

### Frontend Streamlit
~~~
1. docker image build
sudo docker build -t frontend -f streamlit_frontend/Dokcerfile .

2. run streamlit
sh shell/frontend.sh
~~~