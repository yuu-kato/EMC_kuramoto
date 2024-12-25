# 24-11-11 
FROM python:3.12.4-bullseye

# パッケージの一覧更新
RUN apt-get update && apt-get install -y build-essential libomp-dev 

# 作業ディレクトリの指定
WORKDIR /app

# for Python Interactive window in VS Code
RUN python -m pip install jupyter==1.0.0

# update setuptools and pip
RUN python -m pip install setuptools==69.0.3
RUN python -m pip install pip==23.3.2

# formatter
RUN python -m pip install black==23.12.1

# libraries
RUN python -m pip install numpy==1.26.2
RUN python -m pip install matplotlib==3.8.2
RUN python -m pip install scipy==1.11.3
RUN python -m pip install seaborn
