FROM ufoym/deepo:py36

RUN apt-get update && apt-get install git -y
RUN mkdir -p /setvae

RUN git clone https://github.com/jw9730/setvae.git /setvae

WORKDIR /setvae

RUN pip install -r requirements.txt
RUN bash install.sh

CMD ["/bin/bash"]
