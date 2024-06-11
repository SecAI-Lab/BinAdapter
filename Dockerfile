FROM pytorch/pytorch:latest

RUN mkdir /home/binadapter
WORKDIR /home/binadapter
COPY . .

RUN mkdir dataset

RUN pip install -r requirements.txt
RUN pip install scikit-learn

CMD ["/bin/bash"]
 
