FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

# Set the working directory to /code
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# RUN apt-get update && apt-get install -y --no-install-recommends git && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
# Set the working directory to the user's home directory
WORKDIR $HOME/app

ENV PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

COPY --chown=user . $HOME/app
# Copy the current directory contents into the container at $HOME/app setting the owner to the user
CMD ["python3", "app.py"]