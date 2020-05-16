# Use the official image as a parent image.
FROM  pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

# Set the working directory.
WORKDIR /usr/src/app

# Copy the file from your host to your current location.
COPY . .

# Run the command inside your image filesystem.
RUN apt-get update && apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev -y && pip install -r requirements.txt

# Inform Docker that the container is listening on the specified port at runtime.
EXPOSE 5000


# Run the specified command within the container.
CMD [ "python", "single_img_api.py" ]
