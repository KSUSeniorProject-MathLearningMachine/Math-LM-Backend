Docker will work on both Windows and Linux, you just need to install (Docker)[https://docs.docker.com/docker-for-windows/].
For better performance on Windows, install (Windows Subsystem for Linux 2)[https://docs.docker.com/docker-for-windows/wsl/]

The Dockerfile in this repo configures the image environment for our app. Docker-Compose is used as a simple way to do cross-platform building. It reduces the number of commands that have to be run manually, while avoiding platform-dependent bash or batch scripts.

Run docker-compose up --build to build and run the image with app.py.