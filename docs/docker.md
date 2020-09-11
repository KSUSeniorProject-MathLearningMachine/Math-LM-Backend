# Docker Usage

Docker will work on both Windows and Linux, you just need to install [Docker](https://docs.docker.com/docker-for-windows/).
For better performance on Windows, install [Windows Subsystem for Linux 2](https://docs.docker.com/docker-for-windows/wsl/).

The Dockerfile in this repo configures the image environment for our app. Docker-Compose is used as a simple way to do cross-platform building. It reduces the number of commands that have to be run manually, while avoiding platform-dependent bash or batch scripts.

**NOTE:** Before building or running the image, it is necessary to create the enviornment file, which contains private API keys. An [example file](/mathpix-variables.env.example) is provided with instructions.

Run `docker-compose build` to build the image, including code changes.
Run `docker-compose run --entrypoint bash flask` to launch the container and drop into a shell for development.
