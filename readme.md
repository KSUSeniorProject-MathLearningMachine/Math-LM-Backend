# MathLearningMachine
This repo holds the backend application code for MathLearningMachine.

## Contributing
For new feature development, branch of development with a named feature branch. To merge back into development, use a pull request. This way, we can get integrated unit and acceptance test reports in github (once we write them).

We'll use [Flask micro web framework](https://palletsprojects.com/p/flask/) for this application.

Other technologies may or may not include:
* __Hypothesis__ and __PyTest__ for testing
* __Pylint__ for static analysis

and who knows what else.

## Tests
The test infrastructure will use PyTest for discovery and running and Hypothesis for property and model based testing.

## Tools
VS Code is recommended for development. Install the "Remote - Containers" extension, the "Live Share" extension, and the Python extension, all by Microsoft.


### Developing in VS Code
1. Clone the repo from Github and switch to your branch.
2. Open the project folder in VS Code.
3. Before the first launch, Open the terminal and run `docker-compose build` to build the image.
4. Ctrl+Shift+P to open the command palatte (or Click the green icon at the bottom left of the window) and run "Remote-Containers: Open in Container".

The project will open in a docker container, with VS Code attached.

Your git credentials will be shared from your local host to the container, so git will work as expected and you can commit and push from inside the container.

Open the integrated terminal to run commands in the container.

## Running Application
`$ flask run` will start the flask development server and send requests to the flask application.

## Running Unit Tests
Run pytest in the top-level project directory to start unit tests.