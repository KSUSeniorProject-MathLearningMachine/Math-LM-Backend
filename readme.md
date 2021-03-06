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

> See [Docker Usage](./docs/docker.md) for Docker-specific instructions.

`$ flask run` will start the flask development server and send requests to the flask application.

## Running Tests

To run the tests:

    $ docker-compose build
    $ docker-compose run flask bash -c "cd /app/flask/flaskapp && python -m pytest"


## Public API

### Solve Image

**URI**: `/solve-image`

**Method**: POST

**Request Body**:

    {
        "b64_img": (str) "BASE64ENCODEDIMAGE"
    }
    
**Response Body**:

    {
        "confidence": (number) CONFIDENCE_VALUE,
        "input_detected": (str) "LATEX_STRING_OF_INPUT_DETECTED",
        "solved": (str) "LATEX_STRING_OF_CAS_SOLUTION_TO_PROBLEM",
        "detections": (object) ALL_DETECTED_OBJECTS_IN_THE_IMAGE,
    }
    
**Example Response**

    {
      "confidence": 0.46090906455275005,
      "detections": [
        {
          "box": {
            "endX": "226",
            "endY": "248",
            "startX": "85",
            "startY": "89"
          },
          "confidence": "0.7222325",
          "label": "\\mathcal{F}"
        },
        {
          "box": {
            "endX": "315",
            "endY": "263",
            "startX": "263",
            "startY": "80"
          },
          "confidence": "0.6381727",
          "label": "\\pm"
        }
      ],
      "input_detected": "\\mathcal{F}\\pm",
      "solved": "F \\mathcal{math} pm "
    }

### Solve Latex

**URI**: `/solve-latex`

**Method**: POST

**Request Body**:

    {
        "latex": (str) "LATEX_TO_BE_SOLVED"
    }
    
**Response Body**:

    {
        "solved": (str) "LATEX_STRING_OF_CAS_SOLUTION_TO_PROBLEM"
    }
