import jsonschema
from jsonschema import validate

def test_home_page(client, app):
    """A basic test to ensure the home page is functioning correctly"""
    res = client.get('/')
    assert b'Hello, World!' == res.data


def test_solve_image(client, app):
    with open('tests/imagetest.txt', 'r') as file:
        image = file.read().replace('\n', '')

    response_schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number"},
            "input_detected": {"type": "string"},
            "solved": {"type": "string"},
        },
    }

    res = client.post('/solve-image', json={
        'b64_img': image
    })

    assert True == validate_json(instance=res.get_json(), schema=response_schema)
    assert r'F(0)=x' == res.get_json()['input_detected']
    assert r'x = F{\left(0 \right)}' == res.get_json()['solved']


def test_solve_latex(client, app):
    response_schema = {
        "type": "object",
        "properties": {
            "solved": {"type": "string"},
        },
    }

    res = client.post('/solve-latex', json={
        'latex': r'F(x)=2+2'
    })

    assert True == validate_json(instance=res.get_json(), schema=response_schema)
    assert r'F{\left(x \right)} = 4' == res.get_json()['solved']


def validate_json(instance, schema):
    try:
        validate(instance, schema)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True
