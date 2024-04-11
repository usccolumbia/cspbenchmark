import pytest

def pytest_addoption(parser):
    parser.addoption('--rtol', type=float, default=1e-05)
    parser.addoption('--atol', type=float, default=1e-08)
    parser.addoption('--create_mode', default=False, action='store_true')

@pytest.fixture
def cmd_options(request):
    return {'tolerance':{'rtol':request.config.getoption('rtol'), 'atol':request.config.getoption('atol')},
            'create_mode':request.config.getoption('create_mode')}
