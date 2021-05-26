from setuptools import setup

APP = ['gui copy.py']
OPTIONS = {
    'arg_emulation': True,
    'packages': ['certifi',]
}
setup(app=APP, setup_requires=['py2app'])