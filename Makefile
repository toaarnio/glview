lint:
	flake8 glview/glview glview/*.py
#	pylint glview/glview glview/*.py

install:
	pip3 uninstall --yes glview || true
	rm -rf build dist glview.egg-info || true
	python3 setup.py sdist bdist_wheel
	pip3 install --user dist/*.whl
	@python3 -c 'import glview; print(f"Installed glview version {glview.__version__}.")'

release:
	pip3 install --user setuptools wheel twine
	make install
	twine upload dist/*

.PHONY: lint install release
