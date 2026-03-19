lint:
	ruff check glview/[^a]*.py

install:
	pipx uninstall glview || true
	rm -rf build dist glview.egg-info || true
	python3 setup.py sdist bdist_wheel
	pipx install --force dist/*.whl
	@glview --version | grep 'glview version.*'

release:
	pip3 install --user setuptools wheel twine
	make install
	twine upload dist/*

.PHONY: lint install release
