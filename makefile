.PHONY: test build upload distribute glodap_plots

test:
	python -m unittest

build:
	python setup.py sdist bdist_wheel

upload:
	twine upload dist/cbsyst-$$(python setup.py --version)*

distribute:
	make test
	make build
	make upload

glodap_plots:
	cd tests/test_data/GLODAP_data && python plot_GLODAPv2_comparison.py