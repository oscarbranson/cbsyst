.PHONY: test build upload distribute glodap_plots docs docs-serve docs-deploy

test:
	python -m unittest

build:
	python -m build

upload:
	twine upload dist/cbsyst-$$(python -c "import cbsyst; from packaging.version import Version; print(Version(cbsyst.VERSION))")*

distribute:
	make test
	make build
	make upload

glodap_plots:
	cd tests/test_data/GLODAP_data && python plot_GLODAPv2_comparison.py

docs:
	mkdocs build

docs-serve:
	mkdocs serve