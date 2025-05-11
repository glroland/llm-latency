install:
	pip install -r requirements.txt

run:
	cd src && python app.py ../config.csv
#	cd src && python app.py ../config.csv --show_responses
