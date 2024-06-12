env:
	conda activate yolonas
rq:
	pipreqs . --force --ignore ".conda, .venv"