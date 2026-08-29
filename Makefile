.PHONY: dev-api dev-web train eval test lint

dev-api:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-web:
	cd frontend && npm run dev

train:
	cd recsys && python -m recsys.pipelines.train --config config/default.yaml

eval:
	cd recsys && python -m recsys.pipelines.evaluate --config config/default.yaml

test:
	cd backend && pytest -q
	cd recsys && pytest -q

lint:
	ruff check backend recsys
	cd frontend && npm run lint
