ARTIFACT ?= a
OUTPUT ?= reproduced/$(ARTIFACT)
DATA_ROOT ?=
YES ?=

.PHONY: inspect replay regenerate
inspect:
	python -m reproduce $(ARTIFACT) --mode inspect

replay:
	python -m reproduce $(ARTIFACT) --mode replay --output $(OUTPUT) $(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

regenerate:
	python -m reproduce $(ARTIFACT) --mode regenerate --output $(OUTPUT) $(if $(DATA_ROOT),--data-root $(DATA_ROOT),) $(if $(YES),--yes,)

.PHONY: inspect-% replay-% regenerate-%
inspect-%:
	python -m reproduce $* --mode inspect

replay-%:
	python -m reproduce $* --mode replay --output $(OUTPUT) $(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

regenerate-%:
	python -m reproduce $* --mode regenerate --output $(OUTPUT) $(if $(DATA_ROOT),--data-root $(DATA_ROOT),) $(if $(YES),--yes,)
